"""Execution coordinator for trusted ``query_understanding.v3``.

The V3 model is deliberately only a catalog-span selector.  This module is
the single handoff between that selector and the existing V2 evidence/task
graph pipeline:

* a local plan is retained as a *fallback floor*, not a second semantic
  authority;
* only the V3 compiler may replace that floor, and only before the request's
  revision fence is sealed for retrieval;
* route/RBAC/scope hard guards are supplied by the API and cannot be
  overridden by a model candidate; and
* both accepted and rejected outcomes emit stable trace events so the admin
  call chain can show the model selection, trusted compilation and final
  execution decision separately.

No database or retrieval work happens here.  In particular, this coordinator
never grants KB access, derives a fact, or creates proof edges.  It produces
the existing immutable ``ExecutionBaseline``/``RagExecutionBundle`` pair that
the V2 runner already requires.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping

from core.query_analysis_execution import (
    ExecutionBaseline,
    QueryExecutionGate,
    build_execution_baseline,
    build_execution_clarification_baseline,
    evaluate_query_execution_gate,
)
from core.query_contextual_ellipsis import (
    ContextualEllipsisSourceSelection,
    contextual_ellipsis_clarification_question,
    contextual_ellipsis_requires_clarification,
    derive_contextual_ellipsis_source_selection,
)
from core.query_understanding_v3_analyzer import (
    QueryUnderstandingV3RunResult,
    analyze_query_understanding,
)
from core.query_understanding_v3_catalog import (
    SourceSpanCatalogError,
    build_source_span_catalog,
)
from core.query_understanding_v3_compiler import (
    BaselineFloor,
    CompiledQueryUnderstanding,
    QueryUnderstandingV3ExecutionValidation,
    compile_query_understanding,
)
from core.query_understanding_v3_deterministic import (
    bind_deterministic_v3_contextual_ellipsis,
)
from core.rag_trace import content_fields, trace_event
from core.rag_v2.task_graph import RagExecutionBundle


QUERY_UNDERSTANDING_V3_EXECUTION_SCHEMA_VERSION = "rag_query_understanding_execution.v1"

V3ExecutionDecision = Literal["applied", "fallback", "clarification", "skipped"]
V3ExecutionProducer = Literal["model", "deterministic_contextual", "fallback"]


def _stable_fingerprint(value: object) -> str:
    """Hash one request-bound value without retaining its business content."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class QueryUnderstandingV3FenceIdentity:
    """Immutable identity of the only request allowed to adopt V3 output.

    The model and anchor prefetch both run concurrently.  Their completion is
    therefore not enough to mutate execution state: the adoption barrier
    compares this identity against the current request state first.  It binds
    all objects that could otherwise make an old/parallel result unsafe while
    retaining only hashes for content-bearing values.
    """

    conversation_id: str
    turn_id: str
    request_id: str
    question_sha256: str
    kb_scope_sha256: str
    evidence_scope_sha256: str
    route_context_sha256: str
    route_state_revision: int
    task_contract_sha256: str
    schema_version: str = "rag_query_understanding_fence.v1"

    def __post_init__(self) -> None:
        for field_name in (
            "conversation_id",
            "turn_id",
            "request_id",
            "question_sha256",
            "kb_scope_sha256",
            "evidence_scope_sha256",
            "route_context_sha256",
            "task_contract_sha256",
        ):
            value = str(getattr(self, field_name) or "").strip()
            if not value:
                raise ValueError(f"V3 revision fence requires {field_name}")
            object.__setattr__(self, field_name, value)
        if isinstance(self.route_state_revision, bool):
            raise ValueError("V3 revision fence route_state_revision must be numeric")
        try:
            revision = int(self.route_state_revision)
        except (TypeError, ValueError) as exc:
            raise ValueError("V3 revision fence route_state_revision must be numeric") from exc
        if revision < 0:
            raise ValueError("V3 revision fence route_state_revision must not be negative")
        object.__setattr__(self, "route_state_revision", revision)

    @property
    def fingerprint(self) -> str:
        return _stable_fingerprint({
            "schema_version": self.schema_version,
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "request_id": self.request_id,
            "question_sha256": self.question_sha256,
            "kb_scope_sha256": self.kb_scope_sha256,
            "evidence_scope_sha256": self.evidence_scope_sha256,
            "route_context_sha256": self.route_context_sha256,
            "route_state_revision": self.route_state_revision,
            "task_contract_sha256": self.task_contract_sha256,
        })

    def safe_summary(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "fingerprint": self.fingerprint,
            "conversation_bound": bool(self.conversation_id),
            "turn_bound": self.turn_id != "legacy-no-turn",
            "request_bound": bool(self.request_id),
            "route_state_revision": self.route_state_revision,
        }


def build_query_understanding_v3_fence_identity(
    *,
    conversation_id: object,
    turn_id: object | None,
    request_id: object,
    question: object,
    kb_ids: Iterable[object],
    evidence_scope_filter: object,
    route_context: Iterable[Mapping[str, Any]] | None,
    route_state_revision: object,
    task_contract: object,
) -> QueryUnderstandingV3FenceIdentity:
    """Create the fence identity from current API-owned request state."""

    conversation_value = str(conversation_id or "").strip()
    request_value = str(request_id or "").strip()
    question_value = str(question or "")
    if not conversation_value or not request_value or not question_value.strip():
        raise ValueError("V3 revision fence requires conversation, request and question")
    kb_scope = tuple(sorted({str(item) for item in kb_ids if str(item).strip()}))
    if not kb_scope:
        raise ValueError("V3 revision fence requires an authorised KB scope")
    return QueryUnderstandingV3FenceIdentity(
        conversation_id=conversation_value,
        turn_id=str(turn_id or "legacy-no-turn").strip(),
        request_id=request_value,
        question_sha256=_stable_fingerprint(question_value),
        kb_scope_sha256=_stable_fingerprint(kb_scope),
        evidence_scope_sha256=_stable_fingerprint(evidence_scope_filter),
        route_context_sha256=_stable_fingerprint(tuple(dict(item) for item in (route_context or ()))),
        route_state_revision=route_state_revision,
        task_contract_sha256=_stable_fingerprint(task_contract),
    )


@dataclass(frozen=True)
class QueryUnderstandingV3ContextSelection:
    """Trusted projection of only catalog-authorised historical turn keys."""

    current_question: str
    selected_context_turn_keys: tuple[str, ...]
    schema_version: str = "rag_query_understanding_context_selection.v1"

    def __post_init__(self) -> None:
        question = str(self.current_question or "").strip()
        if not question:
            raise ValueError("V3 context selection requires the current question")
        selected = tuple(str(item or "").strip() for item in self.selected_context_turn_keys)
        if any(not item for item in selected):
            raise ValueError("V3 context selection contains an empty turn key")
        if len(selected) > 3 or len(selected) != len(set(selected)):
            raise ValueError("V3 context selection keys are invalid")
        object.__setattr__(self, "current_question", question)
        object.__setattr__(self, "selected_context_turn_keys", selected)

    @property
    def self_contained(self) -> bool:
        return not self.selected_context_turn_keys

    def safe_summary(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "self_contained": self.self_contained,
            "context_turn_count": len(self.selected_context_turn_keys),
        }


class _ActiveCapacity:
    """Event-loop-local non-queueing capacity reservation.

    We intentionally reject surplus model work rather than making a browser
    wait in an invisible queue.  There is no await point in reservation, so a
    single event loop cannot overbook the last available slot.
    """

    def __init__(self) -> None:
        self._inflight = 0

    @property
    def inflight(self) -> int:
        return self._inflight

    def try_acquire(self, maximum: int) -> bool:
        bounded = max(0, int(maximum))
        if bounded < 1 or self._inflight >= bounded:
            return False
        self._inflight += 1
        return True

    def release(self) -> None:
        if self._inflight <= 0:
            raise RuntimeError("V3 query-understanding capacity released without lease")
        self._inflight -= 1


_ACTIVE_CAPACITY = _ActiveCapacity()


@dataclass(frozen=True)
class QueryUnderstandingV3Baseline:
    """Trusted V3 preflight built before any model invocation.

    ``fallback`` is an exact V2 plan/bundle pair.  ``floor`` deliberately
    rebuilds scope partitions from the current source question; it does not
    accept scope data from the local plan, route model or V3 candidate.
    """

    fallback: ExecutionBaseline
    floor: BaselineFloor
    # The strict local grammar is also a preflight safety policy.  A visible
    # ``那/那么…呢`` reference whose immediate antecedent has scope/conditions
    # cannot be downgraded to the bare current phrase merely because the V3
    # model is unavailable or chooses a different catalog span.
    contextual_ellipsis_selection: ContextualEllipsisSourceSelection | None = None
    contextual_ellipsis_clarification_required: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.fallback, ExecutionBaseline):
            raise ValueError("V3 baseline requires an ExecutionBaseline fallback")
        if not isinstance(self.floor, BaselineFloor):
            raise ValueError("V3 baseline requires a BaselineFloor")
        if self.floor.current_question != self.fallback.question:
            raise ValueError("V3 baseline question must match fallback question")
        if self.floor.fallback_plan != self.fallback.plan:
            raise ValueError("V3 floor must carry the exact fallback plan")
        if self.contextual_ellipsis_selection is not None and not isinstance(
            self.contextual_ellipsis_selection,
            ContextualEllipsisSourceSelection,
        ):
            raise ValueError("V3 baseline contextual selection is invalid")
        if self.contextual_ellipsis_clarification_required:
            if self.contextual_ellipsis_selection is None:
                raise ValueError("V3 contextual clarification requires a selection")
            if not contextual_ellipsis_requires_clarification(
                self.contextual_ellipsis_selection
            ):
                raise ValueError("V3 contextual clarification has no blocking reason")
            if not self.fallback.plan.needs_clarification:
                raise ValueError("V3 contextual clarification requires a closed fallback")

    def safe_summary(self) -> dict[str, object]:
        return {
            "schema_version": QUERY_UNDERSTANDING_V3_EXECUTION_SCHEMA_VERSION,
            "fallback": self.fallback.safe_summary(),
            "floor": self.floor.safe_summary(),
            "contextual_ellipsis": (
                self.contextual_ellipsis_selection.safe_summary()
                if self.contextual_ellipsis_selection is not None
                else None
            ),
            "contextual_ellipsis_clarification_required": (
                self.contextual_ellipsis_clarification_required
            ),
        }

    @property
    def execution_preflight_blocked(self) -> bool:
        """Whether V3 must not call a model before the standard clarification."""

        return self.contextual_ellipsis_clarification_required


def build_query_understanding_v3_baseline(
    *,
    fallback: ExecutionBaseline,
    hard_clarification_reason: str | None = None,
) -> QueryUnderstandingV3Baseline:
    """Create a V3 baseline without elevating the local planner to authority."""

    if not isinstance(fallback, ExecutionBaseline):
        raise ValueError("fallback must be an ExecutionBaseline")
    selection: ContextualEllipsisSourceSelection | None = None
    try:
        selection = derive_contextual_ellipsis_source_selection(
            current_question=fallback.question,
            route_context=fallback.route_context,
        )
    except Exception:
        # The coordinator separately traces unexpected local-grammar failures
        # and retains the ordinary fail-closed V3/model path.  Baseline
        # construction must not turn an observability/preflight error into a
        # request failure.
        selection = None
    contextual_clarification_required = bool(
        selection is not None
        and contextual_ellipsis_requires_clarification(selection)
    )
    selected_fallback = fallback
    selected_guard = hard_clarification_reason
    if contextual_clarification_required and selection is not None:
        selected_fallback = build_execution_clarification_baseline(
            baseline=fallback,
            reason="contextual_ellipsis_history_not_inheritable",
            clarification_question=contextual_ellipsis_clarification_question(selection),
        )
        selected_guard = (
            hard_clarification_reason
            or "contextual_ellipsis_history_not_inheritable"
        )
    return QueryUnderstandingV3Baseline(
        fallback=selected_fallback,
        floor=BaselineFloor(
            current_question=selected_fallback.question,
            fallback_plan=selected_fallback.plan,
            hard_clarification_reason=selected_guard,
        ),
        contextual_ellipsis_selection=selection,
        contextual_ellipsis_clarification_required=contextual_clarification_required,
    )


@dataclass(frozen=True)
class QueryUnderstandingV3ExecutionResult:
    """One fully explained selection of a final plan/bundle for this request."""

    decision: V3ExecutionDecision
    reason: str
    request_baseline: QueryUnderstandingV3Baseline
    selected_baseline: ExecutionBaseline
    query_execution_gate: QueryExecutionGate
    analysis_result: QueryUnderstandingV3RunResult | None = None
    validation: QueryUnderstandingV3ExecutionValidation | None = None
    compilation: CompiledQueryUnderstanding | None = None
    context_selection: QueryUnderstandingV3ContextSelection | None = None
    # ``producer`` is deliberately separate from ``decision``.  A strict
    # source-grammar producer is allowed to emit the same V3 contract as the
    # model, but it must remain visible in traces so an operator never mistakes
    # a deterministic selection for a model completion (or vice versa).
    producer: V3ExecutionProducer = "model"
    deterministic_selection: ContextualEllipsisSourceSelection | None = None

    def __post_init__(self) -> None:
        if self.decision not in {"applied", "fallback", "clarification", "skipped"}:
            raise ValueError("unsupported V3 execution decision")
        if self.producer not in {"model", "deterministic_contextual", "fallback"}:
            raise ValueError("unsupported V3 execution producer")
        if not isinstance(self.request_baseline, QueryUnderstandingV3Baseline):
            raise ValueError("V3 execution result requires a request baseline")
        if not isinstance(self.selected_baseline, ExecutionBaseline):
            raise ValueError("V3 execution result requires a selected baseline")
        if not isinstance(self.query_execution_gate, QueryExecutionGate):
            raise ValueError("V3 execution result requires a query execution gate")
        if self.query_execution_gate.baseline != self.selected_baseline:
            raise ValueError("V3 execution gate must belong to selected baseline")
        if self.decision == "applied":
            if self.compilation is None or self.compilation.used_fallback:
                raise ValueError("applied V3 result requires an accepted compilation")
            if self.query_execution_gate.needs_clarification:
                raise ValueError("applied V3 result cannot close execution gate")
            if self.context_selection is None:
                raise ValueError("applied V3 result requires a context selection")
            if self.context_selection.current_question != self.request_baseline.fallback.question:
                raise ValueError("V3 context selection does not match the request question")
            if (
                self.analysis_result is None
                or self.analysis_result.analysis is None
                or self.context_selection.selected_context_turn_keys
                != self.analysis_result.analysis.referenced_context_keys
            ):
                raise ValueError("V3 context selection does not match catalog analysis")
        elif self.context_selection is not None:
            raise ValueError("only an applied V3 result may carry context selection")
        if self.producer == "deterministic_contextual":
            if (
                not isinstance(
                    self.deterministic_selection,
                    ContextualEllipsisSourceSelection,
                )
                or not self.deterministic_selection.selected
            ):
                raise ValueError(
                    "deterministic V3 result requires a selected source contract"
                )
        elif self.deterministic_selection is not None:
            raise ValueError(
                "only a deterministic V3 result may carry a source selection"
            )

    @property
    def execution_bundle(self) -> RagExecutionBundle:
        return self.selected_baseline.execution_bundle

    @property
    def applied(self) -> bool:
        return self.decision == "applied"

    def safe_summary(self) -> dict[str, object]:
        return {
            "schema_version": QUERY_UNDERSTANDING_V3_EXECUTION_SCHEMA_VERSION,
            "decision": self.decision,
            "reason": self.reason,
            "producer": self.producer,
            "selected_execution": self.query_execution_gate.trace_summary(),
            "analysis": (
                self.analysis_result.safe_summary()
                if self.analysis_result is not None
                else None
            ),
            "validation": (
                self.validation.safe_summary()
                if self.validation is not None
                else None
            ),
            "compilation": (
                self.compilation.safe_summary()
                if self.compilation is not None
                else None
            ),
            "context_selection": (
                self.context_selection.safe_summary()
                if self.context_selection is not None
                else None
            ),
            "deterministic_selection": (
                self.deterministic_selection.safe_summary()
                if self.deterministic_selection is not None
                else None
            ),
        }


@dataclass
class QueryUnderstandingV3RevisionFence:
    """Request-local adoption fence between asynchronous understanding and retrieval.

    The API can begin a safe immutable-anchor retrieval while V3 analysis is
    running.  A candidate may alter the final plan only while this fence is
    open.  Once the final task graph is handed to V2, late completions are
    recorded but cannot mutate the ledger, source scope or generation inputs.
    """

    baseline_fingerprint: str
    identity: QueryUnderstandingV3FenceIdentity
    revision: int = 0
    sealed: bool = False
    selected_reason: str = "fallback_baseline"
    last_rejection_reason: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        fingerprint = str(self.baseline_fingerprint or "").strip()
        if not fingerprint:
            raise ValueError("V3 revision fence requires a baseline fingerprint")
        if not isinstance(self.identity, QueryUnderstandingV3FenceIdentity):
            raise ValueError("V3 revision fence requires an immutable identity")
        self.baseline_fingerprint = fingerprint

    def adoption_reason(
        self,
        result: QueryUnderstandingV3ExecutionResult,
        *,
        observed_identity: QueryUnderstandingV3FenceIdentity,
    ) -> str:
        """Return the exact reason an asynchronous decision may not be adopted."""

        if not isinstance(result, QueryUnderstandingV3ExecutionResult):
            raise ValueError("revision fence requires a V3 execution result")
        if not isinstance(observed_identity, QueryUnderstandingV3FenceIdentity):
            raise ValueError("revision fence requires an observed identity")
        if self.sealed:
            return "fence_sealed"
        if result.request_baseline.fallback.fingerprint != self.baseline_fingerprint:
            return "baseline_fingerprint_mismatch"
        if observed_identity.fingerprint != self.identity.fingerprint:
            return "request_identity_mismatch"
        return "adoptable"

    def adopt(
        self,
        result: QueryUnderstandingV3ExecutionResult,
        *,
        observed_identity: QueryUnderstandingV3FenceIdentity,
    ) -> bool:
        reason = self.adoption_reason(result, observed_identity=observed_identity)
        if reason != "adoptable":
            self.last_rejection_reason = reason
            return False
        self.revision += 1
        self.selected_reason = result.reason
        self.last_rejection_reason = None
        return True

    def seal(self) -> int:
        self.sealed = True
        return self.revision

    def safe_summary(self) -> dict[str, object]:
        return {
            "schema_version": QUERY_UNDERSTANDING_V3_EXECUTION_SCHEMA_VERSION,
            "revision": self.revision,
            "sealed": self.sealed,
            "selected_reason": self.selected_reason,
            "last_rejection_reason": self.last_rejection_reason,
            "identity": self.identity.safe_summary(),
        }


def _selected_baseline(
    *,
    request_baseline: QueryUnderstandingV3Baseline,
    compilation: CompiledQueryUnderstanding,
) -> ExecutionBaseline:
    """Rewrap a V3 compiled plan in the existing immutable execution package."""

    fallback = request_baseline.fallback
    return build_execution_baseline(
        plan=compilation.plan,
        local_surface_plan=fallback.local_surface_plan,
        contextual_plan=fallback.contextual_plan,
        question=fallback.question,
        standalone_query=fallback.standalone_query,
        route_context=fallback.route_context,
        deterministic_is_followup=fallback.deterministic_is_followup,
        execution_bundle=compilation.execution_bundle,
    )


def _fallback_result(
    *,
    request_baseline: QueryUnderstandingV3Baseline,
    reason: str,
    analysis_result: QueryUnderstandingV3RunResult | None = None,
    validation: QueryUnderstandingV3ExecutionValidation | None = None,
    compilation: CompiledQueryUnderstanding | None = None,
    producer: V3ExecutionProducer = "fallback",
    deterministic_selection: ContextualEllipsisSourceSelection | None = None,
) -> QueryUnderstandingV3ExecutionResult:
    selected = request_baseline.fallback
    gate = evaluate_query_execution_gate(selected)
    decision: V3ExecutionDecision = (
        "clarification" if gate.needs_clarification else "fallback"
    )
    return QueryUnderstandingV3ExecutionResult(
        decision=decision,
        reason=reason,
        request_baseline=request_baseline,
        selected_baseline=selected,
        query_execution_gate=gate,
        analysis_result=analysis_result,
        validation=validation,
        compilation=compilation,
        producer=producer,
        deterministic_selection=deterministic_selection,
    )


def _contextual_clarification_result(
    *,
    request_baseline: QueryUnderstandingV3Baseline,
    reason: str,
    clarification_question: str,
    analysis_result: QueryUnderstandingV3RunResult | None = None,
    validation: QueryUnderstandingV3ExecutionValidation | None = None,
    compilation: CompiledQueryUnderstanding | None = None,
    producer: V3ExecutionProducer = "model",
    deterministic_selection: ContextualEllipsisSourceSelection | None = None,
) -> QueryUnderstandingV3ExecutionResult:
    """Turn a trusted historical-context rejection into a closed V2 gate.

    A model timeout is allowed to retain the current-turn floor.  A validated
    source-envelope rejection is not: the model selected history, and the
    compiler has proven that history cannot be inherited without discarding a
    scope, condition or source boundary.  Build the same standard
    ``ExecutionBaseline`` used by all other clarification paths, so callers
    cannot accidentally dispatch the old runnable floor after this result.
    """

    selected = build_execution_clarification_baseline(
        baseline=request_baseline.fallback,
        reason=reason,
        clarification_question=clarification_question,
    )
    gate = evaluate_query_execution_gate(selected)
    if not gate.needs_clarification:  # pragma: no cover - contract guard
        raise ValueError("contextual clarification baseline must close execution")
    return QueryUnderstandingV3ExecutionResult(
        decision="clarification",
        reason=reason,
        request_baseline=request_baseline,
        selected_baseline=selected,
        query_execution_gate=gate,
        analysis_result=analysis_result,
        validation=validation,
        compilation=compilation,
        producer=producer,
        deterministic_selection=deterministic_selection,
    )


def _historical_context_clarification_question() -> str:
    """Stable, content-free user prompt for an unsafe V3 history selection."""

    return (
        "当前问题需要沿用上一轮的对象、适用范围或条件，但这些条件未在本轮完整明确。"
        "请在本轮说明要查询的对象，以及产品/版本、项目、地点或其他适用条件。"
    )


class QueryUnderstandingV3ExecutionService:
    """Run, validate and compile one bounded V3 understanding attempt."""

    async def run_active(
        self,
        *,
        baseline: QueryUnderstandingV3Baseline,
        trace_id: str,
        conversation_id: str,
        user_id: str,
        timeout_seconds: float,
        maximum_inflight: int,
    ) -> QueryUnderstandingV3ExecutionResult:
        if not isinstance(baseline, QueryUnderstandingV3Baseline):
            raise ValueError("baseline must be a QueryUnderstandingV3Baseline")

        if baseline.execution_preflight_blocked:
            selection = baseline.contextual_ellipsis_selection
            trace_event(
                "query.understanding.v3.deterministic_contextual_ellipsis",
                trace_id=trace_id,
                conversation_id=conversation_id,
                user_id=user_id,
                mode="active",
                status="preflight_clarification",
                source_selection=(
                    selection.safe_summary() if selection is not None else None
                ),
            )
            result = _fallback_result(
                request_baseline=baseline,
                reason="contextual_ellipsis_history_not_inheritable",
                producer="fallback",
            )
            self._trace_decision(
                result,
                trace_id=trace_id,
                conversation_id=conversation_id,
                user_id=user_id,
            )
            return result

        # The narrow deterministic producer is evaluated before a model lease.
        # It does not create a query or plan; it merely proves two exact source
        # ranges.  A selected pair must still bind to the request catalog and
        # cross the ordinary V3 compiler below.  This makes a V3 timeout unable
        # to erase a proven antecedent, without restoring the legacy V2
        # analyzer as a semantic authority.
        source_selection: ContextualEllipsisSourceSelection | None = (
            baseline.contextual_ellipsis_selection
        )
        analysis_result: QueryUnderstandingV3RunResult | None = None
        producer: V3ExecutionProducer = "model"
        if source_selection is None:
            try:
                source_selection = derive_contextual_ellipsis_source_selection(
                    current_question=baseline.fallback.question,
                    route_context=baseline.fallback.route_context,
                )
            except Exception as exc:
                # A local grammar regression is not an execution grant.  Preserve
                # the ordinary model/baseline behaviour and record the anomaly;
                # no historical source may be approximated from this failure.
                trace_event(
                    "query.understanding.v3.deterministic_contextual_ellipsis",
                    trace_id=trace_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    mode="active",
                    status="selection_failed",
                    reason="deterministic_source_selection_failed",
                    error=exc,
                )
        if source_selection is not None:
            trace_event(
                "query.understanding.v3.deterministic_contextual_ellipsis",
                trace_id=trace_id,
                conversation_id=conversation_id,
                user_id=user_id,
                mode="active",
                status=("selected" if source_selection.selected else "skipped"),
                source_selection=source_selection.safe_summary(),
            )

        if source_selection is not None and source_selection.selected:
            try:
                deterministic_catalog = build_source_span_catalog(
                    current_question=baseline.fallback.question,
                    route_context=baseline.fallback.route_context,
                )
                deterministic = bind_deterministic_v3_contextual_ellipsis(
                    catalog=deterministic_catalog,
                    source_selection=source_selection,
                )
            except (SourceSpanCatalogError, ValueError) as exc:
                result = _fallback_result(
                    request_baseline=baseline,
                    reason="deterministic_catalog_binding_failed",
                    producer="deterministic_contextual",
                    deterministic_selection=source_selection,
                )
                trace_event(
                    "query.understanding.v3.deterministic_contextual_ellipsis",
                    trace_id=trace_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    mode="active",
                    status="binding_failed",
                    source_selection=source_selection.safe_summary(),
                    error=exc,
                )
                self._trace_decision(
                    result,
                    trace_id=trace_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                )
                return result
            trace_event(
                "query.understanding.v3.deterministic_contextual_ellipsis",
                trace_id=trace_id,
                conversation_id=conversation_id,
                user_id=user_id,
                mode="active",
                status=("bound" if deterministic.applied else "binding_rejected"),
                catalog_summary=deterministic_catalog.safe_summary(),
                **deterministic.safe_summary(),
            )
            if deterministic.understanding is None:
                # The trusted source pair could not satisfy the public V3
                # selection contract.  Do not let a model choose a broader
                # historical meaning after the deterministic route rejected it.
                result = _fallback_result(
                    request_baseline=baseline,
                    reason=deterministic.reason,
                    producer="deterministic_contextual",
                    deterministic_selection=source_selection,
                )
                self._trace_decision(
                    result,
                    trace_id=trace_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                )
                return result
            analysis_result = QueryUnderstandingV3RunResult(
                mode="active",
                catalog=deterministic_catalog,
                analysis=deterministic.understanding,
                model="server_deterministic_contextual",
                latency_ms=0,
                origin="deterministic",
            )
            producer = "deterministic_contextual"

        capacity_acquired = False
        try:
            if analysis_result is None:
                if not _ACTIVE_CAPACITY.try_acquire(maximum_inflight):
                    result = _fallback_result(
                        request_baseline=baseline,
                        reason="active_capacity_exhausted",
                        producer="model",
                    )
                    self._trace_decision(
                        result,
                        trace_id=trace_id,
                        conversation_id=conversation_id,
                        user_id=user_id,
                    )
                    return result
                capacity_acquired = True
                analysis_result = await analyze_query_understanding(
                    question=baseline.fallback.question,
                    route_context=baseline.fallback.route_context,
                    trace_id=trace_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    mode="active",
                    timeout_seconds=timeout_seconds,
                )
            if analysis_result.analysis is None or analysis_result.catalog is None:
                result = _fallback_result(
                    request_baseline=baseline,
                    reason=analysis_result.fallback_reason or "analysis_unavailable",
                    analysis_result=analysis_result,
                    producer=producer,
                    deterministic_selection=(
                        source_selection
                        if producer == "deterministic_contextual"
                        else None
                    ),
                )
                self._trace_decision(
                    result,
                    trace_id=trace_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                )
                return result

            compilation = compile_query_understanding(
                catalog=analysis_result.catalog,
                understanding=analysis_result.analysis,
                baseline_floor=baseline.floor,
            )
            validation = compilation.validation
            trace_event(
                "query.understanding.v3.execution_validated",
                trace_id=trace_id,
                conversation_id=conversation_id,
                user_id=user_id,
                baseline=baseline.safe_summary(),
                validation=validation.safe_summary(),
                producer=producer,
            )
            if compilation.used_fallback:
                if validation.requires_clarification:
                    result = _contextual_clarification_result(
                        request_baseline=baseline,
                        reason="historical_context_requires_clarification",
                        clarification_question=(
                            _historical_context_clarification_question()
                        ),
                        analysis_result=analysis_result,
                        validation=validation,
                        compilation=compilation,
                        producer=producer,
                        deterministic_selection=(
                            source_selection
                            if producer == "deterministic_contextual"
                            else None
                        ),
                    )
                else:
                    result = _fallback_result(
                        request_baseline=baseline,
                        reason="execution_validation_rejected",
                        analysis_result=analysis_result,
                        validation=validation,
                        compilation=compilation,
                        producer=producer,
                        deterministic_selection=(
                            source_selection
                            if producer == "deterministic_contextual"
                            else None
                        ),
                    )
                self._trace_decision(
                    result,
                    trace_id=trace_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                )
                return result

            selected = _selected_baseline(
                request_baseline=baseline,
                compilation=compilation,
            )
            gate = evaluate_query_execution_gate(selected)
            if gate.needs_clarification:
                # This should be unreachable for a successfully compiled V3
                # plan, but retain the fail-closed guard if a future compiler
                # changes its output contract.
                result = QueryUnderstandingV3ExecutionResult(
                    decision="clarification",
                    reason="compiled_execution_gate_closed",
                    request_baseline=baseline,
                    selected_baseline=selected,
                    query_execution_gate=gate,
                    analysis_result=analysis_result,
                    validation=validation,
                    compilation=compilation,
                    producer=producer,
                    deterministic_selection=(
                        source_selection
                        if producer == "deterministic_contextual"
                        else None
                    ),
                )
                self._trace_decision(
                    result,
                    trace_id=trace_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                )
                return result

            result = QueryUnderstandingV3ExecutionResult(
                decision="applied",
                reason="catalog_bound_candidate_compiled",
                request_baseline=baseline,
                selected_baseline=selected,
                query_execution_gate=gate,
                analysis_result=analysis_result,
                validation=validation,
                compilation=compilation,
                context_selection=QueryUnderstandingV3ContextSelection(
                    current_question=baseline.fallback.question,
                    selected_context_turn_keys=(
                        analysis_result.analysis.referenced_context_keys
                    ),
                ),
                producer=producer,
                deterministic_selection=(
                    source_selection
                    if producer == "deterministic_contextual"
                    else None
                ),
            )
            trace_event(
                "query.understanding.v3.compiled",
                trace_id=trace_id,
                conversation_id=conversation_id,
                user_id=user_id,
                compilation=compilation.safe_summary(),
                compiler_decision=compilation.compiler_decision,
                selected_execution=gate.trace_summary(),
                producer=producer,
                **content_fields(
                    "query_understanding_v3_execution_plan",
                    json.dumps(
                        compilation.plan.to_dict(),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                ),
            )
            self._trace_decision(
                result,
                trace_id=trace_id,
                conversation_id=conversation_id,
                user_id=user_id,
            )
            return result
        except asyncio.CancelledError:
            trace_event(
                "query.understanding.v3.cancelled",
                trace_id=trace_id,
                conversation_id=conversation_id,
                user_id=user_id,
            )
            raise
        except Exception as exc:
            result = _fallback_result(
                request_baseline=baseline,
                reason="understanding_execution_failed",
                producer=producer,
                deterministic_selection=(
                    source_selection
                    if producer == "deterministic_contextual"
                    else None
                ),
            )
            trace_event(
                "query.understanding.v3.execution_decision",
                trace_id=trace_id,
                conversation_id=conversation_id,
                user_id=user_id,
                error=exc,
                **result.safe_summary(),
            )
            return result
        finally:
            if capacity_acquired:
                _ACTIVE_CAPACITY.release()

    @staticmethod
    def _trace_decision(
        result: QueryUnderstandingV3ExecutionResult,
        *,
        trace_id: str,
        conversation_id: str,
        user_id: str,
    ) -> None:
        trace_event(
            "query.understanding.v3.execution_decision",
            trace_id=trace_id,
            conversation_id=conversation_id,
            user_id=user_id,
            **result.safe_summary(),
        )


_SERVICE = QueryUnderstandingV3ExecutionService()


def get_query_understanding_v3_execution_service() -> QueryUnderstandingV3ExecutionService:
    """Return the process-local V3 coordinator (it owns no DB resources)."""

    return _SERVICE


__all__ = [
    "QUERY_UNDERSTANDING_V3_EXECUTION_SCHEMA_VERSION",
    "QueryUnderstandingV3Baseline",
    "QueryUnderstandingV3ContextSelection",
    "QueryUnderstandingV3ExecutionResult",
    "QueryUnderstandingV3ExecutionService",
    "QueryUnderstandingV3FenceIdentity",
    "QueryUnderstandingV3RevisionFence",
    "V3ExecutionDecision",
    "build_query_understanding_v3_baseline",
    "build_query_understanding_v3_fence_identity",
    "get_query_understanding_v3_execution_service",
]
