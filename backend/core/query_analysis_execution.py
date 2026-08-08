"""Execution authority for model-assisted source-anchored query analysis.

The analyzer is neither an answer generator nor an execution authority.  It
may contribute a strict candidate graph; this module validates it against a
route-merged baseline and invokes the trusted compiler, which alone can add
fully covered answer tasks or optional augmentation bridges.  Scope, KB access,
coverage and proof edges remain baseline/backend owned.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Iterable, Literal, Mapping

from core.query_analysis_compiler import (
    CompiledQueryAnalysisPlan,
    compile_query_analysis_plan,
)
from core.query_analysis_contract import QueryAnalysis
from core.query_contextual_ellipsis import derive_contextual_ellipsis_analysis
from core.query_analysis_validation import (
    QueryAnalysisExecutionValidation,
    query_plan_fingerprint,
    validate_query_analysis_for_execution,
)
from core.query_semantics import ResolvedTurnSemantics
from core.rag_trace import content_fields, trace_event
from core.rag_v2.contracts import QueryPlanV2
from core.rag_v2.task_graph import RagExecutionBundle, compile_rag_execution_bundle


AnalysisExecutionMode = Literal["deterministic"]
AnalysisDecision = Literal[
    "applied",
    "observed",
    "fallback",
    "clarification",
    "skipped",
]

# ``query.plan`` describes what the deterministic planner inferred from the
# question.  Whether that plan may actually enter retrieval is a different
# contract: it depends on the exact plan/bundle pair after normalization.  Keep
# this protocol here, alongside ``ExecutionBaseline``, so API, SSE and trace
# callers cannot each invent a different name or reason for the same gate.
QUERY_EXECUTION_SCHEMA_VERSION = "rag_query_execution.v1"
QUERY_EXECUTION_TRACE_EVENT = "query.execution"
QUERY_EXECUTION_UNRESOLVED_ROLE = "query_execution"
QueryExecutionState = Literal["ready", "needs_clarification"]


def _normalised_context(
    values: Iterable[Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], ...]:
    """Copy the bounded, request-local context accepted by the analyzer."""

    copied: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in values or ():
        key = str(raw.get("candidate_key") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result_items: list[dict[str, Any]] = []
        raw_result_items = raw.get("result_items")
        if isinstance(raw_result_items, (list, tuple)):
            for ordinal, item in enumerate(raw_result_items[:20], start=1):
                if not isinstance(item, Mapping):
                    continue
                handle = str(item.get("handle") or "").strip()
                label = str(item.get("label") or "").strip()[:255]
                if (
                    handle != f"r_{key}_{ordinal:03d}"
                    or not label
                    or str(item.get("resource") or "") != "document"
                ):
                    continue
                result_items.append({
                    "handle": handle,
                    "ordinal": ordinal,
                    "resource": "document",
                    "label": label,
                    "status": str(item.get("status") or "").strip()[:32] or None,
                })
        copied_item: dict[str, Any] = {
            "candidate_key": key,
            "user_input": str(raw.get("user_input") or "")[:1200],
            "assistant_answer": str(raw.get("assistant_answer") or "")[:1200],
        }
        if result_items:
            copied_item["result_items"] = result_items
        copied.append(copied_item)
    return tuple(copied)


def _scope_fingerprint(plan: QueryPlanV2) -> str:
    """Hash only execution constraints for trace correlation without content."""

    payload = [
        {
            "id": item.id,
            "role": item.role,
            "importance": item.importance,
            "coverage_mode": item.coverage_mode,
            "depends_on_requirement_ids": list(item.depends_on_requirement_ids or ()),
            "augmentation_requirement_ids": list(
                item.augmentation_requirement_ids or ()
            ),
            # One canonical scope fingerprint covers product/version/project
            # plus their source provenance.  Scalar projections omit the
            # project boundary and would let two semantically different plans
            # share an execution correlation id.
            "scope_fingerprint": item.scope_fingerprint,
            "bridge_kind": item.bridge_kind,
        }
        for item in plan.requirements
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ExecutionBaseline:
    """Immutable authority package prepared before any analyzer invocation."""

    plan: QueryPlanV2
    execution_bundle: RagExecutionBundle
    local_surface_plan: QueryPlanV2
    contextual_plan: QueryPlanV2
    question: str
    standalone_query: str
    route_context: tuple[dict[str, Any], ...]
    deterministic_is_followup: bool
    fingerprint: str
    scope_fingerprint: str
    preparation_reason: str | None = None

    @property
    def is_runnable(self) -> bool:
        return bool(
            self.execution_bundle.mode == "ledgered"
            and self.execution_bundle.task_graph is not None
        )

    @property
    def allowed_context_turn_keys(self) -> tuple[str, ...]:
        return tuple(item["candidate_key"] for item in self.route_context)

    @property
    def clarification_question(self) -> str:
        if self.plan.needs_clarification:
            return (
                self.plan.clarification_question
                or "请补充需要查询的具体对象、范围或条件。"
            )
        if self.execution_bundle.reason == "untyped_bridge_semantics":
            return (
                "当前问题包含需要先确认的适用关系。请明确具体对象及其适用条件"
                "（例如职级、产品版本或项目范围）后再查询。"
            )
        return "请补充需要查询的具体对象、范围或条件。"

    def safe_summary(self) -> dict[str, object]:
        return {
            "fingerprint": self.fingerprint,
            "scope_fingerprint": self.scope_fingerprint,
            "preparation_reason": self.preparation_reason,
            "runnable": self.is_runnable,
            "execution_bundle": self.execution_bundle.safe_summary(),
            "local_answer_shape": self.local_surface_plan.answer_shape,
            "contextual_answer_shape": self.contextual_plan.answer_shape,
            "allowed_context_turn_count": len(self.route_context),
        }


@dataclass(frozen=True)
class QueryExecutionGate:
    """The final, immutable decision to execute or clarify a V2 plan.

    A query plan can be structurally meaningful and still be unsafe to run:
    for example, a legacy bridge may lack a typed relation.  This object makes
    that distinction explicit.  ``query_execution`` is therefore a stable
    clarification slot, while ``query_plan`` remains solely a planning
    observability phase.
    """

    baseline: ExecutionBaseline
    state: QueryExecutionState
    decision_reason: Literal[
        "execution_baseline_runnable",
        "execution_baseline_not_runnable",
    ]
    unresolved_reason: Literal["missing", "ambiguous"] | None = None

    def __post_init__(self) -> None:
        if self.state == "ready":
            if not self.baseline.is_runnable:
                raise ValueError("a non-runnable baseline cannot be execution-ready")
            if self.decision_reason != "execution_baseline_runnable":
                raise ValueError("ready execution gate has an invalid decision reason")
            if self.unresolved_reason is not None:
                raise ValueError("ready execution gate cannot have an unresolved reason")
            return
        if self.state != "needs_clarification":
            raise ValueError("unsupported query execution gate state")
        if self.baseline.is_runnable:
            raise ValueError("a runnable baseline cannot require execution clarification")
        if self.decision_reason != "execution_baseline_not_runnable":
            raise ValueError("clarification execution gate has an invalid decision reason")
        if self.unresolved_reason not in {"missing", "ambiguous"}:
            raise ValueError("clarification execution gate requires an unresolved reason")

    @property
    def dispatch_authorized(self) -> bool:
        return self.state == "ready"

    @property
    def needs_clarification(self) -> bool:
        return self.state == "needs_clarification"

    @property
    def clarification_question(self) -> str:
        return self.baseline.clarification_question

    def to_dict(self) -> dict[str, object]:
        """Return the stable SSE/API projection without implementation prose."""

        unresolved: list[dict[str, str]] = []
        if self.unresolved_reason is not None:
            unresolved.append({
                "role": QUERY_EXECUTION_UNRESOLVED_ROLE,
                "reason": self.unresolved_reason,
            })
        return {
            "schema_version": QUERY_EXECUTION_SCHEMA_VERSION,
            "state": self.state,
            "dispatch_authorized": self.dispatch_authorized,
            "decision_reason": self.decision_reason,
            "unresolved": unresolved,
        }

    def trace_summary(self) -> dict[str, object]:
        """Return the trace projection, including safe baseline diagnostics."""

        return {
            **self.to_dict(),
            "unresolved_role": (
                QUERY_EXECUTION_UNRESOLVED_ROLE
                if self.unresolved_reason is not None
                else None
            ),
            "unresolved_reason": self.unresolved_reason,
            "baseline_fingerprint": self.baseline.fingerprint,
            "baseline_scope_fingerprint": self.baseline.scope_fingerprint,
            "baseline_runnable": self.baseline.is_runnable,
            "bundle_mode": self.baseline.execution_bundle.mode,
            "baseline": self.baseline.safe_summary(),
        }


def evaluate_query_execution_gate(baseline: ExecutionBaseline) -> QueryExecutionGate:
    """Evaluate one baseline exactly once for API, SSE and trace handoff.

    ``not_ready`` means a required value or typed relationship is missing.
    No path is allowed to relabel the plan itself as clarification merely
    because this execution gate is closed.
    """

    if not isinstance(baseline, ExecutionBaseline):
        raise ValueError("baseline must be an ExecutionBaseline")
    if baseline.is_runnable:
        return QueryExecutionGate(
            baseline=baseline,
            state="ready",
            decision_reason="execution_baseline_runnable",
        )
    if baseline.execution_bundle.mode != "not_ready":
        raise ValueError("non-runnable execution baseline has an unsupported mode")
    return QueryExecutionGate(
        baseline=baseline,
        state="needs_clarification",
        decision_reason="execution_baseline_not_runnable",
        unresolved_reason="missing",
    )


def _prepare_runnable_fact_baseline(
    plan: QueryPlanV2,
) -> tuple[QueryPlanV2, str | None]:
    """Promote a route-authorized single fact from ``unknown`` deterministically.

    ``unknown`` means the local grammar could not label a shape; it does not
    necessarily mean that a ready route has no answer target.  For exactly one
    route-authorized answer with no bridge semantics, the exact resolved query
    is a safe fact task.  This is intentionally part of baseline preparation,
    not a V2 runner fallback: clarification plans and any bridge-bearing plan
    remain non-runnable and cannot retrieve.
    """

    if (
        plan.needs_clarification
        or plan.answer_shape != "unknown"
        or any(item.role == "bridge" for item in plan.requirements)
    ):
        return plan, None
    required_answers = tuple(
        item for item in plan.requirements if item.is_required_answer
    )
    original_query = str(plan.original_query or "").strip()
    if len(required_answers) != 1 or not original_query:
        return plan, None
    rewritten_requirements = tuple(
        replace(
            item,
            description=original_query[:500],
        )
        if item.is_required_answer
        else item
        for item in plan.requirements
    )
    rewritten_query = next(
        item.description
        for item in rewritten_requirements
        if item.is_required_answer
    )
    return (
        replace(
            plan,
            answer_shape="fact",
            retrieval_queries=(rewritten_query,),
            requirements=rewritten_requirements,
            confidence=min(max(float(plan.confidence), 0.6), 0.8),
            source="fallback",
            reason=(
                f"{plan.reason}; route_authorized_single_fact_baseline"
            ).strip("; ")[:500],
        ),
        "route_authorized_single_fact_baseline",
    )


def build_execution_baseline(
    *,
    plan: QueryPlanV2,
    local_surface_plan: QueryPlanV2,
    contextual_plan: QueryPlanV2,
    question: str,
    standalone_query: str,
    route_context: Iterable[Mapping[str, Any]] | None,
    deterministic_is_followup: bool,
    execution_bundle: RagExecutionBundle | None = None,
) -> ExecutionBaseline:
    """Create one exact plan/bundle pair before optional model refinement."""

    if not isinstance(plan, QueryPlanV2):
        raise ValueError("plan must be a QueryPlanV2")
    if not isinstance(local_surface_plan, QueryPlanV2):
        raise ValueError("local_surface_plan must be a QueryPlanV2")
    if not isinstance(contextual_plan, QueryPlanV2):
        raise ValueError("contextual_plan must be a QueryPlanV2")
    prepared_plan, preparation_reason = _prepare_runnable_fact_baseline(plan)
    if execution_bundle is not None and execution_bundle.plan != prepared_plan:
        # A caller that already compiled the pre-normalisation plan cannot
        # retain it: the package must always be an exact plan/graph pair.
        execution_bundle = None
    bundle = execution_bundle or compile_rag_execution_bundle(prepared_plan)
    if not isinstance(bundle, RagExecutionBundle):
        raise ValueError("execution_bundle must be a RagExecutionBundle")
    if bundle.plan != prepared_plan:
        raise ValueError("execution bundle must belong to the final execution plan")
    return ExecutionBaseline(
        plan=prepared_plan,
        execution_bundle=bundle,
        local_surface_plan=local_surface_plan,
        contextual_plan=contextual_plan,
        question=str(question or "").strip(),
        standalone_query=str(standalone_query or "").strip(),
        route_context=_normalised_context(route_context),
        deterministic_is_followup=bool(deterministic_is_followup),
        fingerprint=query_plan_fingerprint(prepared_plan),
        scope_fingerprint=_scope_fingerprint(prepared_plan),
        preparation_reason=preparation_reason,
    )


def build_execution_clarification_baseline(
    *,
    baseline: ExecutionBaseline,
    reason: str,
    clarification_question: str,
) -> ExecutionBaseline:
    """Close an execution baseline through the ordinary clarification gate.

    Semantic safety checks occasionally establish that a current request is
    explicitly context-dependent, while the selected historical source cannot
    be carried without losing a scope or condition.  That is an execution
    precondition, not a retrieval ``no_hit`` and not a reason to let a later
    model retry reinterpret the same source.  This helper creates the normal
    non-runnable V2 baseline so API/SSE callers use the same durable
    clarification flow as every other closed execution gate.
    """

    if not isinstance(baseline, ExecutionBaseline):
        raise ValueError("baseline must be an ExecutionBaseline")
    question = str(baseline.question or "").strip()
    if not question:
        raise ValueError("clarification baseline requires the original question")
    plan = QueryPlanV2(
        original_query=question,
        answer_shape="unknown",
        retrieval_queries=(),
        requirements=(),
        confidence=0.0,
        source="fallback",
        reason=str(reason or "semantic_context_clarification"),
        needs_clarification=True,
        clarification_question=clarification_question,
    )
    return build_execution_baseline(
        plan=plan,
        local_surface_plan=baseline.local_surface_plan,
        contextual_plan=baseline.contextual_plan,
        question=question,
        standalone_query=baseline.standalone_query,
        route_context=baseline.route_context,
        deterministic_is_followup=baseline.deterministic_is_followup,
    )


@dataclass(frozen=True)
class QueryAnalysisExecutionResult:
    """One fully explained execution decision for traces and API handoff."""

    mode: AnalysisExecutionMode
    decision: AnalysisDecision
    reason: str
    baseline: ExecutionBaseline
    execution_bundle: RagExecutionBundle | None
    validation: QueryAnalysisExecutionValidation | None = None
    analysis_latency_ms: int | None = None
    compilation: CompiledQueryAnalysisPlan | None = None
    semantics: ResolvedTurnSemantics | None = None

    @property
    def applied(self) -> bool:
        return self.decision == "applied" and self.execution_bundle is not None

    def safe_summary(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "decision": self.decision,
            "reason": self.reason,
            "baseline_fingerprint": self.baseline.fingerprint,
            "baseline_scope_fingerprint": self.baseline.scope_fingerprint,
            "baseline_runnable": self.baseline.is_runnable,
            "analysis_latency_ms": self.analysis_latency_ms,
            "validation": self.validation.safe_summary()
            if self.validation is not None
            else None,
            "execution_bundle": self.execution_bundle.safe_summary()
            if self.execution_bundle is not None
            else None,
            "compilation": self.compilation.safe_summary()
            if self.compilation is not None
            else None,
            "semantics": self.semantics.safe_summary()
            if self.semantics is not None
            else None,
        }

class QueryAnalysisExecutionService:
    """Compile the bounded local contextual-ellipsis contract."""

    def _compile_analysis(
        self,
        *,
        mode: AnalysisExecutionMode,
        analysis: QueryAnalysis,
        baseline: ExecutionBaseline,
        trace_id: str,
        conversation_id: str,
        user_id: str,
        analysis_latency_ms: int | None,
        applied_reason: str | None = None,
    ) -> QueryAnalysisExecutionResult:
        """Validate and compile any source-anchored analysis through one path.

        Both model-produced and deterministic contextual candidates must cross
        this exact boundary.  The latter is not granted a privileged plan: it
        still receives the same baseline fingerprint, context-key allow-list,
        compiler and task-graph checks as the former.
        """

        validation = validate_query_analysis_for_execution(
            analysis,
            baseline_plan=baseline.plan,
            current_question=baseline.question,
            deterministic_is_followup=baseline.deterministic_is_followup,
            allowed_context_turn_keys=baseline.allowed_context_turn_keys,
        )
        trace_event(
            "query.analysis.execution_validated",
            trace_id=trace_id,
            conversation_id=conversation_id,
            user_id=user_id,
            mode=mode,
            applied=False,
            baseline=baseline.safe_summary(),
            validation=validation.safe_summary(),
        )
        if not validation.accepted:
            result = QueryAnalysisExecutionResult(
                mode=mode,
                decision="fallback",
                reason="execution_validation_rejected",
                baseline=baseline,
                execution_bundle=baseline.execution_bundle,
                validation=validation,
                analysis_latency_ms=analysis_latency_ms,
            )
            self._trace_decision(
                result,
                trace_id=trace_id,
                conversation_id=conversation_id,
                user_id=user_id,
            )
            return result
        compilation = compile_query_analysis_plan(
            analysis,
            execution_validation=validation,
            baseline_plan=baseline.plan,
            current_question=baseline.question,
            baseline_execution_bundle=baseline.execution_bundle,
        )
        result = QueryAnalysisExecutionResult(
            mode=mode,
            decision="applied",
            reason=applied_reason or compilation.compiler_decision,
            baseline=baseline,
            execution_bundle=(
                compilation.execution_bundle
            ),
            validation=validation,
            analysis_latency_ms=analysis_latency_ms,
            compilation=compilation,
            semantics=compilation.semantics,
        )
        trace_event(
            "query.analysis.compiled",
            trace_id=trace_id,
            conversation_id=conversation_id,
            user_id=user_id,
            mode=mode,
            baseline=baseline.safe_summary(),
            compilation=compilation.safe_summary(),
            compiler_decision=compilation.compiler_decision,
            baseline_plan_fingerprint=compilation.baseline_fingerprint,
            applied_plan_fingerprint=compilation.applied_plan_fingerprint,
            baseline_anchor_preserved=compilation.baseline_anchor_preserved,
            **content_fields(
                "query_analysis_execution_plan",
                json.dumps(
                    compilation.plan.to_dict(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            ),
            **content_fields(
                "resolved_turn_semantics",
                json.dumps(
                    compilation.semantics.to_dict(),
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

    def _trace_decision(
        self,
        result: QueryAnalysisExecutionResult,
        *,
        trace_id: str,
        conversation_id: str,
        user_id: str,
    ) -> None:
        trace_event(
            "query.analysis.execution_decision",
            trace_id=trace_id,
            conversation_id=conversation_id,
            user_id=user_id,
            **result.safe_summary(),
        )

    async def run_deterministic_contextual_ellipsis(
        self,
        *,
        baseline: ExecutionBaseline,
        trace_id: str,
        conversation_id: str,
        user_id: str,
    ) -> QueryAnalysisExecutionResult:
        """Try the narrow no-model contextual-ellipsis semantic contract.

        This method deliberately has no capacity lease and no model timeout:
        it is a bounded local parse.  A non-match is not an error and retains
        the exact baseline bundle.  A match still passes through the ordinary
        execution validator/compiler before it can affect retrieval.
        """

        if not baseline.is_runnable:
            result = QueryAnalysisExecutionResult(
                mode="deterministic",
                decision="skipped",
                reason="baseline_not_runnable",
                baseline=baseline,
                execution_bundle=None,
            )
            self._trace_decision(
                result,
                trace_id=trace_id,
                conversation_id=conversation_id,
                user_id=user_id,
            )
            return result

        derivation = derive_contextual_ellipsis_analysis(
            current_question=baseline.question,
            route_context=baseline.route_context,
        )
        trace_event(
            "query.analysis.deterministic_contextual_ellipsis",
            trace_id=trace_id,
            conversation_id=conversation_id,
            user_id=user_id,
            mode="deterministic",
            baseline=baseline.safe_summary(),
            **derivation.safe_summary(),
            **content_fields(
                "deterministic_contextual_ellipsis_source_refs",
                (
                    json.dumps(
                        derivation.content_summary(),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    if derivation.content_summary() is not None
                    else ""
                ),
            ),
        )
        if derivation.analysis is None:
            result = QueryAnalysisExecutionResult(
                mode="deterministic",
                decision="skipped",
                reason=derivation.reason,
                baseline=baseline,
                execution_bundle=baseline.execution_bundle,
            )
            self._trace_decision(
                result,
                trace_id=trace_id,
                conversation_id=conversation_id,
                user_id=user_id,
            )
            return result
        try:
            return self._compile_analysis(
                mode="deterministic",
                analysis=derivation.analysis,
                baseline=baseline,
                trace_id=trace_id,
                conversation_id=conversation_id,
                user_id=user_id,
                analysis_latency_ms=0,
                applied_reason="deterministic_contextual_ellipsis_applied",
            )
        except Exception as exc:
            result = QueryAnalysisExecutionResult(
                mode="deterministic",
                decision="fallback",
                reason="deterministic_contextual_ellipsis_execution_failed",
                baseline=baseline,
                execution_bundle=baseline.execution_bundle,
            )
            trace_event(
                "query.analysis.execution_decision",
                trace_id=trace_id,
                conversation_id=conversation_id,
                user_id=user_id,
                mode="deterministic",
                decision=result.decision,
                reason=result.reason,
                baseline=result.baseline.safe_summary(),
                derivation_reason=derivation.reason,
                error=exc,
            )
            return result

_SERVICE = QueryAnalysisExecutionService()


def get_query_analysis_execution_service() -> QueryAnalysisExecutionService:
    return _SERVICE


__all__ = [
    "ExecutionBaseline",
    "QUERY_EXECUTION_SCHEMA_VERSION",
    "QUERY_EXECUTION_TRACE_EVENT",
    "QUERY_EXECUTION_UNRESOLVED_ROLE",
    "QueryAnalysisExecutionResult",
    "QueryAnalysisExecutionService",
    "QueryExecutionGate",
    "build_execution_baseline",
    "build_execution_clarification_baseline",
    "evaluate_query_execution_gate",
    "get_query_analysis_execution_service",
]
