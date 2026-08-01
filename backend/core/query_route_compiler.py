"""Deterministically compile semantic RAG routes into executable contracts.

``rag_route_decision.v1`` is model output and is never execution authority.
This module binds it to an enabled category, applies local policy guards, and
produces a ``rag_task_contract.v1`` that can be checked again immediately before
the retrieval/generation pipeline is dispatched.

The light-weight category/config protocols intentionally avoid importing ORM
models or the legacy ``IntentDecision`` class.  Callers can project this contract
back to legacy fields without creating a dependency cycle.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal

from core.query_route_contract import (
    ROUTE_DECISION_SCHEMA_VERSION,
    RagRouteDecision,
    RouteClarification,
    RouteDecisionValidationError,
    RouteUnresolvedSlot,
    normalize_turn_candidate_keys,
)
from core.rag_v2.query_plan import infer_implicit_bridge


TASK_CONTRACT_SCHEMA_VERSION = "rag_task_contract.v1"

VALID_CATEGORY_ACTIONS = {"retrieve", "chat", "writing", "system_help"}
VALID_RESPONSE_MODES = {"grounded_qa", "general_chat", "writing", "platform_help"}
VALID_RETRIEVAL_POLICIES = {"required", "skip"}

# These are deterministic, high-certainty local recognizers.  They do not
# depend on the model's confidence (the model may be unavailable and return
# ``0``), so the generic low-confidence safety upgrade must not turn them into
# a knowledge-base request.  The global ``allow_general_chat`` policy still
# runs first and can intentionally override them.
_LOCAL_DIRECT_DECISION_REASONS = frozenset(
    {
        "exact_greeting",
        "explicit_platform_help",
        "inline_writing_content",
    }
)

ResponseMode = Literal["grounded_qa", "general_chat", "writing", "platform_help"]
RetrievalPolicy = Literal["required", "skip"]
RequirementImportance = Literal["required", "helpful"]
RequirementSource = Literal["explicit", "inferred"]

_SOURCE_RE = re.compile(r"^[a-z][a-z0-9_:-]{0,63}$")


class TaskContractCompilationError(ValueError):
    """Raised for invalid compiler inputs or category/config drift."""


class TaskContractDispatchError(RuntimeError):
    """Raised when a task contract does not pass the final execution gate."""


@dataclass(frozen=True)
class RouteCategoryPolicy:
    """Small immutable projection of an enabled ``IntentCategory`` row."""

    code: str
    name: str
    action: str
    enabled: bool = True


@dataclass(frozen=True)
class RouteCompilerConfig:
    """Small immutable projection of the policy-relevant router config."""

    confidence_threshold: float = 0.65
    allow_general_chat: bool = True


@dataclass(frozen=True)
class CompiledAnswerRequirement:
    id: str
    role: Literal["answer", "bridge"]
    origin: Literal["user_text", "semantically_entailed"]
    description: str
    importance: RequirementImportance
    source: RequirementSource

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "role": self.role,
            "origin": self.origin,
            "description": self.description,
            "importance": self.importance,
            "source": self.source,
        }


@dataclass(frozen=True)
class RagTaskContract:
    schema_version: Literal["rag_task_contract.v1"]
    route_schema_version: Literal["rag_route_decision.v1"]
    readiness: Literal["ready", "needs_clarification"]
    intent_code: str
    intent_name: str
    action: str
    confidence: float
    source: str
    relation: Literal["new", "followup", "correction", "continuation"]
    evidence_scope: Literal[
        "enterprise_kb",
        "current_input",
        "platform_self",
        "general_world",
        "mixed",
    ]
    query_mode: Literal["current", "contextualize"]
    context_turn_keys: tuple[str, ...]
    response_mode: ResponseMode
    retrieval_policy: RetrievalPolicy
    need_retrieval: bool
    dispatch_authorized: bool
    decision_reason: str
    selected_kb_count: int
    requirements: tuple[CompiledAnswerRequirement, ...]
    clarification: RouteClarification

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "route_schema_version": self.route_schema_version,
            "readiness": self.readiness,
            "intent_code": self.intent_code,
            "intent_name": self.intent_name,
            "action": self.action,
            "confidence": self.confidence,
            "source": self.source,
            "relation": self.relation,
            "evidence_scope": self.evidence_scope,
            "query_mode": self.query_mode,
            "context_turn_keys": list(self.context_turn_keys),
            "response_mode": self.response_mode,
            "retrieval_policy": self.retrieval_policy,
            "need_retrieval": self.need_retrieval,
            "dispatch_authorized": self.dispatch_authorized,
            "decision_reason": self.decision_reason,
            "selected_kb_count": self.selected_kb_count,
            "requirements": [item.to_dict() for item in self.requirements],
            "clarification": self.clarification.to_dict(),
        }

    def safe_summary(self) -> dict[str, Any]:
        """Return an observability summary without user semantic prose."""

        return safe_rag_task_contract_summary(self)


@dataclass(frozen=True)
class _ExecutionPlan:
    response_mode: ResponseMode
    retrieval_policy: RetrievalPolicy
    decision_reason: str


def _validate_compile_inputs(
    route: RagRouteDecision,
    category: RouteCategoryPolicy,
    config: RouteCompilerConfig,
    *,
    question: str,
    selected_kb_count: int,
    available_turn_keys: Iterable[str],
    source: str,
) -> tuple[str, ...]:
    if not isinstance(route, RagRouteDecision):
        raise TaskContractCompilationError("route 必须是已验证的 RagRouteDecision")
    if route.schema_version != ROUTE_DECISION_SCHEMA_VERSION:
        raise TaskContractCompilationError("route schema_version 不受支持")
    if not isinstance(category, RouteCategoryPolicy):
        raise TaskContractCompilationError("category 必须是 RouteCategoryPolicy")
    if not category.enabled:
        raise TaskContractCompilationError("不能编译已禁用的意图分类")
    if category.code != route.intent_code:
        raise TaskContractCompilationError("route intent_code 与 category.code 不一致")
    if category.action not in VALID_CATEGORY_ACTIONS:
        raise TaskContractCompilationError(f"无效 category.action: {category.action}")
    if not isinstance(category.name, str) or not category.name.strip():
        raise TaskContractCompilationError("category.name 不能为空")

    threshold = config.confidence_threshold
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TaskContractCompilationError("confidence_threshold 必须是数字")
    if not math.isfinite(float(threshold)) or not 0 <= float(threshold) <= 1:
        raise TaskContractCompilationError("confidence_threshold 必须位于 0~1")
    if not isinstance(config.allow_general_chat, bool):
        raise TaskContractCompilationError("allow_general_chat 必须是布尔值")
    if not isinstance(question, str) or not question.strip():
        raise TaskContractCompilationError("question 不能为空")
    if isinstance(selected_kb_count, bool) or not isinstance(selected_kb_count, int):
        raise TaskContractCompilationError("selected_kb_count 必须是整数")
    if selected_kb_count < 0:
        raise TaskContractCompilationError("selected_kb_count 不能为负数")
    if not isinstance(source, str) or not _SOURCE_RE.fullmatch(source):
        raise TaskContractCompilationError("source 必须是稳定的小写标识符")

    try:
        normalized_turn_keys = normalize_turn_candidate_keys(available_turn_keys)
    except RouteDecisionValidationError as exc:
        raise TaskContractCompilationError(str(exc)) from exc
    unknown = [
        key
        for key in route.query_resolution.context_turn_keys
        if key not in set(normalized_turn_keys)
    ]
    if unknown:
        raise TaskContractCompilationError(
            f"route 引用了不可用的历史 turn candidate: {unknown}"
        )
    return normalized_turn_keys


def _compile_base_plan(
    route: RagRouteDecision,
    category: RouteCategoryPolicy,
    config: RouteCompilerConfig,
    *,
    explicit_greeting: bool,
    explicit_platform_help: bool,
    inline_writing: bool,
    requires_knowledge: bool,
    knowledge_writing: bool,
) -> _ExecutionPlan:
    # This global policy remains the first guard: administrators can require
    # every otherwise direct task to be grounded in selected enterprise data.
    if not config.allow_general_chat:
        return _ExecutionPlan(
            response_mode="grounded_qa",
            retrieval_policy="required",
            decision_reason="general_chat_disabled",
        )

    # These flags must come from high-certainty local checks.  They may correct
    # a model/category mistake but never carry any resource identity.
    if explicit_greeting:
        return _ExecutionPlan(
            response_mode="general_chat",
            retrieval_policy="skip",
            decision_reason="exact_greeting",
        )
    if explicit_platform_help:
        return _ExecutionPlan(
            response_mode="platform_help",
            retrieval_policy="skip",
            decision_reason="explicit_platform_help",
        )
    if knowledge_writing:
        return _ExecutionPlan(
            response_mode="writing",
            retrieval_policy="required",
            decision_reason="knowledge_dependent_writing",
        )
    if inline_writing:
        return _ExecutionPlan(
            response_mode="writing",
            retrieval_policy="skip",
            decision_reason="inline_writing_content",
        )

    enterprise_scope = route.evidence_scope in {"enterprise_kb", "mixed"}
    if category.action == "retrieve":
        return _ExecutionPlan(
            response_mode="grounded_qa",
            retrieval_policy="required",
            decision_reason="classified_retrieval",
        )
    if category.action == "system_help":
        # Merely claiming platform_self is not enough to bypass retrieval.  The
        # caller must also supply the positive local guard handled above.
        return _ExecutionPlan(
            response_mode="grounded_qa",
            retrieval_policy="required",
            decision_reason="platform_help_scope_guard",
        )
    if category.action == "writing":
        if enterprise_scope or requires_knowledge or knowledge_writing:
            return _ExecutionPlan(
                response_mode="writing",
                retrieval_policy="required",
                decision_reason="knowledge_dependent_writing",
            )
        return _ExecutionPlan(
            response_mode="writing",
            retrieval_policy="skip",
            decision_reason="classified_writing",
        )
    if category.action == "chat":
        if enterprise_scope or requires_knowledge:
            return _ExecutionPlan(
                response_mode="grounded_qa",
                retrieval_policy="required",
                decision_reason="knowledge_scope_guard",
            )
        return _ExecutionPlan(
            response_mode="general_chat",
            retrieval_policy="skip",
            decision_reason="classified_general_chat",
        )
    # The validation layer prevents this branch.  Keep an explicit fail-closed
    # return to make future action extensions safe by default.
    return _ExecutionPlan(
        response_mode="grounded_qa",
        retrieval_policy="required",
        decision_reason="invalid_action_fallback",
    )


def _compile_requirements(
    route: RagRouteDecision,
    *,
    question: str,
) -> tuple[CompiledAnswerRequirement, ...]:
    compiled: list[CompiledAnswerRequirement] = []
    for index, item in enumerate(route.requirements, start=1):
        is_explicit_answer = item.role == "answer" and item.origin == "user_text"
        compiled.append(
            CompiledAnswerRequirement(
                id=f"r{index}",
                role=item.role,
                origin=item.origin,
                description=item.description,
                importance="required" if is_explicit_answer else "helpful",
                source="explicit" if item.origin == "user_text" else "inferred",
            )
        )
    # The route model is semantic assistance, not the sole safety boundary.
    # If it times out or compresses an implicit mapping question into one
    # answer target, derive a domain-neutral bridge from the user's wording.
    # No concrete classification value is guessed here; evidence must still
    # prove that intermediate relationship later in the V2 pipeline.
    if (
        route.readiness == "ready"
        and route.evidence_scope in {"enterprise_kb", "mixed"}
        and not any(item.role == "bridge" for item in compiled)
        and len(compiled) < 8
    ):
        # Prefer the original user question.  A route model is allowed to
        # summarize an answer requirement (for example, reduce
        # ``普通员工的出差标准`` to ``查询出差标准``); that summary must not erase
        # the qualifier that makes a bridge necessary.  Requirement text is
        # retained as a secondary source for callers that compile an already
        # contextualized/decomposed route.
        bridge_inputs = [
            question,
            *(
                item.description
                for item in route.requirements
                if item.role == "answer"
            ),
        ]
        inferred_bridge = next(
            (
                inferred
                for value in bridge_inputs
                for inferred in (infer_implicit_bridge(value),)
                if inferred is not None
            ),
            None,
        )
        if inferred_bridge is not None:
            compiled.append(CompiledAnswerRequirement(
                id=f"r{len(compiled) + 1}",
                role="bridge",
                origin="semantically_entailed",
                description=inferred_bridge.description,
                importance="helpful",
                source="inferred",
            ))
    return tuple(compiled)


def _knowledge_base_clarification() -> RouteClarification:
    return RouteClarification(
        question="该问题需要查询知识库，请至少选择一个知识库。",
        unresolved=(
            RouteUnresolvedSlot(
                role="knowledge_base",
                reason="missing",
                candidate_keys=(),
            ),
        ),
    )


def compile_rag_task_contract(
    route: RagRouteDecision,
    category: RouteCategoryPolicy,
    config: RouteCompilerConfig,
    *,
    question: str,
    selected_kb_count: int,
    available_turn_keys: Iterable[str] = (),
    source: str,
    explicit_greeting: bool = False,
    explicit_platform_help: bool = False,
    inline_writing: bool = False,
    requires_knowledge: bool = False,
    knowledge_writing: bool = False,
) -> RagTaskContract:
    """Compile a validated semantic route into one deterministic task contract."""

    normalized_turn_keys = _validate_compile_inputs(
        route,
        category,
        config,
        question=question,
        selected_kb_count=selected_kb_count,
        available_turn_keys=available_turn_keys,
        source=source,
    )
    for field_name, value in {
        "explicit_greeting": explicit_greeting,
        "explicit_platform_help": explicit_platform_help,
        "inline_writing": inline_writing,
        "requires_knowledge": requires_knowledge,
        "knowledge_writing": knowledge_writing,
    }.items():
        if not isinstance(value, bool):
            raise TaskContractCompilationError(f"{field_name} 必须是布尔值")

    plan = _compile_base_plan(
        route,
        category,
        config,
        explicit_greeting=explicit_greeting,
        explicit_platform_help=explicit_platform_help,
        inline_writing=inline_writing,
        requires_knowledge=requires_knowledge,
        knowledge_writing=knowledge_writing,
    )

    # Confidence never authorizes execution.  It is only used as a one-way
    # safety guard: a low-confidence decision may be upgraded to retrieval but
    # can never downgrade a required retrieval plan to direct generation.
    if (
        plan.retrieval_policy == "skip"
        and route.confidence < float(config.confidence_threshold)
        and plan.decision_reason not in _LOCAL_DIRECT_DECISION_REASONS
    ):
        plan = _ExecutionPlan(
            response_mode=(
                "writing" if plan.response_mode == "writing" else "grounded_qa"
            ),
            retrieval_policy="required",
            decision_reason="low_confidence_safe_retrieval",
        )

    need_retrieval = plan.retrieval_policy == "required"
    readiness = route.readiness
    clarification = route.clarification
    decision_reason = (
        "semantic_clarification"
        if readiness == "needs_clarification"
        else plan.decision_reason
    )
    if readiness == "ready" and need_retrieval and selected_kb_count == 0:
        readiness = "needs_clarification"
        clarification = _knowledge_base_clarification()
        decision_reason = "knowledge_base_required"

    dispatch_authorized = readiness == "ready"
    contract = RagTaskContract(
        schema_version=TASK_CONTRACT_SCHEMA_VERSION,
        route_schema_version=ROUTE_DECISION_SCHEMA_VERSION,
        readiness=readiness,
        intent_code=route.intent_code,
        intent_name=category.name.strip(),
        action=category.action,
        confidence=route.confidence,
        source=source,
        relation=route.relation,
        evidence_scope=route.evidence_scope,
        query_mode=route.query_resolution.mode,
        context_turn_keys=route.query_resolution.context_turn_keys,
        response_mode=plan.response_mode,
        retrieval_policy=plan.retrieval_policy,
        need_retrieval=need_retrieval,
        dispatch_authorized=dispatch_authorized,
        decision_reason=decision_reason,
        selected_kb_count=selected_kb_count,
        requirements=_compile_requirements(route, question=question),
        clarification=clarification,
    )

    # Compilation must never emit a contract that claims authorization but
    # already fails its own invariant gate.
    if dispatch_authorized:
        reason = rag_task_contract_gate_reason(
            contract,
            selected_kb_count=selected_kb_count,
            available_turn_keys=normalized_turn_keys,
        )
        if reason is not None:
            raise TaskContractCompilationError(
                f"编译结果未通过合同门禁: {reason}"
            )
    return contract


def _requirements_are_valid(
    requirements: tuple[CompiledAnswerRequirement, ...],
) -> bool:
    for index, item in enumerate(requirements, start=1):
        if not isinstance(item, CompiledAnswerRequirement) or item.id != f"r{index}":
            return False
        if item.role not in {"answer", "bridge"}:
            return False
        if item.origin not in {"user_text", "semantically_entailed"}:
            return False
        if not isinstance(item.description, str) or not item.description.strip():
            return False
        expected_importance = (
            "required"
            if item.role == "answer" and item.origin == "user_text"
            else "helpful"
        )
        expected_source = "explicit" if item.origin == "user_text" else "inferred"
        if item.importance != expected_importance or item.source != expected_source:
            return False
    return True


def rag_task_contract_gate_reason(
    contract: RagTaskContract,
    *,
    selected_kb_count: int | None = None,
    available_turn_keys: Iterable[str] | None = None,
) -> str | None:
    """Return ``None`` only when the contract is safe to dispatch."""

    if not isinstance(contract, RagTaskContract):
        return "not_task_contract"
    if contract.schema_version != TASK_CONTRACT_SCHEMA_VERSION:
        return "unsupported_task_schema"
    if contract.route_schema_version != ROUTE_DECISION_SCHEMA_VERSION:
        return "unsupported_route_schema"
    if contract.readiness != "ready":
        return "not_ready"
    if not contract.dispatch_authorized:
        return "dispatch_not_authorized"
    if contract.clarification.question != "" or contract.clarification.unresolved:
        return "ready_contract_has_clarification"
    if contract.action not in VALID_CATEGORY_ACTIONS:
        return "invalid_action"
    if contract.response_mode not in VALID_RESPONSE_MODES:
        return "invalid_response_mode"
    if contract.retrieval_policy not in VALID_RETRIEVAL_POLICIES:
        return "invalid_retrieval_policy"
    if contract.need_retrieval != (contract.retrieval_policy == "required"):
        return "retrieval_fields_inconsistent"
    if contract.response_mode in {"general_chat", "platform_help"} and (
        contract.retrieval_policy != "skip"
    ):
        return "direct_mode_requires_skip"
    if contract.response_mode == "grounded_qa" and (
        contract.retrieval_policy != "required"
    ):
        return "grounded_mode_requires_retrieval"
    if (
        isinstance(contract.selected_kb_count, bool)
        or not isinstance(contract.selected_kb_count, int)
        or contract.selected_kb_count < 0
    ):
        return "invalid_selected_kb_count"
    if selected_kb_count is not None:
        if (
            isinstance(selected_kb_count, bool)
            or not isinstance(selected_kb_count, int)
            or selected_kb_count < 0
        ):
            return "invalid_runtime_kb_count"
        if selected_kb_count != contract.selected_kb_count:
            return "selected_kb_count_changed"
    effective_kb_count = (
        contract.selected_kb_count
        if selected_kb_count is None
        else selected_kb_count
    )
    if contract.need_retrieval and effective_kb_count == 0:
        return "required_retrieval_without_kb"
    if contract.relation == "new" and contract.context_turn_keys:
        return "new_route_has_context_binding"
    if contract.query_mode == "contextualize" and not contract.context_turn_keys:
        return "contextualize_missing_context_binding"
    try:
        normalized_contract_keys = normalize_turn_candidate_keys(
            contract.context_turn_keys
        )
    except RouteDecisionValidationError:
        return "invalid_context_turn_keys"
    if tuple(contract.context_turn_keys) != normalized_contract_keys:
        return "invalid_context_turn_keys"
    if available_turn_keys is not None:
        try:
            available = set(normalize_turn_candidate_keys(available_turn_keys))
        except RouteDecisionValidationError:
            return "invalid_available_turn_keys"
        if any(key not in available for key in contract.context_turn_keys):
            return "context_turn_unavailable"
    if not _requirements_are_valid(contract.requirements):
        return "invalid_requirements"
    if not any(item.role == "answer" for item in contract.requirements):
        return "missing_answer_requirement"
    if (
        contract.need_retrieval
        and not any(item.role == "bridge" for item in contract.requirements)
        and any(
            infer_implicit_bridge(item.description) is not None
            for item in contract.requirements
            if item.role == "answer"
        )
    ):
        # A contract manually reconstructed or mutated after compilation must
        # not downgrade an implicit mapping back to a single fact target.
        return "implicit_mapping_missing_bridge"
    if not isinstance(contract.source, str) or not _SOURCE_RE.fullmatch(contract.source):
        return "invalid_source"
    if not isinstance(contract.decision_reason, str) or not contract.decision_reason:
        return "invalid_decision_reason"
    return None


def is_rag_task_contract_dispatchable(
    contract: RagTaskContract,
    *,
    selected_kb_count: int | None = None,
    available_turn_keys: Iterable[str] | None = None,
) -> bool:
    return (
        rag_task_contract_gate_reason(
            contract,
            selected_kb_count=selected_kb_count,
            available_turn_keys=available_turn_keys,
        )
        is None
    )


def require_rag_task_contract_dispatchable(
    contract: RagTaskContract,
    *,
    selected_kb_count: int | None = None,
    available_turn_keys: Iterable[str] | None = None,
) -> None:
    reason = rag_task_contract_gate_reason(
        contract,
        selected_kb_count=selected_kb_count,
        available_turn_keys=available_turn_keys,
    )
    if reason is not None:
        raise TaskContractDispatchError(f"RAG 任务合同不可执行: {reason}")


def safe_rag_task_contract_summary(contract: RagTaskContract) -> dict[str, Any]:
    """Build a trace-safe summary without questions or requirement prose."""

    unresolved = contract.clarification.unresolved
    return {
        "schema_version": contract.schema_version,
        "route_schema_version": contract.route_schema_version,
        "readiness": contract.readiness,
        "intent_code": contract.intent_code,
        "action": contract.action,
        "confidence": contract.confidence,
        "source": contract.source,
        "relation": contract.relation,
        "evidence_scope": contract.evidence_scope,
        "query_mode": contract.query_mode,
        "context_turn_count": len(contract.context_turn_keys),
        "response_mode": contract.response_mode,
        "retrieval_policy": contract.retrieval_policy,
        "need_retrieval": contract.need_retrieval,
        "dispatch_authorized": contract.dispatch_authorized,
        "decision_reason": contract.decision_reason,
        "selected_kb_count": contract.selected_kb_count,
        "requirement_count": len(contract.requirements),
        "required_requirement_count": sum(
            item.importance == "required" for item in contract.requirements
        ),
        "bridge_requirement_count": sum(
            item.role == "bridge" for item in contract.requirements
        ),
        "clarification_unresolved_count": len(unresolved),
        "clarification_roles": [item.role for item in unresolved],
        "clarification_reasons": [item.reason for item in unresolved],
    }


__all__ = [
    "TASK_CONTRACT_SCHEMA_VERSION",
    "CompiledAnswerRequirement",
    "RagTaskContract",
    "RouteCategoryPolicy",
    "RouteCompilerConfig",
    "TaskContractCompilationError",
    "TaskContractDispatchError",
    "compile_rag_task_contract",
    "is_rag_task_contract_dispatchable",
    "rag_task_contract_gate_reason",
    "require_rag_task_contract_dispatchable",
    "safe_rag_task_contract_summary",
]
