import asyncio
import uuid
import logging
import json
import re
import time
from dataclasses import dataclass, replace
from typing import Any, Literal
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import delete as sa_delete, select
from database import AsyncSessionLocal as TaskReadSessionLocal, get_db
from models.db_models import (
    ChatTurn,
    Conversation,
    Document,
    DocumentChunk,
    IntentRouteLog,
    KnowledgeBase,
    Message,
    User,
    now_utc,
)
from core.chat_turns import (
    MAX_PERSIST_ATTEMPTS,
    RECOVERABLE_TURN_STATUSES,
    TurnRequestConflict,
    assert_turn_request_matches,
    build_turn_request_context,
    commit_with_retry,
    find_turn_for_user,
    message_turn_metadata,
    normalize_request_id,
    normalize_turn_id,
    reclaim_stale_turn,
    renew_turn_lease,
    reserve_turn,
    transition_turn,
    turn_duration_ms,
    turn_lease_expired,
)
from core.active_task_state import (
    build_active_task_state,
    parse_active_task_state,
    resolve_active_task_state,
)
from core.semantic_memory import extract_resolved_entity_memory
from models.schemas import (
    ChatRequest,
    ConversationBatchDeleteRequest,
    ConversationOut,
    ConversationRenameRequest,
    MessageOut,
)
from core.retrieval_first_runner import run_retrieval_first_stream

# ``v2`` remains the durable route/trace label, but its only grounded-QA
# executor is the retrieval-first adapter.  There is no second V2 answer path.
run_rag_v2_stream = run_retrieval_first_stream
from core.knowledge_catalog import run_knowledge_catalog_stream
from core.knowledge_result import (
    AuthorizedKnowledgeResult,
    authorize_knowledge_result,
    run_knowledge_result_stream,
)
from core.query_semantics import (
    ROUTE_CLARIFICATION_CONTINUATION_SCHEMA_VERSION,
    RouteClarificationContinuation,
    KnowledgeRequestSemantics,
    RouteScopePartition,
    content_knowledge_request,
    document_catalog_request_for_question,
)
from core.authorized_scope import resolve_authorized_scope_clarification
from core.clarification import (
    ClarificationContract,
    build_clarification_state,
    contract_from_dict,
    proposed_clarification_event,
    public_clarification_event,
    resolve_clarification_reply,
    validate_clarification_state,
)
from core.clarification_presenter import stream_clarification_text
from core.rag_v2.query_plan import (
    partition_plan_by_applicability_scopes,
    plan_query_locally,
)
from core.rag_v2.task_graph import (
    RagExecutionBundle,
)
from core.query_analysis_execution import (
    ExecutionBaseline,
    QUERY_EXECUTION_TRACE_EVENT,
    QUERY_EXECUTION_UNRESOLVED_ROLE,
    QueryExecutionGate,
    build_execution_baseline,
    build_execution_clarification_baseline,
    evaluate_query_execution_gate,
)
from core.direct_response import run_direct_response_stream
from core.deps import get_accessible_kb_ids, require_permission
from core.permissions import CHAT_USE
from core.intent_router import (
    build_verified_evidence_scope_result,
    classify_intent_result,
)
from core.conversation_context import (
    ConversationContext,
    apply_active_task_context,
    apply_resolved_turn_semantics,
    apply_result_reference_memory_context,
    build_active_task_v2_execution_context,
    build_current_turn_v2_execution_context,
    build_resolved_v2_execution_context,
    build_verified_followup_v2_execution_context,
    has_verified_deterministic_followup_context,
    prepare_conversation_context,
    resolve_routed_conversation_context,
    resolve_result_reference_sources,
    route_context_payloads,
)
from core.result_reference_memory import (
    build_result_reference_memory,
    parse_result_reference_memory,
    resolve_result_reference_memory,
)
from core.query_route_compiler import RagTaskContract
from core.rag_dispatch import select_rag_runner as _select_rag_pipeline_version
from core.query_route_contract import RouteClarification, RouteUnresolvedSlot
from core.query_constraints import (
    ApplicabilityScope,
    ScopeSourceSpan,
    candidate_section_key,
    extract_query_constraints,
)
from core.evidence_ambiguity import query_requests_all_scopes
from core.evidence_status import (
    ANSWER_SOURCE_REQUIRED_EVIDENCE_STATUSES,
    CANONICAL_EVIDENCE_STATUSES,
    NON_ANSWER_EVIDENCE_STATUSES,
    SUCCESSFUL_EVIDENCE_SCOPE_STATUSES,
    canonical_evidence_status,
)
from core.rag_trace import content_fields, log_exception_safely, trace_event
from core.logging_config import stream_in_conversation_log
from core.read_sessions import ReadSessionFactory, isolated_read_session
from config import get_settings

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)
# Status semantics are centralized in ``core.evidence_status``.  Keep local
# aliases only to avoid a large unrelated call-site rewrite; they are all
# canonical values, and inbound legacy rows/streams are normalized before
# these sets are consulted.
_NON_ANSWER_SOURCE_STATUSES = NON_ANSWER_EVIDENCE_STATUSES
_ANSWER_SOURCE_REQUIRED_STATUSES = ANSWER_SOURCE_REQUIRED_EVIDENCE_STATUSES
_SUPPORTED_EVIDENCE_STATUSES = CANONICAL_EVIDENCE_STATUSES
_EVIDENCE_SOURCE_VALIDATION_FAILURE_MESSAGE = (
    "回答依据校验失败，无法可靠生成知识库答案。请稍后重试。"
)
_SUCCESSFUL_EVIDENCE_SCOPE_STATUSES = SUCCESSFUL_EVIDENCE_SCOPE_STATUSES
def _search_process_event(
    execution_path: str,
    steps: tuple[tuple[str, str], ...],
) -> dict:
    """Publish the server-authoritative process the current branch will run."""

    return {
        "type": "search_process",
        "schema_version": "search_process.v1",
        "execution_path": execution_path,
        "steps": [{"key": key, "label": label} for key, label in steps],
    }


_CHOICE_TEXT_FIELDS = (
    "products",
    "canonical_products",
    "versions",
    "projects",
    "filenames",
)
def _route_scope_partitions(
    choices: tuple[dict[str, Any], ...],
) -> tuple[RouteScopePartition, ...]:
    """Project server-owned clarification choices into typed scope values."""

    partitions: list[RouteScopePartition] = []
    for choice in choices:
        dimensions: dict[str, str | None] = {}
        for target, fields in (
            ("product", ("products", "canonical_products")),
            ("version", ("versions",)),
            ("project", ("projects",)),
        ):
            values: list[str] = []
            for field in fields:
                raw_values = choice.get(field, [])
                if not isinstance(raw_values, (list, tuple)):
                    return ()
                for raw in raw_values:
                    value = str(raw or "").strip()
                    if value and value not in values:
                        values.append(value)
                # ``products`` is the display value and ``canonical_products``
                # is an internal identity projection; never treat both as
                # two alternatives of one selected scope.
                if values:
                    break
            if len(values) > 1:
                return ()
            dimensions[target] = values[0] if values else None
        if not any(dimensions.values()):
            return ()
        partition = RouteScopePartition(**dimensions)
        if partition not in partitions:
            partitions.append(partition)
    return tuple(partitions)


def _trusted_applicability_scopes(
    continuation: RouteClarificationContinuation | None,
) -> tuple[ApplicabilityScope, ...]:
    """Convert typed clarification partitions into hard task scopes.

    Values originate from the already-authorized document catalog stored in
    the clarification contract.  Synthetic trusted spans preserve that
    provenance for project constraints; no user reply or rendered label is
    parsed here.
    """

    if continuation is None:
        return ()
    scopes: list[ApplicabilityScope] = []
    for partition in continuation.scope_partitions:
        parts = [
            ("product", partition.product),
            ("version", partition.version),
            ("project", partition.project),
        ]
        rendered = " ".join(value for _dimension, value in parts if value)
        cursor = 0
        sources: dict[str, ScopeSourceSpan] = {}
        for dimension, value in parts:
            if not value:
                continue
            start = rendered.find(value, cursor)
            end = start + len(value)
            sources[dimension] = ScopeSourceSpan(
                dimension=dimension,
                start=start,
                end=end,
                span=value,
                origin="trusted_requirement",
            )
            cursor = end
        scopes.append(ApplicabilityScope(
            product=partition.product,
            version=partition.version,
            project=partition.project,
            explicit_version=bool(partition.version),
            explicit_project=bool(partition.project),
            product_source=sources.get("product"),
            version_source=sources.get("version"),
            project_source=sources.get("project"),
            matched_text=rendered,
            extraction_reason="server_resolved_clarification_choice",
        ))
    return tuple(scopes)


def _scope_partitioned_execution_baseline(
    baseline: ExecutionBaseline,
    continuation: RouteClarificationContinuation | None,
) -> ExecutionBaseline:
    """Bind resolved applicability to an immutable answer plan."""

    scopes = _trusted_applicability_scopes(continuation)
    if not scopes:
        return baseline
    plan = partition_plan_by_applicability_scopes(
        baseline.plan,
        scopes,
        comparison=len(scopes) > 1,
    )
    assert continuation is not None
    return build_execution_baseline(
        plan=plan,
        local_surface_plan=baseline.local_surface_plan,
        contextual_plan=baseline.contextual_plan,
        question=continuation.semantic_query,
        standalone_query=continuation.canonical_retrieval_query,
        route_context=baseline.route_context,
        deterministic_is_followup=baseline.deterministic_is_followup,
    )


def _route_clarification_continuation(
    question: str,
    pending_state: dict | None,
    *,
    command: dict | None = None,
) -> RouteClarificationContinuation | None:
    """Build a source-separated continuation for one pending route task.

    A clarification reply is a slot value, not an independent retrieval
    question.  Complete new questions and explicit cancels do not inherit the
    pending task.  The task root and answers remain separate for semantic
    analysis; only the contract owns their retrieval rendering.
    """

    if not isinstance(pending_state, dict):
        return None
    original_query = str(pending_state.get("original_query") or "").strip()
    reply = str(question or "").strip()
    if (
        not original_query
        or len(original_query) > 12000
        or not reply
        or len(reply) > 1200
    ):
        return None
    contract = contract_from_dict(pending_state.get("contract"))
    if contract is None or contract.adapter != "semantic":
        return None
    resolution = resolve_clarification_reply(reply, pending_state, command=command)
    if resolution.action not in {"single", "all", "refine"}:
        return None
    if resolution.action == "single" and resolution.choices:
        mapped_reply = str(resolution.choices[0].get("label") or reply)
    elif resolution.action == "all" and resolution.choices:
        labels = "；".join(
            str(choice.get("label") or "").strip()
            for choice in resolution.choices
            if str(choice.get("label") or "").strip()
        )
        mapped_reply = f"全部范围：{labels}" if labels else reply
    else:
        mapped_reply = resolution.answer or reply
    stored_answers = pending_state.get("prior_answers", [])
    if not isinstance(stored_answers, list):
        return None
    answers: list[str] = []
    for value in stored_answers:
        answer = str(value or "").strip()
        if not answer or len(answer) > 1200:
            return None
        answers.append(answer)
    return RouteClarificationContinuation(
        schema_version=ROUTE_CLARIFICATION_CONTINUATION_SCHEMA_VERSION,
        original_query=original_query,
        current_answer=mapped_reply,
        prior_answers=tuple(answers[-5:]),
        scope_partitions=_route_scope_partitions(resolution.choices),
    )


def _clarification_answer_supplies_scope(
    continuation: RouteClarificationContinuation | None,
) -> bool:
    """Whether the server contract supplied typed applicability partitions."""

    if continuation is None:
        return False
    return bool(continuation.scope_partitions)


@dataclass(frozen=True)
class _EvidenceScopeReply:
    action: Literal[
        "single",
        "compare_all",
        "refine",
        "repeat",
        "cancel",
        "new_question",
    ]
    choices: tuple[dict, ...] = ()


def _evidence_scope_reply_display_text(
    question: str,
    reply: _EvidenceScopeReply | None,
) -> str:
    """Turn an accepted internal choice token into readable chat history."""

    original = str(question or "").strip()
    if reply is None or reply.action not in {"single", "compare_all"}:
        return original
    labels = [
        str(choice.get("label") or "").strip()
        for choice in reply.choices
        if isinstance(choice, dict) and str(choice.get("label") or "").strip()
    ]
    if not labels:
        return original
    if reply.action == "single":
        return f"选择：{labels[0]}"
    return "选择：对比全部范围"


def _active_pending_route_state(value: object) -> dict | None:
    # There is one persisted state protocol.  Adapter-specific execution data
    # is validated by the adapter after this common boundary; it never changes
    # the user-facing selection semantics or state lifecycle.
    return validate_clarification_state(value)


def _parse_evidence_scope_reply(
    question: str,
    pending_state: dict,
    *,
    command: dict | None = None,
) -> _EvidenceScopeReply:
    """Resolve every clarification through the shared reply resolver."""

    resolution = resolve_clarification_reply(
        question,
        pending_state,
        command=command,
    )
    mapped_action = {
        "single": "single",
        "all": "compare_all",
        "refine": "refine",
        "cancel": "cancel",
        "new_question": "new_question",
        "repeat": "repeat",
    }[resolution.action]
    return _EvidenceScopeReply(mapped_action, resolution.choices)


def _evidence_scope_filter(
    reply: _EvidenceScopeReply,
    *,
    current_kb_ids: list[uuid.UUID],
) -> dict | None:
    """Rebuild a request-local filter and intersect it with current KB scope."""

    if reply.action not in {"single", "compare_all"} or not reply.choices:
        return None
    current = {str(value) for value in current_kb_ids}
    choices: list[dict] = []
    kb_ids: set[str] = set()
    doc_ids: set[str] = set()
    record_ids: set[str] = set()
    for raw_choice in reply.choices:
        choice_kb_ids = [str(value) for value in raw_choice.get("kb_ids", [])]
        # Choices only carry aggregate KB/doc lists, not a doc->KB mapping.  If
        # any original KB is absent from the current request, partial filtering
        # could retain document ids from a revoked KB.  Keep the pending state
        # and ask again instead of silently narrowing an advertised choice.
        if not choice_kb_ids or not set(choice_kb_ids).issubset(current):
            return None
        choice = {
            "key": raw_choice["key"],
            "label": raw_choice["label"],
            **{
                field: list(raw_choice.get(field, []))
                for field in _CHOICE_TEXT_FIELDS
            },
            "kb_ids": choice_kb_ids,
            "doc_ids": list(raw_choice.get("doc_ids", [])),
            "record_ids": list(raw_choice.get("record_ids", [])),
            "anchor_doc_ids": list(raw_choice.get("anchor_doc_ids", [])),
            "companion_doc_ids": list(raw_choice.get("companion_doc_ids", [])),
        }
        if raw_choice.get("scope_slices"):
            choice["scope_slices"] = [
                dict(value)
                for value in raw_choice.get("scope_slices", [])
                if isinstance(value, dict)
            ]
        choices.append(choice)
        kb_ids.update(choice_kb_ids)
        doc_ids.update(choice["doc_ids"])
        record_ids.update(choice["record_ids"])
    if not kb_ids or not doc_ids:
        return None
    return {
        "mode": "compare_all" if reply.action == "compare_all" else "single",
        "kb_ids": sorted(kb_ids),
        "doc_ids": sorted(doc_ids),
        "record_ids": sorted(record_ids),
        "choices": choices,
    }


def _scoped_evidence_query(original_query: str, scope_filter: dict) -> str:
    labels = [
        str(choice.get("label") or "").strip()
        for choice in scope_filter.get("choices", [])
        if str(choice.get("label") or "").strip()
    ]
    if scope_filter.get("mode") == "compare_all":
        instruction = "请分别对比以下适用范围后回答：" + "；".join(labels)
    else:
        instruction = "本次只查询以下适用范围：" + "；".join(labels)
    return f"{original_query.strip()}\n{instruction}".strip()


def _refined_evidence_query(original_query: str, refinement: str) -> str:
    return (
        f"{original_query.strip()}。"
        f"用户补充的适用范围：{str(refinement or '').strip()}"
    ).strip()


def _clarification_event_pending_state(
    payload: object,
    *,
    original_query: str,
    selected_kb_ids: list[uuid.UUID],
    base_user_message_id: uuid.UUID,
    clarification_message_id: uuid.UUID,
) -> dict | None:
    """Persist one server-produced clarification contract as unified state."""

    if not isinstance(payload, dict):
        return None
    contract = contract_from_dict(payload)
    if contract is None:
        return None
    try:
        return build_clarification_state(
            contract=contract,
            original_query=original_query,
            selected_kb_ids=selected_kb_ids,
            base_user_message_id=base_user_message_id,
            clarification_message_id=clarification_message_id,
        )
    except ValueError:
        return None


async def _persist_clarification_presentation(
    *,
    conversation_id: uuid.UUID,
    assistant_message_id: uuid.UUID,
    turn_id: uuid.UUID | None,
    content: str,
    trace_id: str,
) -> None:
    """Persist exactly the text emitted by the answer-model presenter.

    The structured state is committed before streaming so the picker is safe
    to use even when presentation fails.  Text is written afterward from an
    isolated session; no fixed business sentence is synthesized when the
    model emits nothing.
    """

    rendered = str(content or "")
    if not rendered:
        return
    try:
        async with TaskReadSessionLocal() as session:
            message = await session.get(Message, assistant_message_id)
            if (
                not isinstance(message, Message)
                or message.conversation_id != conversation_id
                or message.role != "assistant"
            ):
                raise RuntimeError("clarification assistant message is unavailable")
            message.content = rendered
            if turn_id is not None:
                turn = await session.get(ChatTurn, turn_id)
                if (
                    not isinstance(turn, ChatTurn)
                    or turn.conversation_id != conversation_id
                    or turn.assistant_message_id != assistant_message_id
                ):
                    raise RuntimeError("clarification turn is unavailable")
                turn.answer_content = rendered
            await session.commit()
    except Exception as exc:
        log_exception_safely(
            logger,
            "[clarification/presentation persistence] trace=%s conv=%s",
            trace_id,
            conversation_id,
            exc=exc,
        )
        trace_event(
            "clarification.presentation_persistence_failed",
            trace_id=trace_id,
            conversation_id=conversation_id,
            assistant_message_id=assistant_message_id,
            turn_id=turn_id,
            error=exc,
        )


def _closed_query_execution_task_contract(
    task_contract: RagTaskContract,
    query_execution_gate: QueryExecutionGate,
    *,
    unresolved_role: str = QUERY_EXECUTION_UNRESOLVED_ROLE,
) -> RagTaskContract:
    """Project a closed execution gate into the sole durable route contract.

    The local gate is terminal and always takes the same route-clarification
    path.  Keeping the projection here prevents different timing paths from
    persisting different pending-state schemas.
    """

    if not query_execution_gate.needs_clarification:
        raise ValueError("only a closed query execution gate can block a task contract")
    unresolved_reason = query_execution_gate.unresolved_reason
    if unresolved_reason is None:
        raise ValueError("a closed query execution gate requires an unresolved reason")
    return replace(
        task_contract,
        readiness="needs_clarification",
        dispatch_authorized=False,
        decision_reason=query_execution_gate.decision_reason,
        clarification=RouteClarification(
            question=query_execution_gate.clarification_question,
            unresolved=(
                RouteUnresolvedSlot(
                    role=unresolved_role,
                    reason=unresolved_reason,
                ),
            ),
        ),
    )


async def _route_clarification_response(
    *,
    db: AsyncSession,
    conv: Conversation,
    user: User,
    question: str,
    decision_reason: str,
    trace_id: str,
    selected_kb_ids: list[uuid.UUID],
    task_contract: RagTaskContract | None,
    query_execution_gate: QueryExecutionGate | None = None,
    emit_clarification_event: bool = True,
    turn: ChatTurn | None = None,
    existing_user_message: Message | None = None,
    parent_stream_logging: bool = False,
    original_query: str | None = None,
    prior_clarification_answers: tuple[str, ...] = (),
    clarification_choices: tuple[dict, ...] = (),
    clarification_contract: ClarificationContract | None = None,
) -> StreamingResponse:
    """Persist one structured clarification state and stream its expression.

    ``existing_user_message`` is used by post-first-SSE clarification paths.
    The user turn has already been durably accepted at that point, but the
    terminal response must still use the same pending route state and turn
    lifecycle as a synchronous clarification.
    """

    if (
        query_execution_gate is not None
        and not query_execution_gate.needs_clarification
    ):
        raise ValueError(
            "a route clarification response requires a closed query execution gate"
        )
    query_execution_payload = (
        query_execution_gate.to_dict()
        if query_execution_gate is not None
        else None
    )

    unresolved = (
        task_contract.clarification.unresolved
        if task_contract is not None
        else ()
    )
    if clarification_contract is not None:
        contract = clarification_contract
    elif clarification_choices:
        contract = ClarificationContract(
            adapter="semantic",
            dimension=(
                "product_version"
                if any("version" in str(item.get("label") or "").casefold() for item in clarification_choices)
                else "query"
            ),
            reason_code="authorized_scope_ambiguous",
            selection_mode="choice",
            choices=tuple(clarification_choices),
        )
    else:
        unresolved_role = str(unresolved[0].role if unresolved else "query").strip()
        dimension = re.sub(r"[^a-z0-9_]+", "_", unresolved_role.casefold()).strip("_") or "query"
        reason_code = re.sub(r"[^a-z0-9_]+", "_", str(decision_reason or "clarification").casefold()).strip("_") or "clarification"
        contract = ClarificationContract(
            adapter="semantic",
            dimension=dimension[:64],
            reason_code=reason_code[:64],
            selection_mode="refine",
            choices=(),
        )

    input_route_state_revision = (
        _stored_pending_request_identity(turn)[0]
        if turn is not None
        else int(getattr(conv, "route_state_revision", 0) or 0)
    )
    if turn is not None and turn.status == "accepted":
        transition_turn(turn, "generating", trace_id=trace_id)
    user_metadata = (
        message_turn_metadata(turn, status="generating") if turn is not None else {}
    )
    if existing_user_message is not None:
        if existing_user_message.conversation_id != conv.id:
            raise ValueError("existing clarification user message belongs to another conversation")
        user_msg = existing_user_message
        user_msg.content = question
        for key, value in user_metadata.items():
            setattr(user_msg, key, value)
    else:
        user_msg = Message(
            id=uuid.uuid4(),
            conversation_id=conv.id,
            role="user",
            content=question,
            **user_metadata,
        )
    if turn is not None:
        turn.user_message_id = user_msg.id
        transition_turn(
            turn,
            "generated",
            trace_id=trace_id,
            evidence_status="skipped",
            retrieval_executed=False,
            answer_content="",
            answer_sources=[],
            search_snapshot={
                "schema_version": "rag_search_snapshot.v1",
                "candidates": [],
                "answer_sources": [],
                "counters": {
                    "retrieval_executed": False,
                    "evidence_status": "skipped",
                    "displayed_result_count": 0,
                    "answer_source_count": 0,
                    "hit_count": 0,
                    "trace_id": trace_id,
                },
            },
            tokens=0,
        )
        transition_turn(turn, "completed", assistant_message_id=uuid.uuid4())
        assistant_id = turn.assistant_message_id
        assistant_metadata = message_turn_metadata(turn, status="completed")
    else:
        assistant_id = uuid.uuid4()
        assistant_metadata = {}
    assistant_msg = Message(
        id=assistant_id,
        conversation_id=conv.id,
        role="assistant",
        content="",
        sources=[],
        **assistant_metadata,
    )
    if turn is not None:
        for key, value in assistant_metadata.items():
            setattr(user_msg, key, value)
    pending_original_query = str(original_query or question or "").strip()[:12000]
    pending_clarification_answers = [
        str(value).strip()[:1200]
        for value in prior_clarification_answers[-6:]
        if str(value).strip()
    ]
    conv.pending_route_state = build_clarification_state(
        contract=contract,
        original_query=pending_original_query,
        selected_kb_ids=selected_kb_ids,
        base_user_message_id=user_msg.id,
        clarification_message_id=assistant_msg.id,
        prior_answers=pending_clarification_answers,
    )
    conv.route_state_revision = int(getattr(conv, "route_state_revision", 0) or 0) + 1
    if existing_user_message is None:
        db.add_all([user_msg, assistant_msg])
    else:
        # The user message was committed before the post-stream gate.
        # Only the assistant message is new; re-adding the persistent user
        # instance would make lightweight transaction fakes observe a second
        # synthetic user message and obscures the actual one-turn lifecycle.
        db.add(assistant_msg)
    final_pending_route_state = json.loads(
        json.dumps(conv.pending_route_state, ensure_ascii=False)
    )
    final_route_state_revision = int(conv.route_state_revision)
    if turn is not None:
        turn_values = {
            "status": turn.status,
            "trace_id": turn.trace_id,
            "evidence_status": turn.evidence_status,
            "retrieval_executed": turn.retrieval_executed,
            "error_code": turn.error_code,
            "answer_content": turn.answer_content,
            "answer_sources": list(turn.answer_sources or []),
            "search_snapshot": turn.search_snapshot,
            "tokens": turn.tokens,
            "user_message_id": turn.user_message_id,
            "assistant_message_id": turn.assistant_message_id,
            "generated_at": turn.generated_at,
            "completed_at": turn.completed_at,
            "updated_at": turn.updated_at,
            "lease_owner": None,
            "lease_expires_at": None,
        }
        message_specs = (
            {
                "id": user_msg.id,
                "conversation_id": conv.id,
                "role": "user",
                "content": question,
                "sources": None,
                "tokens": None,
                **assistant_metadata,
            },
            {
                "id": assistant_msg.id,
                "conversation_id": conv.id,
                "role": "assistant",
                "content": "",
                "sources": [],
                "tokens": None,
                **assistant_metadata,
            },
        )
        conversation_id = conv.id
        turn_id = turn.id

        async def reapply(session: AsyncSession):
            nonlocal conv, turn
            persisted_conv, persisted_turn = await _reapply_immediate_response(
                session,
                conversation_id=conversation_id,
                final_pending_route_state=final_pending_route_state,
                final_route_state_revision=final_route_state_revision,
                input_route_state_revision=input_route_state_revision,
                messages=message_specs,
                turn_id=turn_id,
                turn_values=turn_values,
            )
            conv = persisted_conv
            if persisted_turn is not None:
                turn = persisted_turn

        await commit_with_retry(db, reapply=reapply)
    else:
        await commit_with_retry(db)

    if emit_clarification_event:
        trace_event(
            "clarification.created",
            trace_id=trace_id,
            conversation_id=conv.id,
            user_id=user.id,
            decision_reason=decision_reason,
            selected_kb_count=len(selected_kb_ids),
            route_state_revision=conv.route_state_revision,
            unresolved_count=len(unresolved),
            unresolved_roles=[item.role for item in unresolved],
            unresolved_reasons=[item.reason for item in unresolved],
            clarification_contract=contract.to_dict(public=False),
        )
    trace_event(
        "chat.response",
        trace_id=trace_id,
        conversation_id=conv.id,
        user_id=user.id,
        evidence_status="skipped",
        retrieval_executed=False,
        decision_reason=decision_reason,
        displayed_result_count=0,
        answer_source_count=0,
        context_evidence_count=0,
        hit_count=0,
        direct_evidence_count=0,
        related_reference_count=0,
        sources=[],
        clarification_contract=contract.to_dict(public=False),
    )

    async def generate_clarification():
        events: list[dict] = [
            {
                "type": "conversation_started",
                "conversation_id": str(conv.id),
                **(
                    {"turn_id": str(turn.id), "request_id": turn.request_id}
                    if turn is not None
                    else {}
                ),
            },
        ]
        if turn is not None:
            events.append(_turn_state_event(turn))
        events.append(
            public_clarification_event(
                final_pending_route_state,
                route_state_revision=final_route_state_revision,
                conversation_id=conv.id,
                persisted=True,
            )
        )
        events.append(_search_process_event(
            "clarification",
            (("analyze", "问题分析"), ("generate", "生成")),
        ))
        if task_contract is not None:
            # The clarification branch does not enter a response runner, so
            # publish the same contract-authoritative intent state here.  Do
            # not include the model's raw route/rationale; the deterministic
            # contract projection is sufficient for the client to show that
            # clarification is required and dispatch is forbidden.
            intent_decision = {
                "intent_code": task_contract.intent_code,
                "intent_name": task_contract.intent_name,
                "action": task_contract.action,
                "confidence": task_contract.confidence,
                "source": task_contract.source,
                "relation": task_contract.relation,
                "readiness": task_contract.readiness,
                "response_mode": task_contract.response_mode,
                "retrieval_policy": task_contract.retrieval_policy,
                "need_retrieval": task_contract.need_retrieval,
                "dispatch_authorized": task_contract.dispatch_authorized,
                "decision_reason": task_contract.decision_reason,
                "task_contract": task_contract.to_dict(),
            }
            if query_execution_payload is not None:
                # The durable contract remains the authorization source; this
                # versioned projection exposes why V2 stopped before retrieval
                # without misrepresenting it as a planner-level clarification.
                intent_decision["query_execution"] = query_execution_payload
            events.append({"type": "intent", "decision": intent_decision})
        events.extend(
            [
                {"type": "search_step", "step": "analyze", "status": "done"},
            {
                "type": "search_results",
                "results": [],
                "answer_sources": [],
                "total": 0,
                "displayed_result_count": 0,
                "answer_source_count": 0,
                "context_evidence_count": 0,
                "hit_count": 0,
                "retrieval_executed": False,
                "evidence_status": "skipped",
                "decision_reason": decision_reason,
                "direct_evidence_count": 0,
                "related_reference_count": 0,
                "trace_id": trace_id,
                "is_followup": False,
                "carryover_source_count": 0,
            },
            {"type": "search_step", "step": "generate", "status": "active"},
            ]
        )
        for event in events:
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        presentation_parts: list[str] = []
        async for delta in stream_clarification_text(
            contract=contract,
            original_query=pending_original_query,
            trace_id=trace_id,
        ):
            presentation_parts.append(delta)
            yield f"data: {json.dumps({'type': 'text_delta', 'content': delta}, ensure_ascii=False)}\n\n"
        presentation_text = "".join(presentation_parts)
        await _persist_clarification_presentation(
            conversation_id=conv.id,
            assistant_message_id=assistant_msg.id,
            turn_id=turn.id if turn is not None else None,
            content=presentation_text,
            trace_id=trace_id,
        )
        yield f"data: {json.dumps({'type': 'search_step', 'step': 'generate', 'status': 'done'}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'conversation_id': str(conv.id)}, ensure_ascii=False)}\n\n"

    clarification_stream = generate_clarification()
    if not parent_stream_logging:
        clarification_stream = stream_in_conversation_log(
            clarification_stream, conversation_id=conv.id
        )
    return StreamingResponse(
        clarification_stream,
        media_type="text/event-stream",
        headers=_turn_response_headers(
            conversation_id=conv.id, trace_id=trace_id, turn=turn
        ),
    )


async def _clarification_control_response(
    *,
    db: AsyncSession,
    conv: Conversation,
    user: User,
    question: str,
    pending_state: dict,
    trace_id: str,
    action: Literal["repeat", "cancel"],
    repeat_reason: str | None = None,
    turn: ChatTurn | None = None,
) -> StreamingResponse:
    """Resolve a clarification control action through the common presenter."""

    pending_contract = contract_from_dict(pending_state.get("contract"))
    if pending_contract is None:
        raise ValueError("invalid unified clarification state")
    presentation_contract = pending_contract

    input_route_state_revision = (
        _stored_pending_request_identity(turn)[0]
        if turn is not None
        else int(getattr(conv, "route_state_revision", 0) or 0)
    )

    if action == "cancel":
        presentation_contract = ClarificationContract(
            adapter=pending_contract.adapter,
            dimension=pending_contract.dimension,
            reason_code="selection_cancelled",
            selection_mode="refine",
        )
        answer = ""
        evidence_status = "skipped"
        decision_reason = "evidence_scope_selection_cancelled"
    else:
        if repeat_reason == "scope_unavailable":
            presentation_contract = ClarificationContract(
                adapter=pending_contract.adapter,
                dimension=pending_contract.dimension,
                reason_code="scope_changed",
                selection_mode="refine",
            )
        elif repeat_reason == "route_contract_unavailable":
            pass
        answer = ""
        evidence_status = "needs_clarification"
        decision_reason = "evidence_scope_selection_invalid"

    if turn is not None and turn.status == "accepted":
        transition_turn(turn, "generating", trace_id=trace_id)
    user_metadata = (
        message_turn_metadata(turn, status="generating") if turn is not None else {}
    )
    user_msg = Message(
        id=uuid.uuid4(),
        conversation_id=conv.id,
        role="user",
        content=question,
        **user_metadata,
    )
    if turn is not None:
        turn.user_message_id = user_msg.id
        transition_turn(
            turn,
            "generated",
            trace_id=trace_id,
            evidence_status=evidence_status,
            retrieval_executed=False,
            answer_content=answer,
            answer_sources=[],
            search_snapshot={
                "schema_version": "rag_search_snapshot.v1",
                "candidates": [],
                "answer_sources": [],
                "counters": {
                    "retrieval_executed": False,
                    "evidence_status": evidence_status,
                    "displayed_result_count": 0,
                    "answer_source_count": 0,
                    "hit_count": 0,
                    "trace_id": trace_id,
                },
            },
            tokens=0,
        )
        transition_turn(turn, "completed", assistant_message_id=uuid.uuid4())
        assistant_id = turn.assistant_message_id
        assistant_metadata = message_turn_metadata(turn, status="completed")
    else:
        assistant_id = uuid.uuid4()
        assistant_metadata = {}
    assistant_msg = Message(
        id=assistant_id,
        conversation_id=conv.id,
        role="assistant",
        content=answer,
        sources=[],
        **assistant_metadata,
    )
    if turn is not None:
        for key, value in assistant_metadata.items():
            setattr(user_msg, key, value)
    should_clear_state = action == "cancel" or repeat_reason == "scope_unavailable"
    if should_clear_state:
        current = getattr(conv, "pending_route_state", None)
        if isinstance(current, dict) and current.get("state_id") == pending_state.get("state_id"):
            conv.pending_route_state = None
            conv.route_state_revision = int(
                getattr(conv, "route_state_revision", 0) or 0
            ) + 1
    db.add_all([user_msg, assistant_msg])
    final_pending_route_state = (
        json.loads(json.dumps(conv.pending_route_state, ensure_ascii=False))
        if isinstance(getattr(conv, "pending_route_state", None), dict)
        else None
    )
    final_route_state_revision = int(
        getattr(conv, "route_state_revision", 0) or 0
    )
    if turn is not None:
        turn_values = {
            "status": turn.status,
            "trace_id": turn.trace_id,
            "evidence_status": turn.evidence_status,
            "retrieval_executed": turn.retrieval_executed,
            "error_code": turn.error_code,
            "answer_content": turn.answer_content,
            "answer_sources": list(turn.answer_sources or []),
            "search_snapshot": turn.search_snapshot,
            "tokens": turn.tokens,
            "user_message_id": turn.user_message_id,
            "assistant_message_id": turn.assistant_message_id,
            "generated_at": turn.generated_at,
            "completed_at": turn.completed_at,
            "updated_at": turn.updated_at,
            "lease_owner": None,
            "lease_expires_at": None,
        }
        message_specs = (
            {
                "id": user_msg.id,
                "conversation_id": conv.id,
                "role": "user",
                "content": question,
                "sources": None,
                "tokens": None,
                **assistant_metadata,
            },
            {
                "id": assistant_msg.id,
                "conversation_id": conv.id,
                "role": "assistant",
                "content": answer,
                "sources": [],
                "tokens": None,
                **assistant_metadata,
            },
        )
        conversation_id = conv.id
        turn_id = turn.id

        async def reapply(session: AsyncSession):
            nonlocal conv, turn
            persisted_conv, persisted_turn = await _reapply_immediate_response(
                session,
                conversation_id=conversation_id,
                final_pending_route_state=final_pending_route_state,
                final_route_state_revision=final_route_state_revision,
                input_route_state_revision=input_route_state_revision,
                messages=message_specs,
                turn_id=turn_id,
                turn_values=turn_values,
            )
            conv = persisted_conv
            if persisted_turn is not None:
                turn = persisted_turn

        await commit_with_retry(db, reapply=reapply)
    else:
        await commit_with_retry(db)

    if should_clear_state:
        trace_event(
            "clarification.resolved",
            trace_id=trace_id,
            conversation_id=conv.id,
            user_id=user.id,
            pending_state_id=pending_state["state_id"],
            resolution=(
                "cancelled" if action == "cancel" else "scope_invalidated"
            ),
            route_state_revision=conv.route_state_revision,
        )
    else:
        trace_event(
            "clarification.repeated",
            trace_id=trace_id,
            conversation_id=conv.id,
            user_id=user.id,
            pending_state_id=pending_state["state_id"],
            dimension=pending_contract.dimension,
            choice_count=len(pending_contract.choices),
            reason=repeat_reason or "invalid_selection",
            route_state_revision=conv.route_state_revision,
        )
    trace_event(
        "chat.response",
        trace_id=trace_id,
        conversation_id=conv.id,
        user_id=user.id,
        evidence_status=evidence_status,
        retrieval_executed=False,
        decision_reason=decision_reason,
        displayed_result_count=0,
        answer_source_count=0,
        context_evidence_count=0,
        hit_count=0,
        direct_evidence_count=0,
        related_reference_count=0,
        sources=[],
        clarification_contract=presentation_contract.to_dict(public=False),
    )

    async def generate_direct():
        events: list[dict] = [
            {
                "type": "conversation_started",
                "conversation_id": str(conv.id),
                **(
                    {"turn_id": str(turn.id), "request_id": turn.request_id}
                    if turn is not None
                    else {}
                ),
            },
            {"type": "search_step", "step": "analyze", "status": "done"},
            {
                "type": "search_results",
                "results": [],
                "answer_sources": [],
                "total": 0,
                "displayed_result_count": 0,
                "answer_source_count": 0,
                "context_evidence_count": 0,
                "hit_count": 0,
                "retrieval_executed": False,
                "evidence_status": evidence_status,
                "decision_reason": decision_reason,
                "direct_evidence_count": 0,
                "related_reference_count": 0,
                "trace_id": trace_id,
                "is_followup": True,
                "carryover_source_count": 0,
            },
        ]
        if turn is not None:
            events.insert(1, _turn_state_event(turn))
        events.insert(
            2 if turn is not None else 1,
            _search_process_event(
                "clarification",
                (("analyze", "问题分析"), ("generate", "生成")),
            ),
        )
        if action == "repeat" and repeat_reason != "scope_unavailable":
            events.append(
                public_clarification_event(
                    pending_state,
                    route_state_revision=int(
                        getattr(conv, "route_state_revision", 0) or 0
                    ),
                    conversation_id=conv.id,
                    persisted=True,
                )
            )
        events.extend(
            [
                {"type": "search_step", "step": "generate", "status": "active"},
            ]
        )
        for event in events:
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        presentation_parts: list[str] = []
        async for delta in stream_clarification_text(
            contract=presentation_contract,
            original_query=str(pending_state.get("original_query") or question),
            trace_id=trace_id,
        ):
            presentation_parts.append(delta)
            yield f"data: {json.dumps({'type': 'text_delta', 'content': delta}, ensure_ascii=False)}\n\n"
        presentation_text = "".join(presentation_parts)
        await _persist_clarification_presentation(
            conversation_id=conv.id,
            assistant_message_id=assistant_msg.id,
            turn_id=turn.id if turn is not None else None,
            content=presentation_text,
            trace_id=trace_id,
        )
        yield f"data: {json.dumps({'type': 'search_step', 'step': 'generate', 'status': 'done'}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'conversation_id': str(conv.id)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        stream_in_conversation_log(generate_direct(), conversation_id=conv.id),
        media_type="text/event-stream",
        headers=_turn_response_headers(
            conversation_id=conv.id, trace_id=trace_id, turn=turn
        ),
    )


def _parse_sse_payload(chunk: str) -> dict | None:
    if not chunk.startswith("data: "):
        return None
    try:
        payload = json.loads(chunk.removeprefix("data: ").strip())
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _bounded_source_identity_snapshot(value: object) -> dict | None:
    """Keep identity/ranking fields only; never persist producer content."""

    if not isinstance(value, dict):
        return None
    metadata_identity = _metadata_source_snapshot_identity(value)
    chunk_identity = _source_snapshot_identity(value)
    if metadata_identity is not None:
        kb_id, doc_id = metadata_identity
        item: dict[str, object] = {
            "source_kind": "document_metadata",
            "id": str(doc_id),
            "doc_id": str(doc_id),
            "kb_id": str(kb_id),
        }
    elif chunk_identity is not None:
        kb_id, doc_id, chunk_id = chunk_identity
        item = {
            "source_kind": "document_chunk",
            "id": str(chunk_id),
            "chunk_id": str(chunk_id),
            "doc_id": str(doc_id),
            "kb_id": str(kb_id),
        }
    else:
        return None
    for key in (
        "filename",
        "file_type",
        "status",
        "status_label",
        "is_active",
        "knowledge_base_name",
        "chunk_index",
        "score",
        "retrieval_score",
        "vector_score",
        "keyword_score",
        "trigram_score",
        "answer_support",
        "topic_relevance",
        "constraint_status",
        "constraint_reason",
        "evidence_role",
        "source_verification",
        "rerank_status",
        "rerank_reason",
        "active_channels",
    ):
        value_item = value.get(key)
        if value_item is None:
            continue
        if isinstance(value_item, (str, int, float, bool, list, dict)):
            item[key] = value_item
    return item


def _bounded_search_snapshot(payload: object) -> dict:
    """Build a bounded, content-free final search state for history/replay."""

    data = payload if isinstance(payload, dict) else {}
    candidates: list[dict] = []
    raw_results = data.get("results")
    if isinstance(raw_results, list):
        for raw in raw_results:
            item = _bounded_source_identity_snapshot(raw)
            if item is not None:
                candidates.append(item)
            if len(candidates) >= 20:
                break
    answer_sources: list[dict] = []
    raw_answer_sources = data.get("answer_sources")
    if isinstance(raw_answer_sources, list):
        for raw in raw_answer_sources:
            item = _bounded_source_identity_snapshot(raw)
            if item is not None:
                answer_sources.append(item)
            if len(answer_sources) >= 20:
                break
    counters = {}
    for key in (
        "total",
        "displayed_result_count",
        "answer_source_count",
        "context_evidence_count",
        "hit_count",
        "direct_evidence_count",
        "related_reference_count",
        "retrieval_executed",
        "evidence_status",
        "coverage_status",
        "trace_id",
        "answer_provenance",
        "general_fallback_mode",
        "grounding_policy",
        "version_resolution_mode",
        "evidence_execution_strategy",
        "model_adjudication_state",
        "unverified_generation",
        "source_verification",
        "unverified_reference_count",
        "retrieval_status",
        "answerability_status",
        "intent_status",
        "semantic_confidence",
    ):
        if key in data:
            value = data.get(key)
            if key == "answer_provenance":
                value = str(value or "").strip().casefold()
                if value not in {"knowledge_base", "general_model"}:
                    continue
            elif key == "general_fallback_mode":
                value = str(value or "").strip().casefold()
                if value not in {
                    "off",
                    "no_hit",
                    "no_hit_or_insufficient",
                }:
                    continue
            elif key == "grounding_policy":
                value = str(value or "").strip().casefold()
                if value not in {"required", "preferred", "none"}:
                    continue
            elif key == "version_resolution_mode":
                value = str(value or "").strip().casefold()
                if value not in {"exact", "partition", "compare", "all", "unknown"}:
                    continue
            elif key == "evidence_execution_strategy":
                value = str(value or "").strip().casefold()
                if value not in {
                    "deterministic",
                    "bounded_small_document",
                    "joint_adjudication",
                    "no_candidates",
                }:
                    continue
            elif key == "model_adjudication_state":
                value = str(value or "").strip().casefold()
                if value not in {
                    "not_requested",
                    "skipped",
                    "no_candidates",
                    "succeeded",
                    "failed",
                }:
                    continue
            elif key == "unverified_generation":
                if not isinstance(value, bool):
                    continue
            elif key == "source_verification":
                value = str(value or "").strip().casefold()
                if value not in {"verified", "unverified"}:
                    continue
            elif key == "unverified_reference_count":
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    continue
            elif key == "retrieval_status":
                value = str(value or "").strip().casefold()
                if value not in {
                    "no_match",
                    "authorized_candidates_found",
                    "unauthorized_only",
                }:
                    continue
            elif key == "answerability_status":
                value = str(value or "").strip().casefold()
                if value not in {
                    "answerable",
                    "scope_unresolved",
                    "evidence_incomplete",
                    "provider_failed",
                    "refused",
                    "unavailable",
                }:
                    continue
            elif key == "intent_status":
                value = str(value or "").strip().casefold()
                if value not in {
                    "unknown",
                    "lookup",
                    "explain",
                    "compare",
                    "modify_guide",
                    "troubleshoot",
                }:
                    continue
            elif key == "semantic_confidence":
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not 0 <= float(value) <= 1
                ):
                    continue
            counters[key] = value
    evidence_quality = _bounded_evidence_quality(data.get("evidence_quality"))
    if evidence_quality is not None:
        counters["evidence_quality"] = evidence_quality
    return {
        "schema_version": "rag_search_snapshot.v1",
        "candidates": candidates,
        "answer_sources": answer_sources,
        "counters": counters,
    }


def _bounded_evidence_quality(value: object) -> dict[str, object] | None:
    """Validate the small quality vector persisted with a search snapshot."""

    if not isinstance(value, dict):
        return None
    levels = {"high", "medium", "low", "unknown"}
    result: dict[str, object] = {}
    for key in ("coverage", "reliability", "freshness", "consistency"):
        level = str(value.get(key) or "").strip().casefold()
        if level not in levels:
            return None
        result[key] = level
    completeness = str(value.get("completeness") or "").strip().casefold()
    if completeness not in {"complete", "partial", "unknown"}:
        return None
    result["completeness"] = completeness
    ratio = value.get("coverage_ratio")
    if ratio is not None:
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not 0 <= float(ratio) <= 1
        ):
            return None
        ratio = round(float(ratio), 4)
    result["coverage_ratio"] = ratio
    raw_missing = value.get("missing_requirement_ids")
    if not isinstance(raw_missing, list) or len(raw_missing) > 8:
        return None
    missing: list[str] = []
    for raw in raw_missing:
        item = str(raw or "").strip()
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", item):
            return None
        if item not in missing:
            missing.append(item)
    result["missing_requirement_ids"] = missing
    return result


def _clarification_locked_search_results(payload: dict, *, trace_id: str) -> dict:
    """Keep every search event fail-closed after an evidence clarification.

    A custom or rolling producer may accidentally emit another ``search_results``
    after the clarification gate.  The later event may still be useful as a
    related-results panel, but it can no longer restore generation authority,
    answer sources, or direct-hit counters.
    """

    raw_results = payload.get("results")
    results: list[dict] = []
    if isinstance(raw_results, list):
        for raw in raw_results:
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            if item.get("evidence_role") == "direct":
                item["evidence_role"] = "related"
                item["score"] = 0.0
                item["pipeline_override_reason"] = (
                    "证据范围待用户选择，不得作为当前回答依据"
                )
            results.append(item)
    related_count = sum(
        item.get("evidence_role") == "related" for item in results
    )
    displayed_count = payload.get(
        "displayed_result_count",
        payload.get("total", len(results)),
    )
    if (
        isinstance(displayed_count, bool)
        or not isinstance(displayed_count, int)
        or displayed_count < 0
    ):
        displayed_count = len(results)
    return {
        **payload,
        "results": results,
        "answer_sources": [],
        "displayed_result_count": displayed_count,
        "answer_source_count": 0,
        "context_evidence_count": 0,
        "hit_count": 0,
        "direct_evidence_count": 0,
        "related_reference_count": related_count,
        "evidence_scope_anchor_hit": False,
        "evidence_scope_anchor_doc_ids": [],
        "evidence_status": "needs_clarification",
        "decision_reason": "evidence_scope_ambiguous",
        "trace_id": payload.get("trace_id") or trace_id,
    }


def _source_snapshot_identity(
    source: object,
) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID] | None:
    """读取历史来源的知识库/文档/片段标识。

    缺少任一标识时按不可披露处理。只验证 doc_id 会让文档重新分块
    后已删除的旧 content 仍从 Message.sources JSON 中返回。
    """

    if (
        not isinstance(source, dict)
        or str(source.get("source_kind") or "").strip().casefold()
        == "document_metadata"
    ):
        return None
    try:
        kb_id = uuid.UUID(str(source.get("kb_id")))
        doc_id = uuid.UUID(str(source.get("doc_id")))
        chunk_id = uuid.UUID(str(source.get("id") or source.get("chunk_id")))
        # A producer may include both aliases.  They must describe the same
        # chunk; otherwise accepting the first one would make the source
        # identity ambiguous and could pair trusted metadata with another
        # chunk's content.
        raw_id = source.get("id")
        raw_chunk_id = source.get("chunk_id")
        if raw_id is not None and raw_chunk_id is not None:
            if uuid.UUID(str(raw_id)) != uuid.UUID(str(raw_chunk_id)):
                return None
    except (TypeError, ValueError, AttributeError):
        return None
    return kb_id, doc_id, chunk_id


def _metadata_source_snapshot_identity(
    source: object,
) -> tuple[uuid.UUID, uuid.UUID] | None:
    """Read one authoritative document-metadata source identity.

    Metadata answers prove catalog facts from the document row itself, so a
    chunk id would be both artificial and stale after re-chunking.  The source
    kind is mandatory and the producer ``id`` alias, when present, must equal
    ``doc_id`` before the database refresh below can trust it.
    """

    if (
        not isinstance(source, dict)
        or str(source.get("source_kind") or "").strip().casefold()
        != "document_metadata"
    ):
        return None
    try:
        kb_id = uuid.UUID(str(source.get("kb_id")))
        doc_id = uuid.UUID(str(source.get("doc_id")))
        if source.get("id") is not None and uuid.UUID(str(source.get("id"))) != doc_id:
            return None
        if source.get("chunk_id") is not None:
            return None
    except (TypeError, ValueError, AttributeError):
        return None
    return kb_id, doc_id


async def _validate_stream_answer_sources(
    db: AsyncSession,
    *,
    raw_sources: object,
    raw_results: object,
    selected_kb_ids: list[uuid.UUID],
    read_session_factory: ReadSessionFactory | None = None,
    allow_unverified: bool = False,
) -> tuple[list[dict], set[tuple[uuid.UUID, uuid.UUID]], str | None]:
    """Validate and refresh producer-provided answer evidence.

    ``search_results`` is an internal SSE boundary, but it is still possible
    for a rolling/custom producer to emit stale or forged source snapshots.
    Before those snapshots are persisted (or used to resolve a pending scope),
    require membership in the current request's KB set and reload one of two
    trusted identities: ``kb/doc/chunk`` for active/ready content evidence, or
    ``kb/doc`` for catalog metadata.  Metadata deliberately accepts inactive,
    processing and failed documents because those states can themselves be
    the fact being answered.  Every returned field is refreshed from the
    database so stale producer content cannot cross the boundary.

    An empty source list is a valid no-evidence result and returns ``None`` as
    the validation error.  Any non-empty list is fail-closed when the DB
    adapter is unavailable or an identity is missing.
    """

    if not isinstance(raw_sources, list):
        return [], set(), "answer_sources_not_a_list"
    if not isinstance(raw_results, list):
        raw_results = []

    allowed_kb_ids: set[uuid.UUID] = set()
    for raw_kb_id in selected_kb_ids:
        parsed_kb_id = (
            raw_kb_id
            if isinstance(raw_kb_id, uuid.UUID)
            else _parse_uuid(raw_kb_id)
        )
        if parsed_kb_id is not None:
            allowed_kb_ids.add(parsed_kb_id)

    # Empty answer evidence is expected for no-hit/error/clarification states.
    # We still do not need a database round trip in that case; the caller will
    # independently force the scope anchor state to false.
    if not raw_sources:
        return [], set(), None
    if not allowed_kb_ids:
        return [], set(), "selected_kb_ids_empty"

    parsed_sources: list[tuple[dict, tuple[Any, ...]]] = []
    seen_source_identities: set[tuple[Any, ...]] = set()
    for raw_source in raw_sources:
        if not isinstance(raw_source, dict):
            return [], set(), "answer_source_not_an_object"
        metadata_identity = _metadata_source_snapshot_identity(raw_source)
        chunk_identity = _source_snapshot_identity(raw_source)
        if metadata_identity is not None:
            identity: tuple[Any, ...] = ("document_metadata", *metadata_identity)
        elif chunk_identity is not None:
            identity = ("document_chunk", *chunk_identity)
        else:
            return [], set(), "answer_source_identity_invalid"
        kb_id = identity[1]
        if kb_id not in allowed_kb_ids:
            return [], set(), "answer_source_kb_forbidden"
        if identity in seen_source_identities:
            return [], set(), "answer_source_duplicate"
        seen_source_identities.add(identity)
        role = str(raw_source.get("evidence_role") or "").strip().casefold()
        if role and role not in {
            "direct",
            "related",
            "unverified",
        }:
            return [], set(), "answer_source_role_invalid"
        verification = str(
            raw_source.get("source_verification") or "verified"
        ).strip().casefold()
        if verification not in {"verified", "unverified"}:
            return [], set(), "answer_source_verification_invalid"
        verification_basis = str(
            raw_source.get("verification_basis") or ""
        ).strip().casefold()
        deterministic_scope = (
            verification_basis == "deterministic_candidate_scope_confirmed"
        )
        is_unverified = role == "unverified" or verification == "unverified"
        # Unverified generation admits legacy unverified sources and the
        # deterministic dominant-document auto-selection.  The latter keeps an
        # honest ``source_verification=unverified`` (the reranker never
        # confirmed it) while its server-side scope decision is expressed by
        # ``verification_basis``; anything else fails closed.
        unverified_admission = allow_unverified and (
            (
                role == "unverified"
                and verification == "unverified"
            )
            or (
                role == "direct"
                and verification == "unverified"
                and deterministic_scope
            )
        )
        if is_unverified and not unverified_admission:
            return [], set(), "unverified_answer_source_not_allowed"
        if allow_unverified and not is_unverified:
            return [], set(), "verified_source_in_unverified_generation"
        parsed_sources.append((dict(raw_source), identity))

    # A source claimed as generation context must also be present in the
    # producer's displayed result snapshot.  This catches a common rolling
    # upgrade failure where answer_sources is populated from a previous pass.
    result_identities: set[tuple[Any, ...]] = set()
    for raw_result in raw_results:
        if not isinstance(raw_result, dict):
            continue
        metadata_identity = _metadata_source_snapshot_identity(raw_result)
        chunk_identity = _source_snapshot_identity(raw_result)
        if metadata_identity is not None:
            result_identities.add(("document_metadata", *metadata_identity))
        elif chunk_identity is not None:
            result_identities.add(("document_chunk", *chunk_identity))
    if any(identity not in result_identities for _, identity in parsed_sources):
        return [], set(), "answer_source_not_in_results"

    chunk_ids = {
        identity[3]
        for _, identity in parsed_sources
        if identity[0] == "document_chunk"
    }
    metadata_document_ids = {
        identity[2]
        for _, identity in parsed_sources
        if identity[0] == "document_metadata"
    }
    # ``isolated_read_session`` always rolls back its owned read transaction
    # before closing it.  SQLAlchemy consequently expires ORM instances on
    # that boundary, even when the session factory normally uses
    # ``expire_on_commit=False``.  Export every field needed by the answer
    # snapshot *inside* that boundary; retaining ORM rows for projection below
    # would turn an otherwise successful validation into a detached-instance
    # failure after the request has already generated an answer.
    current: dict[tuple[Any, ...], dict[str, object]] = {}
    try:
        # This defensive validation is allowed to fail closed, but it must not
        # inherit an aborted transaction from an unrelated optional read (for
        # example a rolling registry migration).  The request session still
        # owns turn/message persistence and is intentionally never rolled back
        # by validation code.
        async with isolated_read_session(
            request_db=db,
            session_factory=read_session_factory,
        ) as read_db:
            if chunk_ids:
                chunk_statement = (
                    select(DocumentChunk, Document)
                    .join(
                        Document,
                        (Document.id == DocumentChunk.doc_id)
                        & (Document.kb_id == DocumentChunk.kb_id),
                    )
                    .where(
                        DocumentChunk.id.in_(chunk_ids),
                        DocumentChunk.kb_id.in_(allowed_kb_ids),
                        Document.is_active.is_(True),
                        Document.status == "ready",
                    )
                )
                result = await read_db.execute(chunk_statement)
                rows = result.all() if hasattr(result, "all") else []
                for row in rows or ():
                    try:
                        chunk, document = row
                        if (
                            document is None
                            or document.is_active is not True
                            or str(document.status or "").strip().casefold() != "ready"
                            or document.id != chunk.doc_id
                            or document.kb_id != chunk.kb_id
                        ):
                            continue
                        identity = (
                            "document_chunk",
                            chunk.kb_id,
                            chunk.doc_id,
                            chunk.id,
                        )
                        current[identity] = {
                            "source_kind": "document_chunk",
                            "id": str(chunk.id),
                            "chunk_id": str(chunk.id),
                            "doc_id": str(chunk.doc_id),
                            "kb_id": str(chunk.kb_id),
                            "content": chunk.content,
                            "chunk_index": chunk.chunk_index,
                            "metadata": dict(chunk.metadata_ or {}),
                            "filename": document.filename,
                            "file_type": document.file_type,
                            "source_url": document.source_url,
                            "image_url": document.image_url,
                            "doc_tags": list(document.tags or []),
                        }
                    except (TypeError, ValueError, AttributeError):
                        continue
            if metadata_document_ids:
                metadata_statement = (
                    select(Document, KnowledgeBase)
                    .join(KnowledgeBase, KnowledgeBase.id == Document.kb_id)
                    .where(
                        Document.id.in_(metadata_document_ids),
                        Document.kb_id.in_(allowed_kb_ids),
                    )
                )
                result = await read_db.execute(metadata_statement)
                rows = result.all() if hasattr(result, "all") else []
                for row in rows or ():
                    try:
                        document, knowledge_base = row
                        if (
                            document is None
                            or knowledge_base is None
                            or document.kb_id != knowledge_base.id
                        ):
                            continue
                        status = (
                            str(document.status or "").strip().casefold()
                            if document.is_active is True
                            else "inactive"
                        )
                        identity = (
                            "document_metadata",
                            document.kb_id,
                            document.id,
                        )
                        current[identity] = {
                            "source_kind": "document_metadata",
                            "id": str(document.id),
                            "doc_id": str(document.id),
                            "kb_id": str(document.kb_id),
                            "filename": document.filename,
                            "file_type": document.file_type,
                            "status": status,
                            "status_label": {
                                "ready": "已就绪",
                                "processing": "处理中",
                                "failed": "处理失败",
                                "inactive": "已停用",
                            }.get(status, status or "未知"),
                            "is_active": bool(document.is_active),
                            "doc_tags": list(document.tags or []),
                            "knowledge_base_name": knowledge_base.name,
                            "created_at": (
                                document.created_at.isoformat()
                                if document.created_at
                                else None
                            ),
                            "updated_at": (
                                document.updated_at.isoformat()
                                if document.updated_at
                                else None
                            ),
                            "content": (
                                f"文档名称：{document.filename}；"
                                f"知识库：{knowledge_base.name}；"
                                f"状态：{status or '未知'}；"
                                f"文件类型：{document.file_type or '未知'}"
                            ),
                        }
                    except (TypeError, ValueError, AttributeError):
                        continue
    except Exception as exc:
        # Do not expose producer content or clear a pending scope when the
        # authorization refresh itself is unavailable.  The caller records a
        # compact reason and persists an empty source list instead.
        logger.warning(
            "[chat/evidence source validation] refresh failed error=%s",
            type(exc).__name__,
        )
        return [], set(), f"source_refresh_failed:{type(exc).__name__}"

    if any(identity not in current for _, identity in parsed_sources):
        return [], set(), "answer_source_not_current"

    refreshed: list[dict] = []
    answer_pairs: set[tuple[uuid.UUID, uuid.UUID]] = set()
    for source, identity in parsed_sources:
        refreshed_snapshot = current[identity]
        refreshed.append(
            {
                **source,
                **refreshed_snapshot,
            }
        )
        answer_pairs.add((identity[1], identity[2]))
    return refreshed, answer_pairs, None


def _parse_uuid(value: object) -> uuid.UUID | None:
    """Parse one UUID without allowing malformed values into a SQL filter."""

    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError, AttributeError):
        return None


def _scope_anchor_coverage_from_sources(
    scope_filter: dict | None,
    answer_pairs: set[tuple[uuid.UUID, uuid.UUID]],
    answer_sources: object = None,
) -> tuple[bool | None, list[str]]:
    """Recompute selected-scope anchors from refreshed answer sources.

    The boolean and document list emitted by a producer are advisory only.
    Pending state can be resolved only when every server-derived choice has at
    least one anchor document in the evidence set that was actually accepted
    by ``_validate_stream_answer_sources``.
    """

    if not isinstance(scope_filter, dict):
        return None, []
    choices = [
        value
        for value in scope_filter.get("choices", [])
        if isinstance(value, dict)
    ]
    if any(choice.get("scope_slices") for choice in choices):
        sources = [
            value for value in (answer_sources or []) if isinstance(value, dict)
        ]
        covered_doc_ids: set[str] = set()
        for choice in choices:
            anchor_slices = [
                value
                for value in choice.get("scope_slices", [])
                if isinstance(value, dict) and value.get("is_anchor") is True
            ]
            if not anchor_slices:
                return False, []
            matched_source = next(
                (
                    source
                    for source in sources
                    if any(
                        _source_matches_scope_slice(source, scope_slice)
                        for scope_slice in anchor_slices
                    )
                ),
                None,
            )
            if matched_source is None:
                return False, sorted(covered_doc_ids)
            covered_doc_ids.add(str(matched_source.get("doc_id") or ""))
        return True, sorted(value for value in covered_doc_ids if value)

    anchor_pairs: set[tuple[uuid.UUID, uuid.UUID]] = set()
    for raw_choice in scope_filter.get("choices", []):
        if not isinstance(raw_choice, dict):
            continue
        kb_ids = {
            parsed
            for value in raw_choice.get("kb_ids", [])
            if (parsed := _parse_uuid(value)) is not None
        }
        anchor_doc_ids = {
            str(value).strip()
            for value in raw_choice.get("anchor_doc_ids", [])
            if str(value).strip()
        }
        for kb_id in kb_ids:
            for raw_doc_id in anchor_doc_ids:
                doc_id = _parse_uuid(raw_doc_id)
                if doc_id is not None:
                    anchor_pairs.add((kb_id, doc_id))
    if not anchor_pairs:
        return False, []
    covered = anchor_pairs & answer_pairs
    covered_doc_ids = sorted({str(doc_id) for _, doc_id in covered})
    return covered == anchor_pairs, covered_doc_ids


def _source_matches_scope_slice(source: dict, scope_slice: dict) -> bool:
    if (
        str(source.get("kb_id") or "") != str(scope_slice.get("kb_id") or "")
        or str(source.get("doc_id") or "")
        != str(scope_slice.get("doc_id") or "")
    ):
        return False
    source_chunk_id = str(
        source.get("id") or source.get("chunk_id") or ""
    ).strip()
    section_key = candidate_section_key(source)
    allowed_section_key = str(scope_slice.get("section_key") or "").strip()
    allowed_chunk_ids = {
        str(value).strip()
        for value in scope_slice.get("chunk_ids", [])
        if str(value).strip()
    }
    return bool(
        (allowed_section_key and section_key == allowed_section_key)
        or (source_chunk_id and source_chunk_id in allowed_chunk_ids)
    )


def _source_matches_scope_filter(source: dict, scope_filter: dict | None) -> bool:
    if not isinstance(scope_filter, dict):
        return False
    choices = [
        value
        for value in scope_filter.get("choices", [])
        if isinstance(value, dict)
    ]
    scope_slices = [
        scope_slice
        for choice in choices
        for scope_slice in choice.get("scope_slices", [])
        if isinstance(scope_slice, dict)
    ]
    if scope_slices:
        return any(
            _source_matches_scope_slice(source, scope_slice)
            for scope_slice in scope_slices
        )
    allowed_pairs = _scope_document_pairs(scope_filter)
    try:
        pair = (
            uuid.UUID(str(source.get("kb_id") or "")),
            uuid.UUID(str(source.get("doc_id") or "")),
        )
    except (TypeError, ValueError, AttributeError):
        return False
    return pair in allowed_pairs


def _scope_document_pairs(
    scope_filter: dict | None,
) -> set[tuple[uuid.UUID, uuid.UUID]]:
    """Expand a server-derived scope choice into exact KB/document pairs."""

    if not isinstance(scope_filter, dict):
        return set()
    pairs: set[tuple[uuid.UUID, uuid.UUID]] = set()
    for raw_choice in scope_filter.get("choices", []):
        if not isinstance(raw_choice, dict):
            continue
        kb_ids = {
            parsed
            for value in raw_choice.get("kb_ids", [])
            if (parsed := _parse_uuid(value)) is not None
        }
        doc_ids = {
            parsed
            for value in raw_choice.get("doc_ids", [])
            if (parsed := _parse_uuid(value)) is not None
        }
        pairs.update((kb_id, doc_id) for kb_id in kb_ids for doc_id in doc_ids)
    return pairs


def _source_snapshot_is_answer_evidence(source: object) -> bool:
    """Reject legacy broad-candidate snapshots that were never answer evidence.

    Older releases persisted every displayed retrieval candidate in
    ``Message.sources``.  Their shared ``evidence_status`` marker lets history
    reads fail closed for requests that had no generation context, while
    snapshots created before the marker remain backward compatible.
    """

    if not isinstance(source, dict):
        return False
    raw_status = source.get("evidence_status")
    # Snapshots created before the status marker retain the documented
    # backwards-compatible behaviour.  Once a marker exists, unknown values
    # fail closed rather than becoming citations by virtue of not appearing in
    # a local deny-list.
    if raw_status in (None, ""):
        return True
    status = canonical_evidence_status(raw_status)
    return status in _ANSWER_SOURCE_REQUIRED_STATUSES


def _read_evidence_status(value: object) -> str | None:
    """Project persisted legacy statuses to the canonical public protocol."""

    if value in (None, ""):
        return None
    return canonical_evidence_status(value) or "error"


async def _historical_clarification(
    pending_route_state: object,
    *,
    conversation_id: uuid.UUID | None,
    route_state_revision: object,
    assistant_message_ids: set[str],
    accessible_set: set[uuid.UUID] | None,
    db: AsyncSession,
) -> dict | None:
    """Build a public picker only from the currently valid server state.

    Pending-state UUID allow-lists are routing internals, not client
    authorization.  Historical restoration therefore rechecks the user's
    current KB scope and every referenced document before exposing a reduced
    choice projection.
    """

    state = _active_pending_route_state(pending_route_state)
    if (
        state is None
        or conversation_id is None
        or state.get("clarification_message_id") not in assistant_message_ids
    ):
        return None
    contract = contract_from_dict(state.get("contract"))
    if contract is None:
        return None
    try:
        revision = int(route_state_revision or 0)
        selected_kb_ids = {
            uuid.UUID(value) for value in state["selected_kb_ids_snapshot"]
        }
    except (TypeError, ValueError, AttributeError):
        return None
    if revision < 0:
        return None
    if accessible_set is not None and not selected_kb_ids.issubset(accessible_set):
        return None

    choices = contract.choices
    referenced_doc_ids = {
        uuid.UUID(value)
        for choice in choices
        for value in choice["doc_ids"]
    }
    if referenced_doc_ids:
        statement = select(Document.id, Document.kb_id).where(
            Document.id.in_(referenced_doc_ids),
            Document.kb_id.in_(selected_kb_ids),
            Document.is_active.is_(True),
            Document.status == "ready",
        )
        if accessible_set is not None:
            statement = statement.where(Document.kb_id.in_(accessible_set))
        current_doc_kbs = {
            doc_id: kb_id
            for doc_id, kb_id in (await db.execute(statement)).all()
        }
        if set(current_doc_kbs) != referenced_doc_ids:
            return None
        for choice in choices:
            choice_kb_ids = {uuid.UUID(value) for value in choice["kb_ids"]}
            if any(
                current_doc_kbs[uuid.UUID(value)] not in choice_kb_ids
                for value in choice["doc_ids"]
            ):
                return None

    event = public_clarification_event(
        state,
        route_state_revision=revision,
        conversation_id=conversation_id,
        persisted=True,
    )
    return event


async def _messages_with_current_source_scope(
    rows: list[Message],
    *,
    user: User,
    db: AsyncSession,
    pending_route_state: object = None,
    route_state_revision: object = 0,
) -> list[MessageOut]:
    """按当前角色范围和文档状态过滤历史 ``sources`` 快照。

    assistant 正文属于用户自己的既有会话记录；额外展开的原始检索片段则必须
    每次按当前 RBAC 重新授权，防止角色范围被撤销或文档停用后仍从 JSONB 快照
    读取 ``content`` / ``source_url``。
    """

    accessible = await get_accessible_kb_ids(user, db)
    accessible_set = set(accessible) if accessible is not None else None
    historical_clarification = await _historical_clarification(
        pending_route_state,
        conversation_id=(rows[0].conversation_id if rows else None),
        route_state_revision=route_state_revision,
        assistant_message_ids={
            str(row.id) for row in rows if row.role == "assistant"
        },
        accessible_set=accessible_set,
        db=db,
    )
    referenced_sources: set[tuple[uuid.UUID, uuid.UUID, uuid.UUID]] = set()
    for row in rows:
        for source in (row.sources if isinstance(row.sources, list) else ()):
            if not _source_snapshot_is_answer_evidence(source):
                continue
            identity = _source_snapshot_identity(source)
            if identity is None:
                continue
            kb_id, _, _ = identity
            if accessible_set is not None and kb_id not in accessible_set:
                continue
            referenced_sources.add(identity)
        raw_snapshot = getattr(row, "search_snapshot", None)
        if isinstance(raw_snapshot, dict):
            for collection_key in ("candidates", "answer_sources"):
                collection = raw_snapshot.get(collection_key)
                if not isinstance(collection, list):
                    continue
                for source in collection[:20]:
                    identity = _source_snapshot_identity(source)
                    if identity is None:
                        continue
                    kb_id, _, _ = identity
                    if accessible_set is not None and kb_id not in accessible_set:
                        continue
                    referenced_sources.add(identity)

    current_sources: dict[tuple[uuid.UUID, uuid.UUID, uuid.UUID], dict] = {}
    if referenced_sources:
        chunk_ids = {chunk_id for _, _, chunk_id in referenced_sources}
        statement = (
            select(DocumentChunk, Document)
            .join(Document, Document.id == DocumentChunk.doc_id)
            .where(
                DocumentChunk.id.in_(chunk_ids),
                Document.kb_id == DocumentChunk.kb_id,
                Document.is_active.is_(True),
                Document.status == "ready",
            )
        )
        if accessible_set is not None:
            statement = statement.where(DocumentChunk.kb_id.in_(accessible_set))
        for chunk, document in (await db.execute(statement)).all():
            identity = (chunk.kb_id, chunk.doc_id, chunk.id)
            if identity not in referenced_sources:
                continue
            # 排名、证据角色与分数保留当轮快照；可披露的文档内容和
            # 元数据始终从当前有效 chunk 重载，避免返回已删除或已更新的旧片段。
            current_sources[identity] = {
                "id": str(chunk.id),
                "chunk_id": str(chunk.id),
                "doc_id": str(chunk.doc_id),
                "kb_id": str(chunk.kb_id),
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata_ or {},
                "filename": document.filename,
                "file_type": document.file_type,
                "source_url": document.source_url,
                "image_url": document.image_url,
                "doc_tags": document.tags or [],
            }

    output: list[MessageOut] = []
    for row in rows:
        visible_sources = []
        if isinstance(row.sources, list):
            for source in row.sources:
                if not _source_snapshot_is_answer_evidence(source):
                    continue
                identity = _source_snapshot_identity(source)
                current = current_sources.get(identity) if identity else None
                if current is not None:
                    visible_sources.append({**dict(source), **current})
        visible_search_snapshot = None
        raw_search_snapshot = getattr(row, "search_snapshot", None)
        if isinstance(raw_search_snapshot, dict):
            counters = raw_search_snapshot.get("counters")
            safe_counters = {}
            if isinstance(counters, dict):
                for key in (
                    "total",
                    "displayed_result_count",
                    "answer_source_count",
                    "context_evidence_count",
                    "hit_count",
                    "direct_evidence_count",
                    "related_reference_count",
                    "retrieval_executed",
                    "evidence_status",
                    "coverage_status",
                    "trace_id",
                    "answer_provenance",
                    "general_fallback_mode",
                    "grounding_policy",
                    "version_resolution_mode",
                    "evidence_execution_strategy",
                    "model_adjudication_state",
                    "retrieval_status",
                    "answerability_status",
                    "intent_status",
                    "semantic_confidence",
                ):
                    value = counters.get(key)
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        if key == "evidence_status":
                            value = _read_evidence_status(value)
                        elif key == "answer_provenance":
                            value = str(value or "").strip().casefold()
                            if value not in {"knowledge_base", "general_model"}:
                                continue
                        elif key == "general_fallback_mode":
                            value = str(value or "").strip().casefold()
                            if value not in {
                                "off",
                                "no_hit",
                                "no_hit_or_insufficient",
                            }:
                                continue
                        elif key == "grounding_policy":
                            value = str(value or "").strip().casefold()
                            if value not in {"required", "preferred", "none"}:
                                continue
                        elif key == "version_resolution_mode":
                            value = str(value or "").strip().casefold()
                            if value not in {"exact", "partition", "compare", "all", "unknown"}:
                                continue
                        elif key == "evidence_execution_strategy":
                            value = str(value or "").strip().casefold()
                            if value not in {
                                "deterministic",
                                "bounded_small_document",
                                "joint_adjudication",
                                "no_candidates",
                            }:
                                continue
                        elif key == "model_adjudication_state":
                            value = str(value or "").strip().casefold()
                            if value not in {
                                "not_requested",
                                "skipped",
                                "no_candidates",
                                "succeeded",
                                "inconclusive",
                                "failed",
                            }:
                                continue
                        elif key == "retrieval_status":
                            value = str(value or "").strip().casefold()
                            if value not in {
                                "no_match",
                                "authorized_candidates_found",
                                "unauthorized_only",
                            }:
                                continue
                        elif key == "answerability_status":
                            value = str(value or "").strip().casefold()
                            if value not in {
                                "answerable",
                                "scope_unresolved",
                                "evidence_incomplete",
                                "provider_failed",
                                "refused",
                                "unavailable",
                            }:
                                continue
                        elif key == "intent_status":
                            value = str(value or "").strip().casefold()
                            if value not in {
                                "unknown",
                                "lookup",
                                "explain",
                                "compare",
                                "modify_guide",
                                "troubleshoot",
                            }:
                                continue
                        elif key == "semantic_confidence":
                            if (
                                isinstance(value, bool)
                                or not isinstance(value, (int, float))
                                or not 0 <= float(value) <= 1
                            ):
                                continue
                        safe_counters[key] = value
                evidence_quality = _bounded_evidence_quality(
                    counters.get("evidence_quality")
                )
                if evidence_quality is not None:
                    safe_counters["evidence_quality"] = evidence_quality

            def refreshed_snapshot_items(key: str) -> list[dict]:
                refreshed_items: list[dict] = []
                raw_items = raw_search_snapshot.get(key)
                if not isinstance(raw_items, list):
                    return refreshed_items
                for raw_item in raw_items[:20]:
                    safe_identity = _bounded_source_identity_snapshot(raw_item)
                    identity = _source_snapshot_identity(safe_identity)
                    current = current_sources.get(identity) if identity else None
                    if safe_identity is not None and current is not None:
                        # ``safe_identity`` contains only bounded ranking and
                        # identity fields; every body/URL/name comes from the
                        # current authorized active+ready document row.
                        refreshed_items.append({**safe_identity, **current})
                return refreshed_items

            visible_search_snapshot = {
                "schema_version": "rag_search_snapshot.v1",
                "candidates": refreshed_snapshot_items("candidates"),
                "answer_sources": refreshed_snapshot_items("answer_sources"),
                # Keep outcome/counters even when current authorization removes
                # every candidate so the UI can still distinguish no-hit from
                # a historical hit whose evidence is no longer accessible.
                "counters": safe_counters,
            }
        # 历史脏数据可能把 sources 存成 dict；不先让 Pydantic 验证 ORM
        # 对象，否则一条异常消息会让整个会话返回 500。
        serialized = MessageOut(
            id=row.id,
            conversation_id=row.conversation_id,
            role=row.role,
            content=row.content,
            sources=visible_sources if isinstance(row.sources, list) else None,
            clarification=(
                historical_clarification
                if historical_clarification is not None
                and historical_clarification["clarification_message_id"]
                == str(row.id)
                else None
            ),
            tokens=row.tokens,
            turn_id=getattr(row, "turn_id", None),
            request_id=getattr(row, "request_id", None),
            status=getattr(row, "turn_status", None),
            turn_status=getattr(row, "turn_status", None),
            trace_id=getattr(row, "trace_id", None),
            evidence_status=_read_evidence_status(
                getattr(row, "evidence_status", None)
            ),
            retrieval_executed=getattr(row, "retrieval_executed", None),
            error_code=getattr(row, "error_code", None),
            delivery_status=getattr(row, "delivery_status", None),
            persistence_status=getattr(row, "persistence_status", None),
            duration_ms=getattr(row, "duration_ms", None),
            search_snapshot=visible_search_snapshot,
            created_at=row.created_at,
        )
        output.append(serialized)
    return output


def _public_stream_error_message(exc: BaseException) -> str:
    """返回可安全展示给前端的生成错误。

    详细异常已按 trace_id 写入服务端日志；不把上游 URL、响应体或
    请求信息通过 SSE 直接暴露给终端用户。
    """

    error_name = type(exc).__name__.casefold()
    if "timeout" in error_name or isinstance(exc, TimeoutError):
        return "模型服务响应超时，请稍后重试"
    return "回答生成失败，请稍后重试"


def _durable_turn_supported(session: object) -> bool:
    """Whether the injected session is a real enough session for turn ledger.

    A handful of legacy unit tests use tiny ``add/commit`` doubles that predate
    the turn protocol.  Production ``AsyncSession`` always exposes ``execute``
    and ``flush``; retaining this capability check keeps those tests focused on
    SSE parsing while the real path remains fully durable.
    """

    return callable(getattr(session, "execute", None)) and callable(
        getattr(session, "flush", None)
    )


def _pending_route_identity(conv: Conversation) -> tuple[int, str | None]:
    revision = max(0, int(getattr(conv, "route_state_revision", 0) or 0))
    pending = _active_pending_route_state(
        getattr(conv, "pending_route_state", None)
    )
    return revision, (str(pending["state_id"]) if pending is not None else None)


def _request_context_for_payload(
    payload: ChatRequest,
    *,
    conversation_id: uuid.UUID,
    pending_route_revision: int,
    pending_state_id: str | None,
) -> dict:
    return build_turn_request_context(
        question=payload.question,
        conversation_id=conversation_id,
        knowledge_base_ids=payload.knowledge_base_ids,
        search_config=payload.search_config.model_dump(),
        pending_route_revision=pending_route_revision,
        pending_state_id=pending_state_id,
        clarification_reply=(
            payload.clarification_reply.model_dump()
            if payload.clarification_reply is not None
            else None
        ),
    )


def _stored_pending_request_identity(turn: ChatTurn) -> tuple[int, str | None]:
    context = getattr(turn, "request_context", None)
    pending = context.get("pending_route") if isinstance(context, dict) else None
    if not isinstance(pending, dict):
        return 0, None
    try:
        revision = max(0, int(pending.get("revision") or 0))
    except (TypeError, ValueError):
        revision = 0
    state_id = str(pending.get("state_id") or "").strip() or None
    return revision, state_id


def _stored_resume_identity(turn: ChatTurn) -> tuple[int, str | None] | None:
    context = getattr(turn, "resume_context", None)
    if not isinstance(context, dict):
        return None
    try:
        revision = max(0, int(context.get("revision") or 0))
    except (TypeError, ValueError):
        return None
    state_id = str(context.get("state_id") or "").strip() or None
    return revision, state_id


async def _reapply_immediate_response(
    session: AsyncSession,
    *,
    conversation_id: uuid.UUID,
    final_pending_route_state: dict | None,
    final_route_state_revision: int,
    input_route_state_revision: int,
    messages: tuple[dict, ...],
    turn_id: uuid.UUID | None,
    turn_values: dict | None,
) -> tuple[Conversation, ChatTurn | None]:
    """Reconstruct an immediate clarification/direct-response transaction.

    This callback runs only after a failed commit and rollback.  Every ORM row
    is loaded again; repeated invocation is an idempotent upsert rather than an
    empty commit of detached objects.
    """

    persisted_turn: ChatTurn | None = None
    if turn_id is not None:
        persisted_turn = await session.get(
            ChatTurn,
            turn_id,
            with_for_update=True,
        )
        if not isinstance(persisted_turn, ChatTurn):
            raise RuntimeError("chat turn 不存在，无法重试保存即时回答")
    persisted_conv = await session.get(
        Conversation,
        conversation_id,
        with_for_update=True,
    )
    if not isinstance(persisted_conv, Conversation):
        raise RuntimeError("会话不存在，无法重试保存即时回答")
    if persisted_turn is not None:
        if persisted_turn.status != "completed":
            persisted_revision = int(
                getattr(persisted_conv, "route_state_revision", 0) or 0
            )
            if persisted_revision != input_route_state_revision:
                raise RuntimeError("会话范围状态已变化，拒绝覆盖重试")

    # A commit acknowledgement can fail after PostgreSQL already committed.
    # In that case the completed turn is authoritative and no state is replayed.
    already_completed = bool(
        persisted_turn is not None and persisted_turn.status == "completed"
    )
    if not already_completed:
        persisted_conv.pending_route_state = final_pending_route_state
        persisted_conv.route_state_revision = final_route_state_revision
        if persisted_turn is not None and turn_values is not None:
            for key, value in turn_values.items():
                setattr(persisted_turn, key, value)

    for spec in messages:
        message_id = spec["id"]
        persisted_message = await session.get(Message, message_id)
        if not isinstance(persisted_message, Message):
            persisted_message = Message(**spec)
            session.add(persisted_message)
        elif not already_completed:
            for key, value in spec.items():
                if key != "id":
                    setattr(persisted_message, key, value)
    return persisted_conv, persisted_turn


def _turn_state_event(turn: ChatTurn, *, replayed: bool = False) -> dict:
    persistence_status = (
        "completed"
        if turn.status == "completed"
        else (
            "failed"
            if turn.status in {"persist_failed", "failed"}
            else turn.status
        )
    )
    return {
        "type": "turn_state",
        "turn_id": str(turn.id),
        "request_id": turn.request_id,
        "status": turn.status,
        "trace_id": turn.trace_id,
        "evidence_status": _read_evidence_status(turn.evidence_status),
        "retrieval_executed": turn.retrieval_executed,
        "error_code": turn.error_code,
        "duration_ms": turn_duration_ms(turn),
        "persistence_status": persistence_status,
        "same_request_recoverable": bool(
            turn.status in RECOVERABLE_TURN_STATUSES
            and turn.answer_content is not None
        ),
        "replayed": replayed,
    }


def _turn_response_headers(
    *,
    conversation_id: uuid.UUID,
    trace_id: str | None,
    turn: ChatTurn | None,
) -> dict[str, str]:
    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
        "X-Conversation-ID": str(conversation_id),
    }
    if trace_id:
        headers["X-RAG-Trace-ID"] = str(trace_id)
    if turn is not None:
        headers["X-RAG-Request-ID"] = str(turn.request_id)
        headers["X-RAG-Turn-ID"] = str(turn.id)
    return headers


def _turn_replay_response(
    *,
    conv: Conversation,
    turn: ChatTurn,
    status_code: int = 200,
    replayed: bool = True,
) -> StreamingResponse:
    """Replay a durable result without invoking routing/retrieval/model code."""

    async def stream():
        yield (
            "data: "
            + json.dumps(
                {
                    "type": "conversation_started",
                    "conversation_id": str(conv.id),
                    "turn_id": str(turn.id),
                    "request_id": turn.request_id,
                },
                ensure_ascii=False,
            )
            + "\n\n"
        )
        yield "data: " + json.dumps(
            _turn_state_event(turn, replayed=replayed), ensure_ascii=False
        ) + "\n\n"
        if turn.status in {"failed", "cancelled", "persist_failed"}:
            same_request_recoverable = bool(
                turn.status == "persist_failed"
                and turn.answer_content is not None
            )
            if same_request_recoverable:
                message = (
                    "回答已生成但仍未完成保存，请稍后使用相同 request_id 重试"
                )
            elif turn.status == "cancelled":
                message = "该请求已取消，请重新发送并使用新的 request_id"
            else:
                message = (
                    "该请求未能完成，且没有可恢复的已生成回答。"
                    "请重新发送问题并使用新的 request_id。"
                )
            yield "data: " + json.dumps(
                {
                    "type": "error",
                    "message": message,
                    "turn_id": str(turn.id),
                    "request_id": turn.request_id,
                    "status": turn.status,
                    "error_code": turn.error_code,
                    "persistence_status": (
                        "failed" if turn.status != "cancelled" else "cancelled"
                    ),
                    "same_request_recoverable": same_request_recoverable,
                    "retry_with_new_request_id": not same_request_recoverable,
                    "replayed": replayed,
                },
                ensure_ascii=False,
            ) + "\n\n"
        if turn.status not in {"failed", "cancelled"} and turn.answer_content:
            sources = (
                list(turn.answer_sources or [])
                if isinstance(turn.answer_sources, list)
                else []
            )
            snapshot = (
                turn.search_snapshot
                if isinstance(turn.search_snapshot, dict)
                else {}
            )
            candidates = snapshot.get("candidates")
            if not isinstance(candidates, list):
                candidates = sources
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "search_results",
                        "results": candidates,
                        "answer_sources": sources,
                        "total": len(candidates),
                        "displayed_result_count": len(candidates),
                        "answer_source_count": len(sources),
                        "context_evidence_count": len(sources),
                        "hit_count": sum(
                            str(item.get("evidence_role") or "").casefold() == "direct"
                            for item in sources
                            if isinstance(item, dict)
                        ),
                        "retrieval_executed": turn.retrieval_executed,
                        "evidence_status": turn.evidence_status,
                        "trace_id": turn.trace_id,
                        "replayed": replayed,
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )
            yield (
                "data: "
                + json.dumps(
                    {"type": "text_delta", "content": turn.answer_content},
                    ensure_ascii=False,
                )
                + "\n\n"
            )
        if turn.tokens is not None:
            yield "data: " + json.dumps(
                {"type": "usage", "total_tokens": turn.tokens}, ensure_ascii=False
            ) + "\n\n"
        yield "data: " + json.dumps(
            {
                "type": "done",
                "conversation_id": str(conv.id),
                "turn_id": str(turn.id),
                "request_id": turn.request_id,
                "status": turn.status,
                "replayed": replayed,
            },
            ensure_ascii=False,
        ) + "\n\n"

    return StreamingResponse(
        stream_in_conversation_log(stream(), conversation_id=conv.id),
        status_code=status_code,
        media_type="text/event-stream",
        headers=_turn_response_headers(
            conversation_id=conv.id, trace_id=turn.trace_id, turn=turn
        ),
    )


async def _recover_turn_answer(
    *,
    conv: Conversation,
    turn: ChatTurn,
) -> ChatTurn:
    """Finish a generated/persist-failed turn in a fresh retryable session."""

    from database import AsyncSessionLocal

    last_error: BaseException | None = None
    for attempt in range(MAX_PERSIST_ATTEMPTS):
        try:
            async with AsyncSessionLocal() as save_db:
                persisted = await save_db.get(ChatTurn, turn.id)
                if not isinstance(persisted, ChatTurn):
                    # Lightweight test doubles may not model the new table;
                    # use the already-loaded object and keep the operation
                    # idempotent for them.
                    persisted = turn
                if persisted.status == "completed":
                    return persisted
                if persisted.status not in RECOVERABLE_TURN_STATUSES:
                    return persisted
                if persisted.answer_content is None:
                    raise RuntimeError("turn 缺少可恢复的已生成回答")
                assistant_id = persisted.assistant_message_id or uuid.uuid4()
                persisted.assistant_message_id = assistant_id
                transition_turn(
                    persisted,
                    "completed",
                    assistant_message_id=assistant_id,
                )
                persisted.error_code = None
                existing = await save_db.get(Message, assistant_id)
                metadata = message_turn_metadata(persisted, status="completed")
                if persisted.user_message_id is not None:
                    persisted_user = await save_db.get(
                        Message, persisted.user_message_id
                    )
                    if isinstance(persisted_user, Message):
                        for key, value in metadata.items():
                            setattr(persisted_user, key, value)
                if isinstance(existing, Message):
                    existing.content = persisted.answer_content
                    existing.sources = list(persisted.answer_sources or [])
                    existing.tokens = persisted.tokens
                    for key, value in metadata.items():
                        setattr(existing, key, value)
                else:
                    save_db.add(
                        Message(
                            id=assistant_id,
                            conversation_id=conv.id,
                            role="assistant",
                            content=persisted.answer_content,
                            sources=list(persisted.answer_sources or []),
                            tokens=persisted.tokens,
                            **metadata,
                        )
                    )
                await save_db.commit()
                turn.status = persisted.status
                turn.assistant_message_id = persisted.assistant_message_id
                turn.completed_at = persisted.completed_at
                turn.updated_at = persisted.updated_at
                turn.error_code = persisted.error_code
                return turn
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt + 1 < MAX_PERSIST_ATTEMPTS:
                await asyncio.sleep(0)
    assert last_error is not None
    raise last_error


async def _mark_turn_persist_failed(
    *,
    turn: ChatTurn,
    trace_id: str,
    error_code: str = "assistant_persistence_failed",
) -> bool:
    """Mark a staged payload recoverable and return whether disk proves it.

    An in-memory ``generated`` object is not recovery evidence: the process may
    have failed before that payload reached PostgreSQL.  Retry guidance is
    emitted only when this function commits a row whose answer payload exists.
    """

    from database import AsyncSessionLocal

    try:
        async with AsyncSessionLocal() as save_db:
            persisted = await save_db.get(ChatTurn, turn.id)
            if isinstance(persisted, ChatTurn):
                if persisted.status != "completed":
                    # ``generated`` and ``persist_failed`` are the only states
                    # that can retain an answer for recovery.
                    if (
                        persisted.status == "generated"
                        and persisted.answer_content is not None
                    ):
                        transition_turn(
                            persisted,
                            "persist_failed",
                            trace_id=trace_id,
                            error_code=error_code,
                        )
                    elif (
                        persisted.status == "persist_failed"
                        and persisted.answer_content is not None
                    ):
                        persisted.error_code = error_code
                        persisted.trace_id = trace_id
                    elif persisted.status in {
                        "accepted",
                        "generating",
                        "generated",
                        "persist_failed",
                    }:
                        # The generated payload itself could not be staged, so
                        # there is nothing a same-id retry can safely recover.
                        transition_turn(
                            persisted,
                            "failed",
                            trace_id=trace_id,
                            error_code="generated_payload_not_persisted",
                        )
                await save_db.commit()
                turn.status = persisted.status
                turn.error_code = persisted.error_code
                turn.trace_id = persisted.trace_id
                turn.answer_content = persisted.answer_content
                turn.generated_at = persisted.generated_at
                return bool(
                    persisted.status in RECOVERABLE_TURN_STATUSES
                    and persisted.answer_content is not None
                )
            else:
                # A missing row (including a lightweight test double) cannot
                # prove that the generated payload survived the process.
                turn.status = "failed"
                turn.error_code = "generated_payload_not_persisted"
                turn.trace_id = trace_id
                return False
    except Exception as exc:
        log_exception_safely(
            logger,
            "[chat/turn persist-failed marker] turn=%s trace=%s",
            turn.id,
            trace_id,
            exc=exc,
        )
    return False


async def _mark_turn_terminal(
    *,
    turn: ChatTurn | None,
    status: Literal["failed", "cancelled"],
    trace_id: str,
    error_code: str,
    evidence_status: str | None = None,
    retrieval_executed: bool | None = None,
) -> None:
    if turn is None:
        return
    from database import AsyncSessionLocal

    try:
        async with AsyncSessionLocal() as save_db:
            persisted = await save_db.get(ChatTurn, turn.id)
            if not isinstance(persisted, ChatTurn):
                persisted = turn
            if persisted.status in {"accepted", "generating"}:
                transition_turn(
                    persisted,
                    status,
                    trace_id=trace_id,
                    evidence_status=evidence_status,
                    retrieval_executed=retrieval_executed,
                    error_code=error_code,
                )
                await save_db.commit()
            turn.status = persisted.status
            turn.error_code = persisted.error_code
            turn.evidence_status = persisted.evidence_status
            turn.retrieval_executed = persisted.retrieval_executed
    except Exception as exc:
        log_exception_safely(
            logger,
            "[chat/turn terminal marker] turn=%s trace=%s status=%s",
            turn.id,
            trace_id,
            status,
            exc=exc,
        )


@dataclass
class _ChatRequestLifecycle:
    """Own the terminal state of one accepted request across every stage.

    Individual runners may still record richer stage-specific failures, but
    they are not lifecycle authorities.  This owner starts tracking only after
    a durable turn has been committed or explicitly reclaimed, then covers
    route preparation, response iteration, and the gap before an async
    generator executes its first line.
    """

    trace_id: str
    turn: ChatTurn | None = None
    conversation_id: uuid.UUID | None = None
    user_id: uuid.UUID | None = None

    def bind(
        self,
        *,
        turn: ChatTurn,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        self.turn = turn
        self.conversation_id = conversation_id
        self.user_id = user_id

    def unbind(self) -> None:
        self.turn = None
        self.conversation_id = None
        self.user_id = None

    def _is_active(self) -> bool:
        return self.turn is not None and self.turn.status in {
            "accepted",
            "generating",
        }

    async def finish(
        self,
        *,
        status: Literal["failed", "cancelled"],
        error_code: str,
        stage: str,
        error: BaseException | None = None,
    ) -> None:
        if not self._is_active():
            return
        trace_event(
            "chat.cancelled" if status == "cancelled" else "chat.error",
            trace_id=self.trace_id,
            conversation_id=self.conversation_id,
            user_id=self.user_id,
            stage=stage,
            error=error,
            lifecycle_owner="request",
        )
        await _mark_turn_terminal(
            turn=self.turn,
            status=status,
            trace_id=self.trace_id,
            error_code=error_code,
            evidence_status="error" if status == "failed" else None,
            retrieval_executed=False,
        )

    async def stream(self, body_iterator):
        try:
            async for chunk in body_iterator:
                yield chunk
        except asyncio.CancelledError as exc:
            await asyncio.shield(self.finish(
                status="cancelled",
                error_code="request_stream_cancelled",
                stage="response_stream",
                error=exc,
            ))
            raise
        except Exception as exc:
            await self.finish(
                status="failed",
                error_code="request_stream_failed",
                stage="response_stream",
                error=exc,
            )
            raise
        else:
            # Every successful stream path must complete, fail, or cancel its
            # turn explicitly.  Silent exhaustion is a protocol violation,
            # not a lease-recovery state to leave behind.
            await self.finish(
                status="failed",
                error_code="stream_ended_without_terminal_state",
                stage="response_stream_exhausted",
            )


async def _stage_generated_turn(
    *,
    turn: ChatTurn,
    conv: Conversation,
    answer: str,
    sources: list,
    tokens: int | None,
    evidence_status: str | None,
    retrieval_executed: bool | None,
    trace_id: str,
    search_snapshot: dict | None = None,
) -> ChatTurn:
    """Durably stage the answer before the assistant message transaction."""

    # This path writes a few fields directly for idempotent retry semantics,
    # bypassing ``transition_turn`` after the first state transition.  Apply
    # the same canonical protocol here so a legacy/custom stream cannot leave
    # ``version_mismatch`` (or an unknown status) in durable chat history.
    canonical_status = _read_evidence_status(evidence_status)
    from database import AsyncSessionLocal

    last_error: BaseException | None = None
    for attempt in range(MAX_PERSIST_ATTEMPTS):
        try:
            async with AsyncSessionLocal() as save_db:
                persisted = await save_db.get(ChatTurn, turn.id)
                if not isinstance(persisted, ChatTurn):
                    persisted = turn
                if persisted.status == "completed":
                    return persisted
                if persisted.status not in {"accepted", "generating", "generated", "persist_failed"}:
                    return persisted
                if persisted.status in {"accepted", "generating"}:
                    transition_turn(persisted, "generated")
                elif persisted.status == "persist_failed":
                    transition_turn(persisted, "generated")
                # ``generated`` -> ``generated`` is intentionally avoided by
                # transition_turn; update its payload directly on retries.
                persisted.answer_content = answer
                persisted.answer_sources = list(sources or [])
                persisted.search_snapshot = search_snapshot
                persisted.tokens = tokens
                persisted.evidence_status = canonical_status
                persisted.retrieval_executed = retrieval_executed
                persisted.trace_id = trace_id
                persisted.error_code = None
                persisted.assistant_message_id = (
                    persisted.assistant_message_id or uuid.uuid4()
                )
                persisted.persistence_attempts = int(
                    getattr(persisted, "persistence_attempts", 0) or 0
                ) + 1
                await save_db.commit()
                turn.status = persisted.status
                turn.answer_content = persisted.answer_content
                turn.answer_sources = persisted.answer_sources
                turn.search_snapshot = persisted.search_snapshot
                turn.tokens = persisted.tokens
                turn.evidence_status = persisted.evidence_status
                turn.retrieval_executed = persisted.retrieval_executed
                turn.trace_id = persisted.trace_id
                turn.assistant_message_id = persisted.assistant_message_id
                turn.generated_at = persisted.generated_at
                turn.updated_at = persisted.updated_at
                return turn
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt + 1 < MAX_PERSIST_ATTEMPTS:
                await asyncio.sleep(0)
    assert last_error is not None
    raise last_error


async def _existing_turn_response(
    *,
    conv: Conversation,
    turn: ChatTurn,
    user: User,
    db: AsyncSession,
) -> StreamingResponse:
    """Resolve duplicate request ids without dispatching retrieval again."""

    if turn.status in RECOVERABLE_TURN_STATUSES:
        try:
            turn = await _recover_turn_answer(conv=conv, turn=turn)
        except Exception as exc:
            await _mark_turn_persist_failed(
                turn=turn,
                trace_id=turn.trace_id or uuid.uuid4().hex,
            )
            trace_event(
                "chat.persistence_error",
                trace_id=turn.trace_id,
                conversation_id=conv.id,
                user_id=user.id,
                operation="recover_turn_answer",
                error=exc,
            )
    # A retry may arrive after the user's KB scope changed or a document was
    # disabled/re-chunked.  Reuse the history authorization boundary so the
    # durable identity snapshot is reloaded from current active+ready rows;
    # never replay stale turn JSON directly.
    replay_row = Message(
        id=turn.assistant_message_id or uuid.uuid4(),
        conversation_id=conv.id,
        role="assistant",
        content=turn.answer_content or "",
        sources=list(turn.answer_sources or []),
        tokens=turn.tokens,
        turn_id=turn.id,
        request_id=turn.request_id,
        turn_status=turn.status,
        trace_id=turn.trace_id,
        evidence_status=turn.evidence_status,
        retrieval_executed=turn.retrieval_executed,
        error_code=turn.error_code,
        delivery_status=("delivered" if turn.status == "completed" else "pending"),
        persistence_status=(
            "completed" if turn.status == "completed" else turn.status
        ),
        duration_ms=turn_duration_ms(turn),
        search_snapshot=turn.search_snapshot,
        created_at=turn.completed_at or turn.updated_at or turn.created_at or now_utc(),
    )
    visible = await _messages_with_current_source_scope(
        [replay_row],
        user=user,
        db=db,
    )
    if visible:
        turn.answer_sources = list(visible[0].sources or [])
        turn.search_snapshot = visible[0].search_snapshot
    status_code = 200
    if turn.status in {"accepted", "generating"}:
        status_code = 202
    return _turn_replay_response(
        conv=conv,
        turn=turn,
        status_code=status_code,
        replayed=True,
    )


def _query_analysis_route_context(
    context: ConversationContext,
    *,
    kb_ids: list[uuid.UUID],
) -> tuple[dict[str, Any], ...]:
    """Expose bounded historical candidates to source-anchored analysis.

    The route contract decides intent/category and permission policy, but it
    cannot be a second owner of referential semantics.  A short question such
    as ``餐补呢`` is often locally indistinguishable from a standalone query;
    withholding every candidate before the source-anchored analyzer sees it
    forces the old system to concatenate historical text and parse it again.

    Giving the analyzer candidates does *not* grant it history: the strict
    schema requires each selected ``t*`` to be cited by exact user-text spans,
    and the backend later reloads sources under current RBAC/KB scope.
    """

    return tuple(
        candidate.to_analysis_dict(allowed_kb_ids=kb_ids)
        for candidate in context.route_turn_candidates
    )


async def _prepare_send_message(
    payload: ChatRequest,
    db: AsyncSession,
    user: User,
    *,
    lifecycle: _ChatRequestLifecycle,
):
    trace_id = lifecycle.trace_id
    request_search_config = payload.search_config.model_dump()
    try:
        request_id = normalize_request_id(payload.request_id)
        requested_turn_id = normalize_turn_id(payload.turn_id) if payload.turn_id else None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    # 流式开始前校验请求的知识库都在可访问范围内（accessible 为 None 表示全部）
    accessible = await get_accessible_kb_ids(user, db)
    if accessible is not None and not set(payload.knowledge_base_ids).issubset(set(accessible)):
        raise HTTPException(status_code=403, detail="无权访问部分知识库")

    durable_turn_enabled = _durable_turn_supported(db)
    durable_turn: ChatTurn | None = None
    conv: Conversation | None = None
    if durable_turn_enabled:
        existing_user_turn = await find_turn_for_user(db, user.id, request_id)
        if existing_user_turn is not None:
            if (
                requested_turn_id is not None
                and requested_turn_id != existing_user_turn.id
            ):
                raise HTTPException(
                    status_code=409,
                    detail="同一 request_id 对应的 turn_id 不一致",
                )
            if (
                payload.conversation_id is not None
                and payload.conversation_id != existing_user_turn.conversation_id
            ):
                raise HTTPException(
                    status_code=409,
                    detail="同一 request_id 已属于其他会话",
                )
            existing_conv = await db.get(
                Conversation, existing_user_turn.conversation_id
            )
            if not existing_conv or (
                not user.is_superadmin and existing_conv.user_id != user.id
            ):
                raise HTTPException(status_code=404, detail="会话不存在")
            stored_revision, stored_state_id = (
                _stored_pending_request_identity(existing_user_turn)
            )
            retry_revision = (
                payload.pending_route_revision
                if payload.pending_route_revision is not None
                else stored_revision
            )
            retry_state_id = (
                payload.pending_state_id
                if payload.pending_state_id is not None
                else stored_state_id
            )
            retry_context = _request_context_for_payload(
                payload,
                conversation_id=existing_user_turn.conversation_id,
                pending_route_revision=retry_revision,
                pending_state_id=retry_state_id,
            )
            try:
                assert_turn_request_matches(existing_user_turn, retry_context)
            except TurnRequestConflict as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

            if turn_lease_expired(existing_user_turn):
                # Serialize stale recovery.  Only one retry may replace the
                # expired lease; all others observe its renewed deadline.
                locked_turn = await find_turn_for_user(
                    db,
                    user.id,
                    request_id,
                    for_update=True,
                )
                if not isinstance(locked_turn, ChatTurn):
                    raise HTTPException(status_code=409, detail="请求状态已变化，请重试")
                if locked_turn.status in {"accepted", "generating"}:
                    refresh = getattr(db, "refresh", None)
                    if callable(refresh):
                        await refresh(
                            existing_conv,
                            attribute_names=[
                                "route_state_revision",
                                "pending_route_state",
                            ],
                        )
                    current_revision, current_state_id = _pending_route_identity(
                        existing_conv
                    )
                    stored_revision, stored_state_id = (
                        _stored_pending_request_identity(locked_turn)
                    )
                    allowed_resume_identities = {
                        (stored_revision, stored_state_id)
                    }
                    stored_resume_identity = _stored_resume_identity(locked_turn)
                    if stored_resume_identity is not None:
                        allowed_resume_identities.add(stored_resume_identity)
                    if (
                        current_revision,
                        current_state_id,
                    ) not in allowed_resume_identities:
                        transition_turn(
                            locked_turn,
                            "failed",
                            trace_id=trace_id,
                            error_code="stale_request_context_changed",
                        )
                        await db.commit()
                        raise HTTPException(
                            status_code=409,
                            detail="待处理的澄清上下文已变化，请使用新的 request_id 重新发送",
                        )
                    if reclaim_stale_turn(locked_turn, owner=trace_id):
                        durable_turn = locked_turn
                        conv = existing_conv
                        lifecycle.bind(
                            turn=durable_turn,
                            conversation_id=conv.id,
                            user_id=user.id,
                        )
                        await db.commit()
                        trace_event(
                            "chat.turn_reclaimed",
                            trace_id=trace_id,
                            conversation_id=conv.id,
                            user_id=user.id,
                            turn_id=durable_turn.id,
                            request_id=durable_turn.request_id,
                            execution_attempts=durable_turn.execution_attempts,
                        )
                    else:
                        await db.commit()
                        return await _existing_turn_response(
                            conv=existing_conv,
                            turn=locked_turn,
                            user=user,
                            db=db,
                        )
                else:
                    await db.commit()
                    return await _existing_turn_response(
                        conv=existing_conv,
                        turn=locked_turn,
                        user=user,
                        db=db,
                    )
            else:
                return await _existing_turn_response(
                    conv=existing_conv,
                    turn=existing_user_turn,
                    user=user,
                    db=db,
                )

    # 获取或创建会话。新会话先 flush 取得 id，但不提前提交；若后续路由/校验失败，
    # 请求结束时整个未提交事务会回滚，避免留下空白会话。
    if conv is None and payload.conversation_id:
        conv = await db.get(Conversation, payload.conversation_id)
        # 复用已有会话时校验归属：非超管不可操作他人会话
        if conv and not user.is_superadmin and conv.user_id != user.id:
            raise HTTPException(status_code=404, detail="会话不存在")
    elif conv is None:
        conv = None

    if not conv:
        conv = Conversation(title=payload.question[:50], user_id=user.id)
        db.add(conv)
        await db.flush()

    # Reserve the idempotency ledger before semantic routing.  A duplicate
    # request therefore never re-enters retrieval/model code, including when
    # the first attempt stopped after generation but before assistant-message
    # persistence.  Legacy test doubles are intentionally left on the old path;
    # real AsyncSession instances always support both methods below.
    durable_turn_created = False
    if durable_turn_enabled and durable_turn is None:
        pending_route_revision, pending_state_id = _pending_route_identity(conv)
        requested_pending_revision = (
            payload.pending_route_revision
            if payload.pending_route_revision is not None
            else pending_route_revision
        )
        requested_pending_state_id = (
            payload.pending_state_id
            if payload.pending_state_id is not None
            else pending_state_id
        )
        if (requested_pending_revision, requested_pending_state_id) != (
            pending_route_revision,
            pending_state_id,
        ):
            raise HTTPException(
                status_code=409,
                detail="待处理的澄清上下文已变化，请刷新会话后重试",
            )
        try:
            durable_turn, durable_turn_created = await reserve_turn(
                db,
                conversation_id=conv.id,
                user_id=user.id,
                request_id=request_id,
                turn_id=requested_turn_id,
                question=payload.question,
                trace_id=trace_id,
                knowledge_base_ids=payload.knowledge_base_ids,
                search_config=request_search_config,
                pending_route_revision=pending_route_revision,
                pending_state_id=pending_state_id,
                clarification_reply=(
                    payload.clarification_reply.model_dump()
                    if payload.clarification_reply is not None
                    else None
                ),
            )
        except TurnRequestConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not durable_turn_created:
            # The existing row is authoritative.  Do not mutate its trace or
            # pending state with this duplicate request.
            if durable_turn.conversation_id != conv.id:
                winner_conv = await db.get(
                    Conversation, durable_turn.conversation_id
                )
                if not winner_conv or (
                    not user.is_superadmin and winner_conv.user_id != user.id
                ):
                    raise HTTPException(status_code=404, detail="会话不存在")
                conv = winner_conv
            return await _existing_turn_response(
                conv=conv,
                turn=durable_turn,
                user=user,
                db=db,
            )
        lifecycle.bind(
            turn=durable_turn,
            conversation_id=conv.id,
            user_id=user.id,
        )
        try:
            await db.commit()
        except Exception as exc:
            lifecycle.unbind()
            try:
                await db.rollback()
            except Exception:
                pass
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "请求已接收但暂时无法保存，请使用相同 request_id 重试",
                    "error_code": "turn_reservation_persistence_uncertain",
                    "same_request_recoverable": True,
                    "retry_with_new_request_id": False,
                },
            ) from exc

    stored_pending_state = getattr(conv, "pending_route_state", None)
    pending_route_state = _active_pending_route_state(stored_pending_state)
    expired_pending_state = bool(stored_pending_state and pending_route_state is None)
    if stored_pending_state and pending_route_state is None:
        conv.pending_route_state = None
        conv.route_state_revision = int(getattr(conv, "route_state_revision", 0) or 0) + 1

    pending_contract = (
        contract_from_dict(pending_route_state.get("contract"))
        if isinstance(pending_route_state, dict)
        else None
    )
    evidence_pending_state = (
        pending_route_state
        if pending_contract is not None and pending_contract.adapter == "evidence"
        else None
    )
    semantic_pending_state = (
        pending_route_state
        if pending_contract is not None and pending_contract.adapter == "semantic"
        else None
    )
    if pending_route_state is not None and set(
        pending_route_state.get("selected_kb_ids_snapshot", [])
    ) != {str(value) for value in payload.knowledge_base_ids}:
        return await _clarification_control_response(
            db=db,
            conv=conv,
            user=user,
            question=payload.question,
            pending_state=pending_route_state,
            trace_id=trace_id,
            action="repeat",
            repeat_reason="scope_unavailable",
            turn=durable_turn,
        )
    semantic_reply = (
        resolve_clarification_reply(
            payload.question,
            semantic_pending_state,
            command=(
                payload.clarification_reply.model_dump()
                if payload.clarification_reply is not None
                else None
            ),
        )
        if semantic_pending_state is not None
        else None
    )
    if semantic_reply is not None and semantic_reply.action in {"repeat", "cancel"}:
        return await _clarification_control_response(
            db=db,
            conv=conv,
            user=user,
            question=payload.question,
            pending_state=semantic_pending_state,
            trace_id=trace_id,
            action=semantic_reply.action,
            turn=durable_turn,
        )
    if semantic_reply is not None and semantic_reply.action == "new_question":
        conv.pending_route_state = None
        conv.route_state_revision = int(
            getattr(conv, "route_state_revision", 0) or 0
        ) + 1
        pending_route_state = None
        semantic_pending_state = None
        pending_contract = None
    evidence_reply = (
        _parse_evidence_scope_reply(
            payload.question,
            evidence_pending_state,
            command=(
                payload.clarification_reply.model_dump()
                if payload.clarification_reply is not None
                else None
            ),
        )
        if evidence_pending_state is not None
        else None
    )
    evidence_filter = (
        _evidence_scope_filter(
            evidence_reply,
            current_kb_ids=payload.knowledge_base_ids,
        )
        if evidence_reply is not None
        and evidence_reply.action in {"single", "compare_all"}
        else None
    )
    evidence_repeat_reason = None
    if (
        evidence_reply is not None
        and evidence_reply.action in {"single", "compare_all"}
        and evidence_filter is None
    ):
        evidence_reply = _EvidenceScopeReply("repeat")
        evidence_repeat_reason = "scope_unavailable"

    cleared_evidence_state_id: str | None = None
    if evidence_reply is not None and evidence_reply.action == "new_question":
        cleared_evidence_state_id = str(evidence_pending_state["state_id"])
        conv.pending_route_state = None
        conv.route_state_revision = int(
            getattr(conv, "route_state_revision", 0) or 0
        ) + 1
        pending_route_state = None
        evidence_pending_state = None

    evidence_refinement_active = bool(
        evidence_pending_state is not None
        and evidence_reply is not None
        and evidence_reply.action == "refine"
    )
    refined_evidence_query = (
        _refined_evidence_query(
            str(evidence_pending_state["original_query"]),
            payload.question,
        )
        if evidence_refinement_active and evidence_pending_state is not None
        else None
    )
    evidence_routing_query = (
        _scoped_evidence_query(
            str(evidence_pending_state["original_query"]),
            evidence_filter,
        )
        if evidence_pending_state is not None and evidence_filter is not None
        else refined_evidence_query
    )
    pipeline_base_query = (
        str(evidence_pending_state["original_query"]).strip()
        if evidence_pending_state is not None and evidence_filter is not None
        else refined_evidence_query
    )
    route_continuation = _route_clarification_continuation(
        payload.question,
        pending_route_state,
        command=(
            payload.clarification_reply.model_dump()
            if payload.clarification_reply is not None
            else None
        ),
    )
    route_continuation_query = (
        route_continuation.canonical_retrieval_query
        if route_continuation is not None
        else None
    )
    route_original_query = (
        route_continuation.original_query
        if route_continuation is not None
        else None
    )
    route_clarification_answers = (
        route_continuation.answers if route_continuation is not None else ()
    )
    evidence_pending_execution = bool(
        evidence_pending_state is not None
        and (evidence_filter is not None or evidence_refinement_active)
    )
    user_message_content = _evidence_scope_reply_display_text(
        payload.question,
        evidence_reply if evidence_pending_execution else None,
    )

    # 在保存本轮用户消息之前读取已有对话。带“这些配置/上述内容”等指代的追问
    # 会得到独立检索问题；上一轮来源只作为候选 id，随后按当前知识库范围和文档
    # 状态重新加载，不能直接信任消息 JSON 快照。
    conversation_context = await prepare_conversation_context(
        db,
        conversation_id=conv.id,
        question=payload.question,
        kb_ids=payload.knowledge_base_ids,
        pending_route_state=pending_route_state,
    )
    resolved_active_task = None
    if not evidence_pending_execution and route_continuation_query is None:
        resolved_active_task = await resolve_active_task_state(
            db,
            value=getattr(conv, "active_task_state", None),
            selected_kb_ids=payload.knowledge_base_ids,
            read_session_factory=TaskReadSessionLocal,
        )
        conversation_context = apply_active_task_context(
            context=conversation_context,
            question=payload.question,
            resolved_task=resolved_active_task,
        )
        if conversation_context.active_task is not None:
            trace_event(
                "conversation.active_task_resolved",
                trace_id=trace_id,
                conversation_id=conv.id,
                user_id=user.id,
                resolution="applied",
                task=conversation_context.active_task.safe_summary(),
                **content_fields(
                    "standalone_query",
                    conversation_context.standalone_query,
                ),
            )
        # Result-list memory: ``我想看第四个`` / ``第四个不是《钉钉》吗`` resolve
        # directly against the numbered list the user last saw.  The ordinal is
        # language structure; the document identity is re-authorized below and
        # the execution floor reads it without re-running retrieval.
        resolved_result_reference = await resolve_result_reference_memory(
            db,
            value=getattr(conv, "result_reference_memory", None),
            question=payload.question,
            selected_kb_ids=payload.knowledge_base_ids,
            read_session_factory=TaskReadSessionLocal,
        )
        if resolved_result_reference is not None:
            conversation_context = apply_result_reference_memory_context(
                context=conversation_context,
                question=payload.question,
                resolved_reference=resolved_result_reference,
            )
            trace_event(
                "conversation.result_reference_memory_resolved",
                trace_id=trace_id,
                conversation_id=conv.id,
                user_id=user.id,
                resolution="applied",
                reference=resolved_result_reference.safe_summary(),
            )
    else:
        resolved_result_reference = None
    if pipeline_base_query is not None or route_continuation_query is not None:
        conversation_context = replace(
            conversation_context,
            standalone_query=(pipeline_base_query or route_continuation_query),
        )

    # 在任何路由/检索之前记录请求和多轮上下文，保证调用链的第一阶段始终
    # 是“接收请求”。此前这里等意图模型返回后才写 chat.request，导致模型
    # 事件排在请求之前；路由校验失败时也会留下没有起点的 running 记录。
    settings = get_settings()
    trace_include_content = settings.rag_trace_include_content
    search_config = payload.search_config.model_dump()
    trace_event(
        "chat.request",
        trace_id=trace_id,
        conversation_id=conv.id,
        user_id=user.id,
        selected_kb_ids=payload.knowledge_base_ids,
        search_config=search_config,
        intent=None,
        decision_reason=(
            "pending_evidence_scope_selection"
            if evidence_pending_execution
            else (
                "unresolved_reference"
                if conversation_context.unresolved_reference
                else "pending_intent_routing"
            )
        ),
        is_followup=conversation_context.is_followup,
        followup_reason=conversation_context.followup_reason,
        history_message_count=len(conversation_context.history_messages),
        carryover_source_count=len(conversation_context.carryover_sources),
        **content_fields("question", payload.question),
        **content_fields(
            "standalone_query",
            conversation_context.standalone_query,
        ),
    )
    if expired_pending_state:
        trace_event(
            "clarification.expired",
            trace_id=trace_id,
            conversation_id=conv.id,
            user_id=user.id,
            route_state_revision=conv.route_state_revision,
        )
    trace_event(
        "conversation.context_resolved",
        trace_id=trace_id,
        conversation_id=conv.id,
        user_id=user.id,
        is_followup=conversation_context.is_followup,
        followup_reason=conversation_context.followup_reason,
        unresolved_reference=conversation_context.unresolved_reference,
        history_message_count=len(conversation_context.history_messages),
        carryover_source_count=len(conversation_context.carryover_sources),
        **content_fields(
            "standalone_query",
            conversation_context.standalone_query,
        ),
    )

    if evidence_pending_state is not None and evidence_reply is not None:
        if evidence_reply.action == "cancel":
            return await _clarification_control_response(
                db=db,
                conv=conv,
                user=user,
                question=payload.question,
                pending_state=evidence_pending_state,
                trace_id=trace_id,
                action="cancel",
                turn=durable_turn,
            )
        if evidence_reply.action == "repeat":
            return await _clarification_control_response(
                db=db,
                conv=conv,
                user=user,
                question=payload.question,
                pending_state=evidence_pending_state,
                trace_id=trace_id,
                action="repeat",
                repeat_reason=evidence_repeat_reason,
                turn=durable_turn,
            )

    route_candidates = route_context_payloads(conversation_context)
    trace_event(
        "conversation.context_candidates",
        trace_id=trace_id,
        conversation_id=conv.id,
        user_id=user.id,
        candidate_count=len(route_candidates),
        reusable_source_count=sum(
            max(0, int(item.get("reusable_source_count") or 0))
            for item in route_candidates
            if isinstance(item, dict)
        ),
        has_pending_clarification=bool(pending_route_state),
    )
    if (
        conversation_context.unresolved_reference
        and not route_candidates
        and not evidence_pending_execution
    ):
        trace_event(
            "conversation.reference_unresolved",
            trace_id=trace_id,
            conversation_id=conv.id,
            user_id=user.id,
            reason=conversation_context.followup_reason,
            selected_kb_count=len(payload.knowledge_base_ids),
            **content_fields("question", payload.question),
        )
        response = await _route_clarification_response(
            db=db,
            conv=conv,
            user=user,
            question=payload.question,
            decision_reason="unresolved_reference",
            trace_id=trace_id,
            selected_kb_ids=payload.knowledge_base_ids,
            task_contract=None,
            emit_clarification_event=False,
            turn=durable_turn,
        )
        if cleared_evidence_state_id is not None:
            trace_event(
                "clarification.resolved",
                trace_id=trace_id,
                conversation_id=conv.id,
                user_id=user.id,
                pending_state_id=cleared_evidence_state_id,
                resolution="new_question",
                route_state_revision=conv.route_state_revision,
            )
        return response

    # 普通请求的路由输入就是本轮原文；澄清续接使用合同生成的单任务投影
    # 来判断是否已经 ready。投影将补充条件放进同一任务的限定语中，不会把
    # slot value 渲染为第二个并列问题。
    routing_question = (
        evidence_routing_query
        or (
            route_continuation.canonical_retrieval_query
            if route_continuation is not None
            else payload.question
        )
    )
    if evidence_pending_execution:
        try:
            routing_result = build_verified_evidence_scope_result(
                db,
                pipeline_base_query or routing_question,
                user=user,
                selected_kb_ids=payload.knowledge_base_ids,
                conversation_id=conv.id,
                record_log=True,
                trace_id=trace_id,
                refined=evidence_refinement_active,
            )
            trace_event(
                "evidence.route_contract_built",
                trace_id=trace_id,
                conversation_id=conv.id,
                user_id=user.id,
                pending_state_id=evidence_pending_state["state_id"],
                resolution=(
                    "refined"
                    if evidence_refinement_active
                    else (
                        "compare_all"
                        if evidence_filter.get("mode") == "compare_all"
                        else "selected"
                    )
                ),
                selected_kb_count=len(set(payload.knowledge_base_ids)),
                decision_reason=routing_result.decision.decision_reason,
            )
        except Exception as exc:
            trace_event(
                "evidence.route_contract_failed",
                trace_id=trace_id,
                conversation_id=conv.id,
                user_id=user.id,
                pending_state_id=evidence_pending_state["state_id"],
                resolution=(
                    "refined"
                    if evidence_refinement_active
                    else (
                        "compare_all"
                        if evidence_filter.get("mode") == "compare_all"
                        else "selected"
                    )
                ),
                error=exc,
            )
            log_exception_safely(
                logger,
                "[chat/evidence pending route error] trace=%s conv=%s",
                trace_id,
                conv.id,
                exc=exc,
            )
            return await _clarification_control_response(
                db=db,
                conv=conv,
                user=user,
                question=payload.question,
                pending_state=evidence_pending_state,
                trace_id=trace_id,
                action="repeat",
                repeat_reason="route_contract_unavailable",
                turn=durable_turn,
            )
    else:
        try:
            routing_result = await classify_intent_result(
                db,
                routing_question,
                user=user,
                selected_kb_ids=payload.knowledge_base_ids,
                conversation_id=conv.id,
                record_log=True,
                trace_id=trace_id,
                route_context=route_candidates,
                has_pending_clarification=bool(pending_route_state),
                fallback_relation=conversation_context.relation,
                fallback_query_mode=conversation_context.query_resolution_mode,
                fallback_unresolved=conversation_context.unresolved_reference,
            )
        except Exception as exc:
            # 路由配置/数据库故障不应让调用链停留在 running；接口仍按原语义
            # 抛出异常，但保留安全的阶段错误供后台排查。
            trace_event(
                "chat.error",
                trace_id=trace_id,
                conversation_id=conv.id,
                user_id=user.id,
                stage="intent_routing",
                error=exc,
            )
            log_exception_safely(
                logger,
                "[chat/intent routing error] trace=%s conv=%s",
                trace_id,
                conv.id,
                exc=exc,
            )
            await _mark_turn_terminal(
                turn=durable_turn,
                status="failed",
                trace_id=trace_id,
                error_code="intent_routing_failed",
                evidence_status="error",
                retrieval_executed=False,
            )
            raise
    decision = routing_result.decision
    route_decision = getattr(routing_result, "route_decision", None)
    task_contract = getattr(routing_result, "task_contract", None)
    # Preserve the route contract for audit and clarification state.  The
    # retrieval-first runner consumes only an authorized, dispatchable
    # contract; unresolved route semantics remain an explicit clarification.
    route_task_contract = task_contract
    if route_decision is not None and task_contract is not None:
        conversation_context = await resolve_routed_conversation_context(
            db,
            context=conversation_context,
            question=routing_question,
            kb_ids=payload.knowledge_base_ids,
            route_decision=route_decision,
        )
        if pipeline_base_query is not None:
            conversation_context = replace(
                conversation_context,
                standalone_query=pipeline_base_query,
            )
        trace_event(
            "conversation.context_resolved",
            trace_id=trace_id,
            conversation_id=conv.id,
            user_id=user.id,
            relation=(route_decision.relation if route_decision is not None else "legacy"),
            readiness=route_task_contract.readiness,
            query_mode=task_contract.query_mode,
            context_turn_count=len(route_task_contract.context_turn_keys),
            context_projection="route_projection",
            is_followup=conversation_context.is_followup,
            followup_reason=conversation_context.followup_reason,
            unresolved_reference=conversation_context.unresolved_reference,
            history_message_count=len(conversation_context.history_messages),
            carryover_source_count=len(conversation_context.carryover_sources),
            **content_fields("standalone_query", conversation_context.standalone_query),
        )
        trace_event(
            "intent.routing_decision",
            trace_id=trace_id,
            conversation_id=conv.id,
            user_id=user.id,
            intent=decision.to_dict(),
            route_schema_version=route_decision.schema_version,
            contract_schema_version=route_task_contract.schema_version,
            relation=(route_decision.relation if route_decision is not None else "legacy"),
            readiness=route_task_contract.readiness,
            evidence_scope=route_decision.evidence_scope,
            query_mode=task_contract.query_mode,
            context_turn_count=len(route_task_contract.context_turn_keys),
            requirement_count=len(route_task_contract.requirements),
            dispatch_authorized=route_task_contract.dispatch_authorized,
            execution_dispatch_authorized=task_contract.dispatch_authorized,
            selected_kb_count=len(payload.knowledge_base_ids),
            decision_reason=decision.decision_reason,
        )
        if not task_contract.dispatch_authorized:
            response = await _route_clarification_response(
                db=db,
                conv=conv,
                user=user,
                question=payload.question,
                decision_reason=route_task_contract.decision_reason,
                trace_id=trace_id,
                selected_kb_ids=payload.knowledge_base_ids,
                task_contract=route_task_contract,
                turn=durable_turn,
                original_query=(route_original_query or routing_question),
                prior_clarification_answers=route_clarification_answers,
            )
            if cleared_evidence_state_id is not None:
                trace_event(
                    "clarification.resolved",
                    trace_id=trace_id,
                    conversation_id=conv.id,
                    user_id=user.id,
                    pending_state_id=cleared_evidence_state_id,
                    resolution="new_question",
                    route_state_revision=conv.route_state_revision,
                )
            return response
        intent_payload = {
            **decision.to_dict(),
            "route_decision": route_decision.to_dict(),
            "task_contract": task_contract.to_dict(),
            "diagnostics": getattr(routing_result, "diagnostics", {}),
        }
    else:
        # Rolling-upgrade compatibility for an old classifier implementation.
        # New chat requests in this release always receive the v1 fields.
        trace_event(
            "intent.routing_decision",
            trace_id=trace_id,
            conversation_id=conv.id,
            user_id=user.id,
            intent=decision.to_dict(),
            selected_kb_count=len(payload.knowledge_base_ids),
            decision_reason=decision.decision_reason,
        )
        if decision.need_retrieval and not payload.knowledge_base_ids:
            trace_event(
                "chat.error",
                trace_id=trace_id,
                conversation_id=conv.id,
                user_id=user.id,
                stage="request_validation",
                error=ValueError("该问题需要查询知识库，请至少选择一个知识库"),
                evidence_status="error",
            )
            await _mark_turn_terminal(
                turn=durable_turn,
                status="failed",
                trace_id=trace_id,
                error_code="knowledge_base_required",
                evidence_status="error",
                retrieval_executed=False,
            )
            raise HTTPException(
                status_code=400,
                detail="该问题需要查询知识库，请至少选择一个知识库",
            )
        intent_payload = decision.to_dict()

    pipeline_version, pipeline_reason = _select_rag_pipeline_version(
        task_contract=task_contract,
        evidence_scope_filter=evidence_filter,
        evidence_scope_refinement_active=evidence_refinement_active,
        is_followup=conversation_context.is_followup,
        carryover_sources=conversation_context.carryover_sources,
        selected_kb_count=len(set(payload.knowledge_base_ids)),
    )
    if pipeline_version == "reject":
        trace_event(
            "chat.error",
            trace_id=trace_id,
            conversation_id=conv.id,
            user_id=user.id,
            stage="runner_selection",
            error=ValueError(pipeline_reason),
            contract_present=isinstance(task_contract, RagTaskContract),
            evidence_status="error",
        )
        await _mark_turn_terminal(
            turn=durable_turn,
            status="failed",
            trace_id=trace_id,
            error_code="runner_contract_rejected",
            evidence_status="error",
            retrieval_executed=False,
        )
        raise HTTPException(
            status_code=503,
            detail={
                "message": "请求执行合同校验失败，请重新发送",
                "error_code": "runner_contract_rejected",
                "same_request_recoverable": False,
                "retry_with_new_request_id": True,
            },
        )
    rag_stream_runner = {
        # Grounded QA starts with the authorized Evidence Retrieval Service.
        # Catalog/result operations use their own typed adapters below.
        "v2": run_rag_v2_stream,
        "direct": run_direct_response_stream,
    }[pipeline_version]
    trace_event(
        "chat.pipeline_selected",
        trace_id=trace_id,
        conversation_id=conv.id,
        user_id=user.id,
        version=pipeline_version,
        reason=pipeline_reason,
    )

    execution_bundle: RagExecutionBundle | None = None
    execution_baseline: ExecutionBaseline | None = None
    query_execution_gate: QueryExecutionGate | None = None
    # A route-selected ConversationContext is an analysis candidate pool, not
    # V2 execution state.  Until an immutable semantic contract is applied,
    # V2 must receive this current-turn-only context even if the route layer
    # had tentatively found a historical turn.
    v2_execution_context = None
    semantic_context_applied = False
    knowledge_request = content_knowledge_request()
    authorized_knowledge_result: AuthorizedKnowledgeResult | None = None
    # This value is intentionally distinct from the raw user question.  It
    # starts as the current turn and may later be replaced by the terminal,
    # source-only rendering of ResolvedTurnSemantics.  It is never fed back
    # into route/planning/model analysis.
    effective_retrieval_query = (
        pipeline_base_query
        or route_continuation_query
            or (
                conversation_context.standalone_query
                if (
                    conversation_context.active_task is not None
                or conversation_context.active_task_scope_mode == "topic_only"
                or has_verified_deterministic_followup_context(
                    conversation_context
                )
            )
            else payload.question
        )
    ).strip()
    # The clarification contract keeps the immutable answer target separate
    # from its retrieval rendering.  Scope selections are applied below as
    # typed task partitions; the punctuation-bearing rendered query is never
    # fed back into either semantic analyzer or local planner.
    semantic_task_query = (
        route_continuation.semantic_query
        if route_continuation is not None
        else effective_retrieval_query
    ).strip()
    verified_followup_baseline = bool(
        conversation_context.active_task is None
        and has_verified_deterministic_followup_context(conversation_context)
    )
    if pipeline_version == "v2" and isinstance(task_contract, RagTaskContract):
        if conversation_context.active_task is not None:
            v2_execution_context = build_active_task_v2_execution_context(
                context=conversation_context,
            )
            semantic_context_applied = True
        elif verified_followup_baseline:
            v2_execution_context = build_verified_followup_v2_execution_context(
                context=conversation_context,
            )
            semantic_context_applied = True
        else:
            v2_execution_context = build_current_turn_v2_execution_context(
                retrieval_query=effective_retrieval_query,
            )
        # The deterministic baseline normally inspects only the current user
        # text.  Pending clarification replies are the deliberate exception:
        # an evidence choice such as ``c2`` is a control value, while a route
        # slot answer such as a product name is only a qualifier.  Preserve
        # the pending task's original question as the immutable plan root.
        # Historical qualifiers outside a pending task are still represented
        # only by validated semantic contracts.
        # The planner consumes the immutable task root.  An
        # explicit applicability answer uses the continuation contract's
        # single-task rendering so the selected scope becomes source-authored
        # semantics without turning into a second answer target.
        execution_question = semantic_task_query
        local_surface_plan = plan_query_locally(execution_question)
        contextual_plan = local_surface_plan
        execution_plan = local_surface_plan
        analysis_route_context = _query_analysis_route_context(
            conversation_context,
            kb_ids=payload.knowledge_base_ids,
        )
        execution_baseline = build_execution_baseline(
            plan=execution_plan,
            local_surface_plan=local_surface_plan,
            contextual_plan=contextual_plan,
            question=execution_question,
            standalone_query=execution_question,
            route_context=analysis_route_context,
            deterministic_is_followup=conversation_context.is_followup,
        )
        if _clarification_answer_supplies_scope(route_continuation):
            execution_baseline = _scope_partitioned_execution_baseline(
                execution_baseline,
                route_continuation,
            )
        execution_bundle = execution_baseline.execution_bundle
        query_execution_gate = evaluate_query_execution_gate(execution_baseline)
        # `intent` is the client-visible routing envelope.  Keep the final V2
        # execution decision next to the task contract so a client can tell a
        # planning result from a closed execution gate without parsing trace
        # event names or inferring state from an assistant sentence.
        intent_payload = {
            **intent_payload,
            "query_execution": {
                **query_execution_gate.to_dict(),
                "semantic_compilation_mode": (
                    "active_task_state"
                    if conversation_context.active_task is not None
                    else (
                        "verified_followup_baseline"
                        if verified_followup_baseline
                        else "local_plan"
                    )
                ),
            },
        }
        trace_event(
            QUERY_EXECUTION_TRACE_EVENT,
            trace_id=trace_id,
            conversation_id=conv.id,
            user_id=user.id,
            pipeline_version="v2",
            execution_surface="api_preflight",
            semantic_compilation_mode="local_plan",
            semantic_compilation_pending=False,
            **query_execution_gate.trace_summary(),
        )
        if query_execution_gate.needs_clarification:
            blocked_contract = _closed_query_execution_task_contract(
                task_contract,
                query_execution_gate,
            )
            trace_event(
                "query.plan",
                trace_id=trace_id,
                pipeline_version="v2",
                execution_surface="api_clarification_gate",
                plan=(
                    execution_baseline.plan.to_dict()
                    if settings.rag_trace_include_content
                    else {
                        "schema_version": execution_baseline.plan.schema_version,
                        "answer_shape": execution_baseline.plan.answer_shape,
                        "query_count": len(execution_baseline.plan.retrieval_queries),
                        "requirement_count": len(execution_baseline.plan.requirements),
                        "confidence": execution_baseline.plan.confidence,
                        "source": execution_baseline.plan.source,
                        # A plan is not rewritten just because the following
                        # execution gate is closed.  Operators need this exact
                        # value to distinguish planner ambiguity from a
                        # non-runnable compatibility bundle.
                        "needs_clarification": (
                            execution_baseline.plan.needs_clarification
                        ),
                    }
                ),
                execution_baseline=execution_baseline.safe_summary(),
                **content_fields(
                    "query",
                    conversation_context.standalone_query,
                ),
            )
            response = await _route_clarification_response(
                db=db,
                conv=conv,
                user=user,
                question=payload.question,
                decision_reason=query_execution_gate.decision_reason,
                trace_id=trace_id,
                selected_kb_ids=payload.knowledge_base_ids,
                task_contract=blocked_contract,
                query_execution_gate=query_execution_gate,
                turn=durable_turn,
                original_query=(route_original_query or execution_question),
                prior_clarification_answers=route_clarification_answers,
            )
            if cleared_evidence_state_id is not None:
                trace_event(
                    "clarification.resolved",
                    trace_id=trace_id,
                    conversation_id=conv.id,
                    user_id=user.id,
                    pending_state_id=cleared_evidence_state_id,
                    resolution="new_question",
                    route_state_revision=conv.route_state_revision,
                )
            return response


    # 保存用户消息并把 durable turn 推进到 generating。用户消息和状态在
    # RAG 流启动前一起提交，因此断线/进程退出后仍能判断该 request_id 已被
    # 接受，不会误触发第二次检索。
    if durable_turn is not None and durable_turn.status == "accepted":
        transition_turn(durable_turn, "generating", trace_id=trace_id)
        renew_turn_lease(durable_turn, owner=trace_id)
    user_metadata = (
        message_turn_metadata(durable_turn, status="generating")
        if durable_turn is not None
        else {}
    )
    user_msg = None
    if durable_turn is not None and durable_turn.user_message_id is not None:
        loaded_user_msg = await db.get(Message, durable_turn.user_message_id)
        if isinstance(loaded_user_msg, Message):
            user_msg = loaded_user_msg
            user_msg.content = user_message_content
            for key, value in user_metadata.items():
                setattr(user_msg, key, value)
    if user_msg is None:
        user_msg = Message(
            id=(
                durable_turn.user_message_id
                if durable_turn is not None
                and durable_turn.user_message_id is not None
                else uuid.uuid4()
            ),
            conversation_id=conv.id,
            role="user",
            content=user_message_content,
            **user_metadata,
        )
        db.add(user_msg)
    if durable_turn is not None:
        durable_turn.user_message_id = user_msg.id
    current_pending = getattr(conv, "pending_route_state", None)
    current_pending_contract = (
        contract_from_dict(current_pending.get("contract"))
        if isinstance(current_pending, dict)
        else None
    )
    if (
        isinstance(current_pending, dict)
        and current_pending_contract is not None
        and current_pending_contract.adapter == "semantic"
    ):
        previous_state_id = current_pending.get("state_id")
        conv.pending_route_state = None
        conv.route_state_revision = int(getattr(conv, "route_state_revision", 0) or 0) + 1
        trace_event(
            "clarification.resolved",
            trace_id=trace_id,
            conversation_id=conv.id,
            user_id=user.id,
            pending_state_id=previous_state_id,
            route_state_revision=conv.route_state_revision,
            relation=(route_decision.relation if route_decision is not None else "legacy"),
        )
    if durable_turn is not None:
        resume_revision, resume_state_id = _pending_route_identity(conv)
        durable_turn.resume_context = {
            "revision": resume_revision,
            "state_id": resume_state_id,
        }
    await db.commit()
    if cleared_evidence_state_id is not None:
        trace_event(
            "clarification.resolved",
            trace_id=trace_id,
            conversation_id=conv.id,
            user_id=user.id,
            pending_state_id=cleared_evidence_state_id,
            resolution="new_question",
            route_state_revision=conv.route_state_revision,
        )

    expected_route_state_revision = int(
        getattr(conv, "route_state_revision", 0) or 0
    )
    expected_active_task_revision = int(
        getattr(conv, "active_task_revision", 0) or 0
    )
    expected_result_reference_revision = int(
        getattr(conv, "result_reference_revision", 0) or 0
    )
    async def generate():
        nonlocal durable_turn, execution_bundle, conversation_context
        nonlocal effective_retrieval_query, semantic_context_applied
        nonlocal v2_execution_context, execution_baseline, query_execution_gate
        nonlocal knowledge_request, rag_stream_runner
        nonlocal authorized_knowledge_result
        knowledge_capability_authorized = False
        execution_clarification_role = QUERY_EXECUTION_UNRESOLVED_ROLE
        execution_clarification_choices: tuple[dict, ...] = ()
        execution_clarification_contract: ClarificationContract | None = None
        full_response = []
        sources = []
        tokens = None
        retrieval_executed = None
        evidence_status = None
        coverage_status = None
        unverified_generation_flag = False
        displayed_result_count = None
        context_evidence_count = None
        hit_count = None
        direct_evidence_count = None
        related_reference_count = None
        evidence_scope_anchor_hit = None
        evidence_scope_anchor_doc_ids: list[str] = []
        # This is deliberately independent from the producer's boolean
        # ``evidence_scope_anchor_hit``.  It is recomputed from DB-refreshed
        # answer sources below and is the only value allowed to resolve a
        # pending evidence scope.
        evidence_source_validation_ok: bool | None = None
        evidence_source_validation_error: str | None = None
        evidence_answer_pairs: set[tuple[uuid.UUID, uuid.UUID]] = set()
        evidence_answer_sources: list[dict] = []
        evidence_source_validation_locked = False
        evidence_source_validation_failure_emitted = False
        pending_done_chunk = None
        clarification_payload = None
        clarification_locked = False
        search_snapshot: dict | None = None
        # 会话和用户消息已提交。先告知前端会话 ID，用户在首条回答完成前停止时也能继续该会话。
        yield "data: " + json.dumps(
            {
                "type": "conversation_started",
                "conversation_id": str(conv.id),
                **(
                    {
                        "turn_id": str(durable_turn.id),
                        "request_id": durable_turn.request_id,
                    }
                    if durable_turn is not None
                    else {}
                ),
            },
            ensure_ascii=False,
        ) + "\n\n"
        # A clear document-metadata request must never degrade into vector
        # content retrieval merely because the optional semantic model timed
        # out or returned an unusable contract.  The shared resource grammar
        # can authorize only the closed catalog capability; RBAC and selected
        # KB scope are still enforced by the catalog runner.
        if (
            pipeline_version == "v2"
            and isinstance(task_contract, RagTaskContract)
            and task_contract.dispatch_authorized
            and not knowledge_capability_authorized
        ):
            catalog_floor = document_catalog_request_for_question(
                route_continuation.canonical_retrieval_query
                if route_continuation is not None
                else semantic_task_query
            )
            if catalog_floor is not None:
                knowledge_request = catalog_floor
                knowledge_capability_authorized = True
                trace_event(
                    "knowledge.capability.floor_selected",
                    trace_id=trace_id,
                    conversation_id=conv.id,
                    user_id=user.id,
                    reason="explicit_document_catalog_surface",
                    capability=knowledge_request.safe_summary(),
                    selected_kb_count=len(set(payload.knowledge_base_ids)),
                )

        # Result-list memory floor: an ordinal reference such as ``我想看第四个``
        # or ``第四个不是《钉钉》吗`` has already been resolved by the conversation
        # layer against the persisted numbered list the user saw.  It reads that
        # exact document directly and must never degrade into vector retrieval of
        # the words ``第四个``, even when the intent model times out.
        if (
            pipeline_version == "v2"
            and isinstance(task_contract, RagTaskContract)
            and task_contract.dispatch_authorized
            and resolved_result_reference is not None
            and resolved_result_reference.source
        ):
            memory_source = resolved_result_reference.source
            authorized_knowledge_result = authorize_knowledge_result(
                [memory_source],
                operation="read",
                answer_form="overview",
                provenance="result_reference_memory",
                acknowledgement=resolved_result_reference.acknowledgement,
            )
            knowledge_capability_authorized = True
            trace_event(
                "knowledge.capability.floor_selected",
                trace_id=trace_id,
                conversation_id=conv.id,
                user_id=user.id,
                reason="result_reference_memory",
                capability=authorized_knowledge_result.safe_summary(),
                reference=resolved_result_reference.safe_summary(),
                selected_kb_count=len(set(payload.knowledge_base_ids)),
            )

        # Resolve applicability ambiguity from the caller's current authorized
        # document catalog before retrieval.  Ranking models may score evidence
        # only after this business-semantic boundary is closed; they may never
        # choose a product version on the user's behalf.
        if (
            pipeline_version == "v2"
            and isinstance(task_contract, RagTaskContract)
            and task_contract.dispatch_authorized
            and execution_baseline is not None
            and query_execution_gate is not None
            and not query_execution_gate.needs_clarification
            and authorized_knowledge_result is None
            and not knowledge_request.is_catalog_operation
            and not knowledge_request.is_result_operation
        ):
            try:
                scope_clarification = await resolve_authorized_scope_clarification(
                    db,
                    plan=execution_baseline.plan,
                    query=effective_retrieval_query,
                    kb_ids=payload.knowledge_base_ids,
                )
            except Exception as exc:
                scope_clarification = None
                trace_event(
                    "query.scope_resolution.failed",
                    trace_id=trace_id,
                    conversation_id=conv.id,
                    user_id=user.id,
                    error=exc,
                )
            if scope_clarification is not None:
                execution_clarification_contract = scope_clarification.to_contract()
                execution_baseline = build_execution_clarification_baseline(
                    baseline=execution_baseline,
                    reason="authorized_scope_ambiguous",
                    clarification_question=(
                        execution_clarification_contract.reason_code
                    ),
                )
                execution_bundle = execution_baseline.execution_bundle
                query_execution_gate = QueryExecutionGate(
                    baseline=execution_baseline,
                    state="needs_clarification",
                    decision_reason="execution_baseline_not_runnable",
                    unresolved_reason="ambiguous",
                )
                execution_clarification_role = scope_clarification.dimension
                execution_clarification_choices = tuple(
                    {
                        "key": choice.key,
                        "label": choice.label,
                        "value": choice.version,
                    }
                    for choice in scope_clarification.choices
                )
                trace_event(
                    "query.scope_resolution.clarification",
                    trace_id=trace_id,
                    conversation_id=conv.id,
                    user_id=user.id,
                    dimension=scope_clarification.dimension,
                    reason=scope_clarification.reason,
                    choice_count=len(scope_clarification.choices),
                    selected_kb_count=len(set(payload.knowledge_base_ids)),
                )

        # A closed local gate is terminal.  Do not hand a ``not_ready`` bundle
        # to the retrieval runner: that would reinterpret task ambiguity as
        # evidence-scope ambiguity.
        if (
            pipeline_version == "v2"
            and query_execution_gate is not None
            and query_execution_gate.needs_clarification
        ):
            if not isinstance(task_contract, RagTaskContract):
                raise ValueError("closed V2 query execution gate requires a task contract")
            if execution_baseline is None:
                raise ValueError("closed V2 query execution gate requires an execution baseline")
            blocked_contract = _closed_query_execution_task_contract(
                task_contract,
                query_execution_gate,
                unresolved_role=execution_clarification_role,
            )
            plan_trace = (
                execution_baseline.plan.to_dict()
                if trace_include_content
                else {
                    "schema_version": execution_baseline.plan.schema_version,
                    "answer_shape": execution_baseline.plan.answer_shape,
                    "query_count": len(execution_baseline.plan.retrieval_queries),
                    "requirement_count": len(execution_baseline.plan.requirements),
                    "confidence": execution_baseline.plan.confidence,
                    "source": execution_baseline.plan.source,
                    "needs_clarification": execution_baseline.plan.needs_clarification,
                }
            )
            trace_event(
                "query.plan",
                trace_id=trace_id,
                conversation_id=conv.id,
                user_id=user.id,
                pipeline_version="v2",
                execution_surface="api_post_semantic_gate",
                plan=plan_trace,
                execution_baseline=execution_baseline.safe_summary(),
                **content_fields("query", conversation_context.standalone_query),
            )
            trace_event(
                QUERY_EXECUTION_TRACE_EVENT,
                trace_id=trace_id,
                conversation_id=conv.id,
                user_id=user.id,
                pipeline_version="v2",
                execution_surface="api_post_semantic_gate",
                semantic_compilation_mode=(
                    str(
                        intent_payload.get("query_execution", {}).get(
                            "semantic_compilation_mode", "local_plan"
                        )
                    )
                    if isinstance(intent_payload.get("query_execution"), dict)
                    else "local_plan"
                ),
                semantic_compilation_pending=False,
                **query_execution_gate.trace_summary(),
            )
            clarification_response = await _route_clarification_response(
                db=db,
                conv=conv,
                user=user,
                question=payload.question,
                decision_reason=query_execution_gate.decision_reason,
                trace_id=trace_id,
                selected_kb_ids=payload.knowledge_base_ids,
                task_contract=blocked_contract,
                query_execution_gate=query_execution_gate,
                turn=durable_turn,
                existing_user_message=user_msg,
                parent_stream_logging=True,
                original_query=(route_original_query or effective_retrieval_query),
                prior_clarification_answers=route_clarification_answers,
                clarification_choices=execution_clarification_choices,
                clarification_contract=execution_clarification_contract,
            )
            async for clarification_chunk in clarification_response.body_iterator:
                clarification_text = (
                    clarification_chunk.decode()
                    if isinstance(clarification_chunk, bytes)
                    else clarification_chunk
                )
                clarification_event = _parse_sse_payload(clarification_text)
                if (
                    clarification_event is not None
                    and clarification_event.get("type") == "conversation_started"
                ):
                    # This generator already published the durable conversation
                    # identity before the post-stream gate began.
                    continue
                yield clarification_text
            return
        if durable_turn is not None:
            yield "data: " + json.dumps(
                _turn_state_event(durable_turn), ensure_ascii=False
            ) + "\n\n"
        try:
            catalog_execution = knowledge_request.is_catalog_operation
            result_execution = (
                authorized_knowledge_result is not None
                or knowledge_request.is_result_operation
            )
            result_sources: tuple[dict[str, Any], ...] = ()
            if catalog_execution or result_execution:
                if pipeline_version != "v2" or not knowledge_capability_authorized:
                    raise ValueError(
                        "direct knowledge execution requires an authorized capability"
                    )
                if result_execution:
                    if authorized_knowledge_result is not None:
                        trace_event(
                            "knowledge.result_reference.memory_bound",
                            trace_id=trace_id,
                            conversation_id=conv.id,
                            user_id=user.id,
                            source_count=len(
                                authorized_knowledge_result.sources
                            ),
                            provenance=authorized_knowledge_result.provenance,
                            reference=(
                                resolved_result_reference.safe_summary()
                                if resolved_result_reference is not None
                                else None
                            ),
                        )
                    else:
                        result_sources = resolve_result_reference_sources(
                            context=conversation_context,
                            handles=knowledge_request.result_handles,
                            kb_ids=payload.knowledge_base_ids,
                        )
                        authorized_knowledge_result = authorize_knowledge_result(
                            result_sources,
                            operation=knowledge_request.operation,
                            answer_form=knowledge_request.answer_form,
                            provenance="route_result_handles",
                        )
                    rag_stream_runner = run_knowledge_result_stream
                else:
                    rag_stream_runner = run_knowledge_catalog_stream
                trace_event(
                    "knowledge.capability.dispatch_selected",
                    trace_id=trace_id,
                    conversation_id=conv.id,
                    user_id=user.id,
                    capability=(
                        authorized_knowledge_result.safe_summary()
                        if result_execution
                        and authorized_knowledge_result is not None
                        else knowledge_request.safe_summary()
                    ),
                    selected_kb_count=len(set(payload.knowledge_base_ids)),
                )
            if pipeline_version == "v2" and not (catalog_execution or result_execution):
                if v2_execution_context is None:
                    raise ValueError("V2 pipeline is missing an execution context")
                if (
                    v2_execution_context.semantic_context_applied
                    != semantic_context_applied
                ):
                    raise ValueError("V2 semantic context state is inconsistent")
                trace_event(
                    "query.semantics.execution_context",
                    trace_id=trace_id,
                    conversation_id=conv.id,
                    user_id=user.id,
                    pipeline_version="v2",
                    **v2_execution_context.safe_summary(),
                )
            rag_stream_kwargs = {
                "question": (
                    pipeline_base_query
                    or route_continuation_query
                    or payload.question
                ),
                "kb_ids": payload.knowledge_base_ids,
                "search_config": search_config,
                "conversation_id": str(conv.id),
                "db": db,
                "intent": intent_payload,
                "task_contract": task_contract,
                "trace_id": trace_id,
                # V2 receives only the source-anchored canonical retrieval
                # rendering.  V1 keeps its legacy rewritten query during the
                # rollout, but no V2 planning/model stage consumes it.
                "standalone_query": (
                    v2_execution_context.retrieval_query
                    if pipeline_version == "v2" and v2_execution_context is not None
                    else conversation_context.standalone_query
                ),
                # V2 can receive historical material only through the
                # immutable ResolvedTurnSemantics projection.  The route
                # candidate context is deliberately unavailable here on
                # analysis timeout/rejection/capacity fallback.
                "conversation_history": (
                    list(v2_execution_context.conversation_history)
                    if pipeline_version == "v2" and v2_execution_context is not None
                    else (
                        list(conversation_context.history_messages)
                        if conversation_context.is_followup
                        else []
                    )
                ),
                "carryover_sources": (
                    list(v2_execution_context.carryover_sources)
                    if pipeline_version == "v2" and v2_execution_context is not None
                    else list(conversation_context.carryover_sources)
                ),
                "is_followup": (
                    v2_execution_context.is_followup
                    if pipeline_version == "v2" and v2_execution_context is not None
                    else conversation_context.is_followup
                ),
                "followup_reason": (
                    v2_execution_context.followup_reason
                    if pipeline_version == "v2" and v2_execution_context is not None
                    else conversation_context.followup_reason
                ),
                "evidence_scope_filter": evidence_filter,
            }
            if catalog_execution or result_execution:
                if result_execution:
                    rag_stream_kwargs["authorized_result"] = (
                        authorized_knowledge_result
                    )
                else:
                    rag_stream_kwargs["knowledge_request"] = knowledge_request
            elif pipeline_version == "v2" and execution_bundle is not None:
                # This single immutable handoff makes the compiled DAG the
                # production execution authority.  V1/direct callers retain
                # their historical signatures during the rollout.
                rag_stream_kwargs["execution_bundle"] = execution_bundle
                rag_stream_kwargs["active_task_scope"] = (
                    conversation_context.active_task
                    if conversation_context.active_task_scope_mode == "document"
                    else None
                )
                # The request session owns turn/message state and must never be
                # shared by concurrent DAG retrieval workers.  V2 receives a
                # factory for short-lived read sessions; V1/direct signatures
                # remain unchanged during the rollout.
                rag_stream_kwargs["task_read_session_factory"] = TaskReadSessionLocal
            rag_stream = rag_stream_runner(
                **rag_stream_kwargs,
            )
            async for chunk in rag_stream:
                data = _parse_sse_payload(chunk)
                event_type = data.get("type") if data else None
                source_validation_delta: str | None = None
                # done 必须等 AI 消息持久化成功后再发给前端；其它事件保持实时流式。
                if event_type == "done":
                    pending_done_chunk = chunk
                    continue
                if event_type == "text_delta":
                    if evidence_source_validation_locked:
                        # A failed source validation emits one trusted technical
                        # state; any later producer delta is suppressed.
                        continue
                    full_response.append(str(data.get("content") or ""))
                elif event_type == "clarification_state":
                    if clarification_payload is not None:
                        continue
                    clarification_contract = contract_from_dict(data)
                    if (
                        clarification_contract is None
                        or data.get("status") != "proposed"
                    ):
                        raise ValueError("Pipeline 返回了无效的统一澄清合同")
                    clarification_payload = clarification_contract.to_dict(
                        public=False
                    )
                    clarification_locked = True
                    # The clarification event is the final generation gate.
                    # Fail closed even if a rolling/custom producer emitted a
                    # contradictory hit status or answer_sources beforehand.
                    evidence_status = "needs_clarification"
                    sources = []
                    context_evidence_count = 0
                    hit_count = 0
                    direct_evidence_count = 0
                    full_response.clear()
                    yield (
                        "data: "
                        + json.dumps(
                            proposed_clarification_event(clarification_contract),
                            ensure_ascii=False,
                        )
                        + "\n\n"
                    )
                    continue
                elif event_type == "search_results":
                    if clarification_locked:
                        previous_retrieval_executed = retrieval_executed
                        data = _clarification_locked_search_results(
                            data,
                            trace_id=trace_id,
                        )
                        if previous_retrieval_executed is not None:
                            data["retrieval_executed"] = previous_retrieval_executed
                        chunk = (
                            "data: "
                            + json.dumps(data, ensure_ascii=False)
                            + "\n\n"
                        )
                    retrieval_executed = data.get("retrieval_executed")
                    raw_anchor_hit = data.get("evidence_scope_anchor_hit")
                    evidence_scope_anchor_hit = (
                        raw_anchor_hit if isinstance(raw_anchor_hit, bool) else None
                    )
                    raw_anchor_doc_ids = data.get("evidence_scope_anchor_doc_ids")
                    evidence_scope_anchor_doc_ids = (
                        [str(value) for value in raw_anchor_doc_ids]
                        if isinstance(raw_anchor_doc_ids, list)
                        else []
                    )
                    raw_evidence_status = data.get("evidence_status")
                    raw_coverage_status = data.get("coverage_status")
                    coverage_status = (
                        str(raw_coverage_status).strip().casefold()
                        if raw_coverage_status is not None
                        else coverage_status
                    )
                    # A rolling V1/custom producer may still emit the legacy
                    # ``version_mismatch`` spelling.  Normalize once at the
                    # stream trust boundary so SSE, turn/message persistence
                    # and traces converge on canonical ``scope_mismatch``.
                    normalized_evidence_status = canonical_evidence_status(
                        raw_evidence_status
                    )
                    evidence_status = (
                        normalized_evidence_status
                        if normalized_evidence_status is not None
                        else str(raw_evidence_status or "").strip().casefold()
                    )
                    raw_unverified_generation = data.get("unverified_generation")
                    if isinstance(raw_unverified_generation, bool):
                        unverified_generation_flag = raw_unverified_generation
                    direct_evidence_count = data.get("direct_evidence_count")
                    if (
                        isinstance(direct_evidence_count, bool)
                        or not isinstance(direct_evidence_count, int)
                        or direct_evidence_count < 0
                    ):
                        direct_evidence_count = 0
                    related_reference_count = data.get("related_reference_count")
                    display_results = data.get("results")
                    if not isinstance(display_results, list):
                        display_results = []
                    raw_answer_sources = data.get("answer_sources")
                    has_answer_source_list = isinstance(raw_answer_sources, list)
                    evidence_answer_pairs = set()
                    evidence_answer_sources = []
                    answer_source_required = (
                        normalized_evidence_status
                        in _ANSWER_SOURCE_REQUIRED_STATUSES
                    )
                    evidence_status_invalid = normalized_evidence_status is None
                    non_answer_source_status = (
                        normalized_evidence_status in _NON_ANSWER_SOURCE_STATUSES
                    )
                    if non_answer_source_status:
                        # 状态是服务端最终证据门控：即使异常/旧版/自定义流生产者
                        # 同时错误携带了 answer_sources 和非零 direct 数，也不得
                        # 把这些正文保存成历史回答依据。持久化层必须 fail closed，
                        # 不能只依赖正常 Pipeline 或前端隐藏。
                        direct_evidence_count = 0
                    if evidence_status_invalid:
                        raw_answer_sources = []
                        evidence_source_validation_ok = False
                        evidence_source_validation_error = (
                            "evidence_status_invalid"
                        )
                    elif not has_answer_source_list:
                        # Fail closed for rolling upgrades and custom stream
                        # producers: a missing answer_sources list cannot be
                        # treated as the generation context.
                        raw_answer_sources = []
                        evidence_source_validation_ok = False
                        evidence_source_validation_error = (
                            "answer_sources_not_a_list"
                        )
                    elif non_answer_source_status:
                        # Non-answer states never persist producer snapshots,
                        # even if an old producer accidentally included them.
                        if raw_answer_sources:
                            evidence_source_validation_ok = False
                            evidence_source_validation_error = (
                                "non_answer_status_with_sources"
                            )
                        else:
                            evidence_source_validation_ok = True
                            evidence_source_validation_error = None
                        raw_answer_sources = []
                    else:
                        (
                            answer_source_items,
                            evidence_answer_pairs,
                            evidence_source_validation_error,
                        ) = await _validate_stream_answer_sources(
                            db,
                            raw_sources=raw_answer_sources,
                            raw_results=display_results,
                            selected_kb_ids=payload.knowledge_base_ids,
                            read_session_factory=TaskReadSessionLocal,
                            allow_unverified=bool(
                                normalized_evidence_status
                                in {"partial", "unverified"}
                                and data.get("unverified_generation") is True
                                and str(
                                    data.get("source_verification") or ""
                                ).strip().casefold() == "unverified"
                            ),
                        )
                        if (
                            evidence_filter is not None
                            and evidence_source_validation_error is None
                        ):
                            allowed_scope_pairs = _scope_document_pairs(
                                evidence_filter
                            )
                            if not evidence_answer_pairs.issubset(
                                allowed_scope_pairs
                            ):
                                evidence_source_validation_error = (
                                    "answer_source_scope_forbidden"
                                )
                            elif any(
                                not _source_matches_scope_filter(
                                    source,
                                    evidence_filter,
                                )
                                for source in answer_source_items
                            ):
                                evidence_source_validation_error = (
                                    "answer_source_scope_slice_forbidden"
                                )
                        if (
                            conversation_context.active_task is not None
                            and evidence_source_validation_error is None
                        ):
                            active_doc_ids = set(
                                conversation_context.active_task.doc_ids
                            )
                            active_kb_ids = set(
                                conversation_context.active_task.kb_ids
                            )
                            if any(
                                doc_id not in active_doc_ids
                                or kb_id not in active_kb_ids
                                for kb_id, doc_id in evidence_answer_pairs
                            ):
                                evidence_source_validation_error = (
                                    "answer_source_active_task_scope_forbidden"
                                )
                        evidence_source_validation_ok = (
                            evidence_source_validation_error is None
                        )
                        # If refresh/identity validation fails, do not retain
                        # producer-provided body or metadata.  The validation
                        # lock below replaces the model stream with one
                        # deterministic retry message and keeps pending scope.
                        raw_answer_sources = answer_source_items
                        evidence_answer_sources = list(answer_source_items)
                        if evidence_source_validation_error is not None:
                            raw_answer_sources = []
                            answer_source_items = []
                            evidence_answer_sources = []
                            evidence_answer_pairs = set()
                            if normalized_evidence_status not in {
                                "error",
                                "no_hit",
                                "skipped",
                                "needs_clarification",
                            }:
                                evidence_status = "error"
                                normalized_evidence_status = "error"
                    recomputed_anchor_hit, recomputed_anchor_doc_ids = (
                        _scope_anchor_coverage_from_sources(
                            evidence_filter,
                            evidence_answer_pairs,
                            evidence_answer_sources,
                        )
                    )
                    if evidence_filter is not None:
                        # Ignore producer-advertised anchor fields entirely.
                        evidence_scope_anchor_hit = recomputed_anchor_hit
                        evidence_scope_anchor_doc_ids = recomputed_anchor_doc_ids
                    else:
                        evidence_scope_anchor_hit = False
                        evidence_scope_anchor_doc_ids = []
                    displayed_result_count = data.get(
                        "displayed_result_count",
                        data.get("total", len(display_results)),
                    )
                    if (
                        isinstance(displayed_result_count, bool)
                        or not isinstance(displayed_result_count, int)
                        or displayed_result_count < 0
                    ):
                        displayed_result_count = len(display_results)
                    answer_source_items = [
                        source
                        for source in (raw_answer_sources or [])[:20]
                        if isinstance(source, dict)
                    ]
                    if evidence_source_validation_error is not None:
                        # Once producer evidence fails identity/authorization
                        # validation, its broad panel snapshot is untrusted too.
                        # Do not expose stale or cross-scope content through the
                        # UI while merely clearing the persisted citation list.
                        display_results = []
                        data["results"] = []
                        data["total"] = 0
                        data["displayed_result_count"] = 0
                        data["related_reference_count"] = 0
                        displayed_result_count = 0
                        related_reference_count = 0
                    source_validation_failed = bool(
                        evidence_status_invalid
                        or evidence_source_validation_error is not None
                        or (answer_source_required and not answer_source_items)
                    )
                    if (
                        not clarification_locked
                        and (
                            source_validation_failed
                            or evidence_source_validation_locked
                        )
                    ):
                        if evidence_source_validation_error is None:
                            evidence_source_validation_error = (
                                "required_answer_sources_empty"
                            )
                        display_results = []
                        data["results"] = []
                        data["total"] = 0
                        data["displayed_result_count"] = 0
                        data["related_reference_count"] = 0
                        displayed_result_count = 0
                        related_reference_count = 0
                        evidence_source_validation_ok = False
                        evidence_source_validation_locked = True
                        evidence_answer_pairs = set()
                        evidence_scope_anchor_hit = False
                        evidence_scope_anchor_doc_ids = []
                        evidence_status = "error"
                        normalized_evidence_status = "error"
                        answer_source_items = []
                        raw_answer_sources = []
                        sources = []
                        context_evidence_count = 0
                        hit_count = 0
                        direct_evidence_count = 0
                        full_response.clear()
                        full_response.append(
                            _EVIDENCE_SOURCE_VALIDATION_FAILURE_MESSAGE
                        )
                        if not evidence_source_validation_failure_emitted:
                            source_validation_delta = (
                                "data: "
                                + json.dumps(
                                    {
                                        "type": "text_delta",
                                        "content": (
                                            _EVIDENCE_SOURCE_VALIDATION_FAILURE_MESSAGE
                                        ),
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n\n"
                            )
                            evidence_source_validation_failure_emitted = True
                    # Counts are derived from the validated context snapshot,
                    # never copied from a producer that may have counted broad
                    # candidates or stale sources.
                    direct_evidence_count = sum(
                        str(source.get("evidence_role") or "")
                        .strip()
                        .casefold()
                        == "direct"
                        for source in answer_source_items
                    )
                    data["answer_sources"] = answer_source_items
                    data["answer_source_count"] = len(answer_source_items)
                    data["context_evidence_count"] = len(answer_source_items)
                    data["direct_evidence_count"] = direct_evidence_count
                    data["hit_count"] = direct_evidence_count
                    data["evidence_scope_anchor_hit"] = (
                        evidence_scope_anchor_hit
                    )
                    data["evidence_scope_anchor_doc_ids"] = list(
                        evidence_scope_anchor_doc_ids
                    )
                    data["evidence_status"] = evidence_status
                    # Keep only a bounded identity/ranking snapshot for durable
                    # retries/history.  Message.sources remains the current
                    # validated citation payload and is re-authorized on read.
                    search_snapshot = _bounded_search_snapshot(data)
                    chunk = (
                        "data: "
                        + json.dumps(data, ensure_ascii=False)
                        + "\n\n"
                    )
                    context_evidence_count = len(answer_source_items)
                    hit_count = direct_evidence_count
                    source_meta = {
                        "trace_id": data.get("trace_id") or trace_id,
                        "retrieval_executed": bool(retrieval_executed),
                        "evidence_status": evidence_status,
                        "displayed_result_count": displayed_result_count or 0,
                        "context_evidence_count": context_evidence_count,
                        "hit_count": hit_count,
                        "direct_evidence_count": direct_evidence_count or 0,
                        "related_reference_count": related_reference_count or 0,
                        "is_followup": bool(data.get("is_followup")),
                        "carryover_source_count": data.get("carryover_source_count") or 0,
                    }
                    # 右侧检索面板继续消费 ``results``，但历史回答与引用只保存
                    # Pipeline 实际送入 generation.context 的 answer_sources。
                    # 两者不能再共用一份宽候选列表。
                    sources = [
                        {**source, **source_meta}
                        for source in answer_source_items
                    ]
                elif event_type == "usage":
                    tokens = data.get("total_tokens")
                yield chunk
                if source_validation_delta is not None:
                    yield source_validation_delta
                if (
                    event_type == "search_results"
                    and evidence_source_validation_locked
                ):
                    # ``search_results`` is emitted before final generation in
                    # both pipelines.  Once its source identity/authorization
                    # contract fails, pulling one more item would let the
                    # producer open an LLM stream whose output is guaranteed
                    # to be discarded.  Stop at the gate, close the suspended
                    # async generator, persist the deterministic failure
                    # message below, and emit ``done`` only after that commit.
                    trace_event(
                        "generation.skipped",
                        trace_id=trace_id,
                        pipeline_version=pipeline_version,
                        reason="answer_source_validation_failed",
                        evidence_status=evidence_status,
                        validation_error=evidence_source_validation_error,
                    )
                    close_stream = getattr(rag_stream, "aclose", None)
                    if callable(close_stream):
                        try:
                            await close_stream()
                        except Exception as close_exc:
                            # The answer is already fail-closed.  Cleanup is
                            # best-effort and must not replace it with a generic
                            # stream error or delay persistence/retry guidance.
                            trace_event(
                                "chat.stream_close_error",
                                trace_id=trace_id,
                                conversation_id=conv.id,
                                stage="answer_source_validation",
                                error=close_exc,
                            )
                    break
        except asyncio.CancelledError:
            # 浏览器停止生成、断开连接或服务关闭都会取消流协程。同步入队即可
            # 立即把调用链标成 interrupted；不要在已取消任务里继续等待数据库。
            trace_event(
                "chat.cancelled",
                trace_id=trace_id,
                conversation_id=conv.id,
                user_id=user.id,
                stage="streaming",
                retrieval_executed=retrieval_executed,
                evidence_status=evidence_status,
                displayed_result_count=displayed_result_count or 0,
                answer_source_count=context_evidence_count or 0,
                context_evidence_count=context_evidence_count or 0,
                hit_count=hit_count or 0,
                **content_fields("partial_answer", "".join(full_response)),
            )
            try:
                await asyncio.shield(
                    _mark_turn_terminal(
                        turn=durable_turn,
                        status="cancelled",
                        trace_id=trace_id,
                        error_code="stream_cancelled",
                        evidence_status=evidence_status,
                        retrieval_executed=retrieval_executed,
                    )
                )
            except Exception:
                pass
            raise
        except Exception as e:
            log_exception_safely(
                logger,
                "[chat/stream error] trace=%s conv=%s",
                trace_id,
                conv.id,
                exc=e,
            )
            if retrieval_executed is None:
                retrieval_executed = bool(decision.need_retrieval)
            if evidence_status is None:
                evidence_status = "error" if decision.need_retrieval else "skipped"
            if hit_count is None:
                hit_count = 0
            if context_evidence_count is None:
                context_evidence_count = 0
            if displayed_result_count is None:
                displayed_result_count = 0
            trace_event(
                "chat.error",
                trace_id=trace_id,
                conversation_id=conv.id,
                user_id=user.id,
                error=e,
                retrieval_executed=retrieval_executed,
                evidence_status=evidence_status,
                displayed_result_count=displayed_result_count,
                answer_source_count=context_evidence_count,
                context_evidence_count=context_evidence_count,
                hit_count=hit_count,
                direct_evidence_count=direct_evidence_count or 0,
                related_reference_count=related_reference_count or 0,
                **content_fields("partial_answer", "".join(full_response)),
            )
            await _mark_turn_terminal(
                turn=durable_turn,
                status="failed",
                trace_id=trace_id,
                error_code="stream_failed",
                evidence_status=evidence_status,
                retrieval_executed=retrieval_executed,
            )
            from database import AsyncSessionLocal
            if routing_result.route_log_id is not None:
                try:
                    async with AsyncSessionLocal() as save_db:
                        route_log = await save_db.get(IntentRouteLog, routing_result.route_log_id)
                        if route_log is not None:
                            route_log.retrieval_executed = retrieval_executed
                            route_log.evidence_status = evidence_status
                            route_log.hit_count = hit_count
                            await save_db.commit()
                except Exception as persistence_exc:
                    # 路由统计是 best-effort；失败不能覆盖真正的模型/检索错误，也不能
                    # 阻断随后发给前端的安全 error + done 事件。
                    log_exception_safely(
                        logger,
                        "[chat/error route-log persistence] trace=%s conv=%s",
                        trace_id,
                        conv.id,
                        exc=persistence_exc,
                    )
                    trace_event(
                        "chat.persistence_error",
                        trace_id=trace_id,
                        conversation_id=conv.id,
                        operation="update_route_log_after_stream_error",
                        error=persistence_exc,
                    )
            yield f"data: {json.dumps({'type': 'error', 'message': _public_stream_error_message(e)}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'conversation_id': str(conv.id)})}\n\n"
            return

        # 保存 AI 回复
        answer = "".join(full_response)
        from database import AsyncSessionLocal
        created_pending_state = None
        resolved_pending_state_id = None
        persisted_active_state = None
        persisted_route_state_revision = expected_route_state_revision
        # Stage the generated payload first, then complete the transcript in a
        # retryable transaction.  Any later pending-route CAS or statistics
        # failure can no longer roll the assistant answer back.
        if durable_turn is not None:
            try:
                staged_sources = [
                    item
                    for source in sources[:20]
                    if (item := _bounded_source_identity_snapshot(source)) is not None
                ]
                durable_turn = await _stage_generated_turn(
                    turn=durable_turn,
                    conv=conv,
                    answer=answer,
                    sources=staged_sources,
                    tokens=tokens,
                    evidence_status=evidence_status,
                    retrieval_executed=retrieval_executed,
                    trace_id=trace_id,
                    search_snapshot=search_snapshot,
                )
                durable_turn = await _recover_turn_answer(
                    conv=conv,
                    turn=durable_turn,
                )
            except asyncio.CancelledError:
                await _mark_turn_terminal(
                    turn=durable_turn,
                    status="cancelled",
                    trace_id=trace_id,
                    error_code="response_persistence_cancelled",
                    evidence_status=evidence_status,
                    retrieval_executed=retrieval_executed,
                )
                raise
            except Exception as exc:
                payload_recoverable = await _mark_turn_persist_failed(
                    turn=durable_turn,
                    trace_id=trace_id,
                )
                trace_event(
                    "chat.persistence_error",
                    trace_id=trace_id,
                    conversation_id=conv.id,
                    operation="stage_or_complete_turn",
                    error=exc,
                    **content_fields("answer", answer),
                )
                yield "data: " + json.dumps(
                    {
                        "type": "error",
                        "message": (
                            "回答已生成但暂时无法完成保存，请使用相同 request_id 重试"
                            if payload_recoverable
                            else (
                                "回答已生成，但恢复数据未能保存。"
                                "请重新发送问题并使用新的 request_id。"
                            )
                        ),
                        "request_id": durable_turn.request_id,
                        "turn_id": str(durable_turn.id),
                        "status": durable_turn.status,
                        "same_request_recoverable": payload_recoverable,
                        "persistence_status": "failed",
                        "retry_with_new_request_id": not payload_recoverable,
                    },
                    ensure_ascii=False,
                ) + "\n\n"
                yield "data: " + json.dumps(
                    {
                        "type": "done",
                        "conversation_id": str(conv.id),
                        "request_id": durable_turn.request_id,
                        "turn_id": str(durable_turn.id),
                        "status": durable_turn.status,
                        "persistence_status": "failed",
                        "same_request_recoverable": payload_recoverable,
                        "retry_with_new_request_id": not payload_recoverable,
                    },
                    ensure_ascii=False,
                ) + "\n\n"
                return
        try:
            async with AsyncSessionLocal() as save_db:
                ai_msg = None
                if durable_turn is not None and durable_turn.assistant_message_id:
                    loaded_ai_msg = await save_db.get(
                        Message, durable_turn.assistant_message_id
                    )
                    if isinstance(loaded_ai_msg, Message):
                        ai_msg = loaded_ai_msg
                if ai_msg is None:
                    ai_msg = Message(
                        id=(
                            durable_turn.assistant_message_id
                            if durable_turn is not None
                            and durable_turn.assistant_message_id is not None
                            else uuid.uuid4()
                        ),
                        conversation_id=conv.id,
                        role="assistant",
                        content=answer,
                        sources=sources,
                        tokens=tokens,
                        **(
                            message_turn_metadata(
                                durable_turn, status="completed"
                            )
                            if durable_turn is not None
                            else {}
                        ),
                    )
                    save_db.add(ai_msg)

                new_pending_state = None
                if clarification_payload is not None:
                    new_pending_state = _clarification_event_pending_state(
                        clarification_payload,
                        original_query=(
                            effective_retrieval_query
                            or conversation_context.standalone_query
                        ),
                        selected_kb_ids=payload.knowledge_base_ids,
                        base_user_message_id=user_msg.id,
                        clarification_message_id=ai_msg.id,
                    )
                    if new_pending_state is None:
                        raise ValueError("Pipeline 返回了无效的统一澄清合同")

                normalized_final_evidence_status = str(
                    evidence_status or ""
                ).strip().casefold()
                selected_scope_completed = bool(
                    (evidence_filter is not None or evidence_refinement_active)
                    and evidence_pending_state is not None
                    and pending_done_chunk is not None
                    and retrieval_executed is True
                    and normalized_final_evidence_status
                    in _SUCCESSFUL_EVIDENCE_SCOPE_STATUSES
                    and evidence_source_validation_ok is True
                    and not evidence_source_validation_locked
                    and (
                        evidence_refinement_active
                        or evidence_scope_anchor_hit is True
                    )
                )
                # A fresh valid clarification replaces the old state.  Without
                # one, technical/skip outcomes are persisted as assistant
                # feedback but leave the previous choice pending for retry.
                should_update_route_state = bool(
                    new_pending_state is not None or selected_scope_completed
                )
                # A fully closed hit is always memorable.  An unverified
                # auto-answer (the 50%-80% tier, single-document selection) is
                # also memorable: the sources are real, re-authorized
                # evidence, and the entity memory derived from them is exactly
                # what the next standalone-but-related turn needs.  Without
                # this, the earlier turn would answer, then the next turn
                # would forget it and re-enter the confirmation loop.
                active_task_eligible = bool(
                    normalized_final_evidence_status in {"hit", "partial"}
                    and (
                        normalized_final_evidence_status == "hit"
                        or unverified_generation_flag
                    )
                )
                should_update_active_task = bool(
                    durable_turn is not None
                    and active_task_eligible
                    and (
                        coverage_status == "complete"
                        or normalized_final_evidence_status == "partial"
                        or not any(
                            requirement.role == "answer"
                            and requirement.requires_collection_closure
                            for requirement in (
                                execution_bundle.plan.requirements
                                if execution_bundle is not None
                                else ()
                            )
                        )
                    )
                    and evidence_source_validation_ok is True
                    and not evidence_source_validation_locked
                    and sources
                    and execution_bundle is not None
                )
                # A numbered document catalog answer is the result list the
                # user sees.  It is persisted separately from the active task:
                # ``我想看第四个`` must resolve against this exact ordered list,
                # not against whatever the newest turn happened to read.
                should_update_result_reference_memory = bool(
                    durable_turn is not None
                    and knowledge_request is not None
                    and knowledge_request.is_catalog_operation
                    and knowledge_request.operation in {"list", "count"}
                    and normalized_final_evidence_status == "hit"
                    and evidence_source_validation_ok is True
                    and not evidence_source_validation_locked
                    and sources
                )
                persisted_conv = None
                if (
                    should_update_route_state
                    or should_update_active_task
                    or should_update_result_reference_memory
                ):
                    persisted_conv = await save_db.get(Conversation, conv.id)
                    if persisted_conv is None:
                        raise RuntimeError("会话不存在，无法保存会话执行状态")
                if should_update_route_state and persisted_conv is not None:
                    persisted_revision = int(
                        getattr(persisted_conv, "route_state_revision", 0) or 0
                    )
                    if persisted_revision != expected_route_state_revision:
                        raise RuntimeError("会话范围状态已被其他请求更新")
                    persisted_pending = _active_pending_route_state(
                        getattr(persisted_conv, "pending_route_state", None)
                    )
                    expected_pending_id = (
                        str(evidence_pending_state.get("state_id"))
                        if evidence_pending_state is not None
                        else None
                    )
                    persisted_pending_id = (
                        str(persisted_pending.get("state_id"))
                        if persisted_pending is not None
                        else None
                    )
                    if persisted_pending_id != expected_pending_id:
                        raise RuntimeError("待选择的证据范围已发生变化")

                    if expected_pending_id is not None:
                        resolved_pending_state_id = expected_pending_id
                    if new_pending_state is not None:
                        persisted_conv.pending_route_state = new_pending_state
                        created_pending_state = new_pending_state
                    else:
                        persisted_conv.pending_route_state = None
                    persisted_conv.route_state_revision = persisted_revision + 1
                    persisted_route_state_revision = persisted_conv.route_state_revision

                if should_update_active_task and persisted_conv is not None:
                    persisted_active_revision = int(
                        getattr(persisted_conv, "active_task_revision", 0) or 0
                    )
                    if persisted_active_revision != expected_active_task_revision:
                        raise RuntimeError("会话任务状态已被其他请求更新")
                    previous_active = parse_active_task_state(
                        getattr(persisted_conv, "active_task_state", None)
                    )
                    task_root_query = (
                        str(evidence_pending_state.get("original_query") or "").strip()
                        if evidence_pending_state is not None
                        else (
                            conversation_context.active_task.state.root_query
                            if conversation_context.active_task is not None
                            else effective_retrieval_query
                        )
                    )
                    semantic_memory = extract_resolved_entity_memory(
                        sources=sources,
                        question=(
                            user_message_content
                            or payload.question
                            or task_root_query
                        ),
                        source_turn_id=durable_turn.id,
                        trace_id=trace_id,
                    )
                    active_state = build_active_task_state(
                        root_query=task_root_query,
                        answer_shape=execution_bundle.plan.answer_shape,
                        sources=sources,
                        source_turn_id=durable_turn.id,
                        trace_id=trace_id,
                        previous_revision=(
                            previous_active.revision
                            if previous_active is not None
                            else persisted_active_revision
                        ),
                        semantic_memory=semantic_memory,
                    )
                    persisted_conv.active_task_state = active_state.to_dict()
                    persisted_conv.active_task_revision = (
                        persisted_active_revision + 1
                    )
                    persisted_active_state = active_state

                if (
                    should_update_result_reference_memory
                    and persisted_conv is not None
                ):
                    persisted_result_revision = int(
                        getattr(
                            persisted_conv,
                            "result_reference_revision",
                            0,
                        )
                        or 0
                    )
                    if (
                        persisted_result_revision
                        != expected_result_reference_revision
                    ):
                        raise RuntimeError("会话结果列表状态已被其他请求更新")
                    previous_memory = parse_result_reference_memory(
                        getattr(
                            persisted_conv,
                            "result_reference_memory",
                            None,
                        )
                    )
                    result_reference_root_query = str(
                        effective_retrieval_query or ""
                    ).strip() or payload.question
                    result_memory = build_result_reference_memory(
                        root_query=result_reference_root_query,
                        list_label="知识库文档目录",
                        items=sources[:20],
                        source_turn_id=durable_turn.id,
                        trace_id=trace_id,
                        previous_revision=(
                            previous_memory.revision
                            if previous_memory is not None
                            else persisted_result_revision
                        ),
                    )
                    persisted_conv.result_reference_memory = (
                        result_memory.to_dict()
                    )
                    persisted_conv.result_reference_revision = (
                        persisted_result_revision + 1
                    )
                    trace_event(
                        "conversation.result_reference_memory_persisted",
                        trace_id=trace_id,
                        conversation_id=conv.id,
                        user_id=user.id,
                        memory=result_memory.safe_summary(),
                    )

                if (
                    durable_turn is None
                    and routing_result.route_log_id is not None
                ):
                    route_log = await save_db.get(IntentRouteLog, routing_result.route_log_id)
                    if route_log is not None:
                        route_log.retrieval_executed = (
                            bool(retrieval_executed)
                            if retrieval_executed is not None
                            else bool(decision.need_retrieval)
                        )
                        route_log.evidence_status = evidence_status or (
                            "no_hit" if decision.need_retrieval else "skipped"
                        )
                        route_log.hit_count = int(hit_count or 0)
                await save_db.commit()
            if persisted_active_state is not None:
                trace_event(
                    "conversation.active_task_persisted",
                    trace_id=trace_id,
                    conversation_id=conv.id,
                    user_id=user.id,
                    task=persisted_active_state.safe_summary(),
                )
            if resolved_pending_state_id is not None:
                trace_event(
                    "clarification.resolved",
                    trace_id=trace_id,
                    conversation_id=conv.id,
                    user_id=user.id,
                    pending_state_id=resolved_pending_state_id,
                    resolution=(
                        "compare_all"
                        if evidence_filter and evidence_filter.get("mode") == "compare_all"
                        else (
                            "refined"
                            if evidence_refinement_active
                            else "selected"
                        )
                    ),
                    selected_choice_keys=[
                        choice.get("key")
                        for choice in (evidence_filter or {}).get("choices", [])
                    ],
                    evidence_scope_anchor_hit=evidence_scope_anchor_hit,
                    evidence_scope_anchor_doc_ids=evidence_scope_anchor_doc_ids,
                    route_state_revision=persisted_route_state_revision,
                )
            if created_pending_state is not None:
                trace_event(
                    "clarification.created",
                    trace_id=trace_id,
                    conversation_id=conv.id,
                    user_id=user.id,
                    pending_state_id=created_pending_state["state_id"],
                    dimension=contract_from_dict(
                        created_pending_state["contract"]
                    ).dimension,
                    choice_count=len(
                        contract_from_dict(
                            created_pending_state["contract"]
                        ).choices
                    ),
                    selected_kb_count=len(
                        created_pending_state["selected_kb_ids_snapshot"]
                    ),
                    route_state_revision=persisted_route_state_revision,
                    **content_fields(
                        "original_query",
                        created_pending_state["original_query"],
                    ),
                    clarification_contract=created_pending_state["contract"],
                )
            # 只有回答和路由统计真正提交成功后才把 Trace 标成 success，避免
            # “调用链成功但历史消息不存在”的竞态。
            trace_event(
                "chat.response",
                trace_id=trace_id,
                conversation_id=conv.id,
                user_id=user.id,
                evidence_status=evidence_status,
                retrieval_executed=retrieval_executed,
                displayed_result_count=displayed_result_count or 0,
                answer_source_count=context_evidence_count or 0,
                context_evidence_count=context_evidence_count or 0,
                hit_count=hit_count or 0,
                direct_evidence_count=direct_evidence_count,
                related_reference_count=related_reference_count,
                evidence_source_validation_ok=evidence_source_validation_ok,
                evidence_source_validation_error=evidence_source_validation_error,
                evidence_source_validation_locked=evidence_source_validation_locked,
                tokens=tokens,
                sources=[
                    {
                        "doc_id": source.get("doc_id"),
                        "chunk_id": source.get("id"),
                        "evidence_role": source.get("evidence_role"),
                        "constraint_status": source.get("constraint_status"),
                        "retrieval_score": source.get("retrieval_score"),
                        "effective_score": source.get("score"),
                        "answer_support": source.get("answer_support"),
                        **content_fields(
                            "filename",
                            str(source.get("filename") or ""),
                        ),
                    }
                    for source in sources
                ],
                **content_fields("answer", answer),
            )
        except asyncio.CancelledError:
            trace_event(
                "chat.cancelled",
                trace_id=trace_id,
                conversation_id=conv.id,
                user_id=user.id,
                stage="response_persistence",
                retrieval_executed=retrieval_executed,
                evidence_status=evidence_status,
                displayed_result_count=displayed_result_count or 0,
                answer_source_count=context_evidence_count or 0,
                context_evidence_count=context_evidence_count or 0,
                hit_count=hit_count or 0,
                **content_fields("partial_answer", answer),
            )
            await _mark_turn_terminal(
                turn=durable_turn,
                status="cancelled",
                trace_id=trace_id,
                error_code="response_persistence_cancelled",
                evidence_status=evidence_status,
                retrieval_executed=retrieval_executed,
            )
            raise
        except Exception as exc:
            log_exception_safely(
                logger,
                "[chat/persistence error] trace=%s conv=%s",
                trace_id,
                conv.id,
                exc=exc,
            )
            if durable_turn is not None and durable_turn.status == "completed":
                # The transcript is already durable.  This exception can only
                # come from the independent pending-route/statistics phase.
                # Never emit an ACK for its rolled-back pending state.
                created_pending_state = None
                resolved_pending_state_id = None
                trace_event(
                    "chat.persistence_error",
                    trace_id=trace_id,
                    conversation_id=conv.id,
                    operation="post_answer_state_update",
                    answer_persisted=True,
                    error=exc,
                )
            else:
                trace_event(
                    "chat.persistence_error",
                    trace_id=trace_id,
                    conversation_id=conv.id,
                    error=exc,
                    **content_fields("answer", answer),
                )
                yield f"data: {json.dumps({'type': 'error', 'message': '回答已生成，但保存失败，请重试'})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'conversation_id': str(conv.id)})}\n\n"
                return
        if durable_turn is not None and routing_result.route_log_id is not None:
            try:
                async with AsyncSessionLocal() as stats_db:
                    route_log = await stats_db.get(
                        IntentRouteLog, routing_result.route_log_id
                    )
                    if isinstance(route_log, IntentRouteLog):
                        route_log.retrieval_executed = (
                            bool(retrieval_executed)
                            if retrieval_executed is not None
                            else bool(decision.need_retrieval)
                        )
                        route_log.evidence_status = evidence_status or (
                            "no_hit" if decision.need_retrieval else "skipped"
                        )
                        route_log.hit_count = int(hit_count or 0)
                        await stats_db.commit()
            except Exception as stats_exc:
                log_exception_safely(
                    logger,
                    "[chat/route statistics best-effort] trace=%s conv=%s",
                    trace_id,
                    conv.id,
                    exc=stats_exc,
                )
                trace_event(
                    "chat.persistence_error",
                    trace_id=trace_id,
                    conversation_id=conv.id,
                    operation="update_route_statistics",
                    answer_persisted=True,
                    error=stats_exc,
                )
        if created_pending_state is not None:
            # The proposed state is streamed when ambiguity is discovered;
            # choices become active only after message and state commit.
            yield (
                "data: "
                + json.dumps(
                    public_clarification_event(
                        created_pending_state,
                        route_state_revision=persisted_route_state_revision,
                        conversation_id=conv.id,
                        persisted=True,
                    ),
                    ensure_ascii=False,
                )
                + "\n\n"
            )
        if durable_turn is not None:
            yield "data: " + json.dumps(
                _turn_state_event(durable_turn), ensure_ascii=False
            ) + "\n\n"
        if pending_done_chunk is not None:
            yield pending_done_chunk
        else:
            yield f"data: {json.dumps({'type': 'done', 'conversation_id': str(conv.id)})}\n\n"

    return StreamingResponse(
        stream_in_conversation_log(generate(), conversation_id=conv.id),
        media_type="text/event-stream",
        headers=_turn_response_headers(
            conversation_id=conv.id,
            trace_id=trace_id,
            turn=durable_turn,
        ),
    )


@router.post("/send")
async def send_message(
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_permission(CHAT_USE)),
):
    """Prepare and stream one chat turn under a single lifecycle owner."""

    lifecycle = _ChatRequestLifecycle(trace_id=uuid.uuid4().hex)
    try:
        response = await _prepare_send_message(
            payload,
            db,
            user,
            lifecycle=lifecycle,
        )
    except asyncio.CancelledError as exc:
        await asyncio.shield(lifecycle.finish(
            status="cancelled",
            error_code="request_preparation_cancelled",
            stage="request_preparation",
            error=exc,
        ))
        raise
    except Exception as exc:
        await lifecycle.finish(
            status="failed",
            error_code="request_preparation_failed",
            stage="request_preparation",
            error=exc,
        )
        raise
    response.body_iterator = lifecycle.stream(response.body_iterator)
    return response


@router.get("/history", response_model=list[ConversationOut])
async def get_history(
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_permission(CHAT_USE)),
):
    offset = (page - 1) * page_size
    stmt = select(Conversation).order_by(Conversation.created_at.desc())
    # 非超管只看自己的会话；超管可见全部
    if not user.is_superadmin:
        stmt = stmt.where(Conversation.user_id == user.id)
    rows = (await db.execute(
        stmt.offset(offset).limit(page_size)
    )).scalars().all()
    return rows


@router.get("/{conv_id}/messages", response_model=list[MessageOut])
async def get_messages(
    conv_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_permission(CHAT_USE)),
):
    conv = await db.get(Conversation, conv_id)
    if not conv or (not user.is_superadmin and conv.user_id != user.id):
        raise HTTPException(status_code=404, detail="会话不存在")
    rows = (await db.execute(
        select(Message)
        .where(Message.conversation_id == conv_id)
        .order_by(Message.created_at)
    )).scalars().all()
    return await _messages_with_current_source_scope(
        rows,
        user=user,
        db=db,
        pending_route_state=getattr(conv, "pending_route_state", None),
        route_state_revision=getattr(conv, "route_state_revision", 0),
    )


@router.patch("/{conv_id}", response_model=ConversationOut)
async def rename_conversation(
    conv_id: uuid.UUID,
    payload: ConversationRenameRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_permission(CHAT_USE)),
):
    """更新会话标题；沿用读取/删除时的归属校验。"""
    conv = await db.get(Conversation, conv_id)
    if not conv or (not user.is_superadmin and conv.user_id != user.id):
        raise HTTPException(status_code=404, detail="会话不存在")

    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="会话标题不能为空")

    conv.title = title
    await db.commit()
    await db.refresh(conv)
    return conv


@router.post("/batch-delete")
async def delete_conversations_batch(
    payload: ConversationBatchDeleteRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_permission(CHAT_USE)),
):
    """Atomically delete an authorised, bounded set of conversations."""

    requested_ids = tuple(dict.fromkeys(payload.conversation_ids))
    stmt = select(Conversation).where(Conversation.id.in_(requested_ids))
    if not user.is_superadmin:
        stmt = stmt.where(Conversation.user_id == user.id)
    conversations = (await db.execute(stmt.with_for_update())).scalars().all()
    conversations_by_id = {item.id: item for item in conversations}

    # Fail the whole request when any id is missing or outside the caller's
    # scope.  This both prevents partial destructive results and avoids
    # disclosing whether an inaccessible conversation exists.
    if any(conv_id not in conversations_by_id for conv_id in requested_ids):
        raise HTTPException(
            status_code=404,
            detail="一个或多个会话不存在或无权操作",
        )

    # The foreign keys own transcript/turn cleanup.  One set-based delete
    # avoids loading every message relationship into application memory.
    await db.execute(
        sa_delete(Conversation)
        .where(Conversation.id.in_(requested_ids))
        .execution_options(synchronize_session=False)
    )
    await db.commit()
    return {
        "message": "批量删除成功",
        "deleted_count": len(requested_ids),
        "deleted_ids": [str(conv_id) for conv_id in requested_ids],
    }


@router.delete("/{conv_id}")
async def delete_conversation(
    conv_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_permission(CHAT_USE)),
):
    conv = await db.get(Conversation, conv_id)
    if not conv or (not user.is_superadmin and conv.user_id != user.id):
        raise HTTPException(status_code=404, detail="会话不存在")
    await db.delete(conv)
    await db.commit()
    return {"message": "删除成功"}
