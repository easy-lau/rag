"""Direct, non-retrieval response runner for compiled chat contracts.

This module deliberately does not import or call either RAG pipeline.  A
validated task contract is the only execution input, so general chat, inline
writing and platform help cannot silently pass through the legacy retrieval
stack when the deployment is configured for RAG v2.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from core.llm_stream import stream_with_retry_before_first_delta
from core.openai_client import get_client
from core.query_route_compiler import (
    RagTaskContract,
    require_rag_task_contract_dispatchable,
)
from core.rag_trace import content_fields, json_safe, trace_event


DIRECT_RUNNER_VERSION = "direct_v1"
_DIRECT_RESPONSE_MODES = {"general_chat", "writing", "platform_help"}
_HISTORY_ROLES = {"user", "assistant"}


def _sse(payload: Mapping[str, Any]) -> str:
    return (
        "data: "
        + json.dumps(json_safe(dict(payload)), ensure_ascii=False, allow_nan=False)
        + "\n\n"
    )


def _step_event(step: str, status: str) -> str:
    return _sse({"type": "search_step", "step": step, "status": status})


def _remaining_timeout(*, deadline: float, stage_timeout: float) -> float:
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise asyncio.TimeoutError("direct_response_workflow_deadline_exhausted")
    return min(float(stage_timeout), remaining)


def _bounded_history(
    values: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    total_chars = 0
    for raw in values or ():
        if not isinstance(raw, Mapping):
            continue
        role = str(raw.get("role") or "").strip()
        content = str(raw.get("content") or "").strip()
        if role not in _HISTORY_ROLES or not content:
            continue
        content = content[:2000]
        if total_chars + len(content) > 6000:
            break
        history.append({"role": role, "content": content})
        total_chars += len(content)
        if len(history) >= 6:
            break
    return history


def _system_prompt(response_mode: str) -> str:
    if response_mode == "writing":
        return (
            "你是专业的中文写作助手。只根据用户本轮明确提供的内容和目标完成润色、"
            "改写、总结、翻译或起草。不得声称查询过知识库，不得编造制度、金额、参数或"
            "来源；如果任务缺少完成所必需的原文，应先请用户补充。输出应直接可用、结构清晰。"
        )
    if response_mode == "platform_help":
        return (
            "你是当前企业 RAG 检索平台的使用助手。仅说明选择知识库、提问、查看候选与"
            "采用证据、管理会话等当前平台能力；不得把其他业务系统当成本平台，也不得"
            "虚构未提供的按钮、权限或后台功能。涉及企业制度和业务资料时，应提示用户选择"
            "有权限的知识库后查询。"
        )
    return (
        "你是专业的通用对话助手。准确、简洁地回应用户；不得声称查询过企业知识库。"
        "如果问题涉及企业内部制度、流程、金额、账号、配置或文档事实，应明确说明需要"
        "通过知识库核验，而不是使用常识猜测。"
    )


async def run_direct_response_stream(
    question: str,
    kb_ids: list[uuid.UUID],
    search_config: dict,
    conversation_id: str,
    db: AsyncSession,
    intent: dict | None = None,
    trace_id: str | None = None,
    standalone_query: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    carryover_sources: list[dict] | None = None,
    is_followup: bool = False,
    followup_reason: str | None = None,
    task_contract: RagTaskContract | None = None,
    evidence_scope_filter: dict | None = None,
) -> AsyncGenerator[str, None]:
    """Generate a direct answer without entering a retrieval implementation."""

    del db, search_config, followup_reason
    trace_id = trace_id or uuid.uuid4().hex
    if task_contract is None:
        raise ValueError("direct response requires a compiled task contract")
    require_rag_task_contract_dispatchable(
        task_contract,
        selected_kb_count=len(set(kb_ids)),
    )
    if task_contract.need_retrieval or task_contract.retrieval_policy != "skip":
        raise ValueError("direct response cannot execute a retrieval contract")
    if task_contract.response_mode not in _DIRECT_RESPONSE_MODES:
        raise ValueError("unsupported direct response mode")
    if evidence_scope_filter is not None or carryover_sources:
        raise ValueError("direct response cannot consume evidence scope or sources")

    settings = get_settings()
    started_at = time.perf_counter()
    workflow_timeout = float(
        getattr(settings, "rag_v2_generation_workflow_timeout_seconds", 60.0)
    )
    deadline = started_at + workflow_timeout
    history = _bounded_history(conversation_history if is_followup else ())
    user_question = (standalone_query or question).strip() or question.strip()

    yield _step_event("analyze", "active")
    if intent:
        yield _sse({"type": "intent", "decision": intent})
    trace_event(
        "direct.plan",
        trace_id=trace_id,
        runner_version=DIRECT_RUNNER_VERSION,
        response_mode=task_contract.response_mode,
        retrieval_policy=task_contract.retrieval_policy,
        history_message_count=len(history),
        is_followup=is_followup,
        **content_fields("question", user_question),
    )
    yield _step_event("analyze", "done")
    yield _step_event("expand", "active")
    yield _step_event("expand", "done")
    yield _step_event("retrieve", "active")
    trace_event(
        "retrieval.completed",
        trace_id=trace_id,
        runner_version=DIRECT_RUNNER_VERSION,
        executed=False,
        succeeded=True,
        candidate_count=0,
        elapsed_ms=0,
    )
    yield _step_event("retrieve", "done")
    yield _step_event("rerank", "active")
    trace_event(
        "rerank.completed",
        trace_id=trace_id,
        runner_version=DIRECT_RUNNER_VERSION,
        requested=False,
        attempted=False,
        executed=False,
        succeeded=None,
        candidate_count=0,
        elapsed_ms=0,
        reason="retrieval_skipped",
    )
    yield _step_event("rerank", "done")
    yield _sse({
        "type": "search_results",
        "trace_id": trace_id,
        "results": [],
        "answer_sources": [],
        "total": 0,
        "displayed_result_count": 0,
        "context_evidence_count": 0,
        "answer_source_count": 0,
        "hit_count": 0,
        "direct_evidence_count": 0,
        "related_reference_count": 0,
        "retrieval_executed": False,
        "evidence_status": "skipped",
        "coverage_status": "insufficient",
        "decision_reason": task_contract.decision_reason,
        "pipeline_version": DIRECT_RUNNER_VERSION,
    })

    system_prompt = _system_prompt(task_contract.response_mode)
    messages = [{"role": "system", "content": system_prompt}, *history]
    messages.append({"role": "user", "content": user_question})
    trace_event(
        "generation.context",
        trace_id=trace_id,
        runner_version=DIRECT_RUNNER_VERSION,
        evidence_status="skipped",
        response_mode=task_contract.response_mode,
        retrieval_policy="skip",
        model=settings.chat_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        workflow_timeout_seconds=workflow_timeout,
        history_message_count=len(history),
        context_sources=[],
        **content_fields("context", ""),
    )

    yield _step_event("generate", "active")
    create_kwargs = {
        "model": settings.chat_model,
        "messages": messages,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "stream": True,
    }
    client = get_client().with_options(max_retries=0)

    async def open_stream():
        request_timeout = _remaining_timeout(
            deadline=deadline,
            stage_timeout=float(settings.llm_request_timeout_seconds),
        )
        return await client.chat.completions.create(
            **create_kwargs,
            timeout=request_timeout,
        )

    usage = None
    finish_reason = None
    answer_chars = 0
    prompt_chars = sum(len(message["content"]) for message in messages)
    stream = stream_with_retry_before_first_delta(
        open_stream,
        model=settings.chat_model,
        prompt_chars=prompt_chars,
        timeout_seconds=min(
            float(settings.llm_request_timeout_seconds),
            workflow_timeout,
        ),
        max_attempts=settings.llm_max_attempts,
        retry_base_delay_seconds=settings.llm_retry_base_delay_seconds,
    )
    try:
        while True:
            try:
                chunk = await asyncio.wait_for(
                    anext(stream),
                    timeout=_remaining_timeout(
                        deadline=deadline,
                        stage_timeout=workflow_timeout,
                    ),
                )
            except StopAsyncIteration:
                break
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason
            delta = str(
                getattr(getattr(choice, "delta", None), "content", "") or ""
            )
            if delta:
                answer_chars += len(delta)
                yield _sse({"type": "text_delta", "content": delta})
    finally:
        await stream.aclose()

    yield _step_event("generate", "done")
    if usage is not None:
        yield _sse({
            "type": "usage",
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        })
    trace_event(
        "generation.completed",
        trace_id=trace_id,
        runner_version=DIRECT_RUNNER_VERSION,
        model=settings.chat_model,
        answer_chars=answer_chars,
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
        finish_reason=finish_reason,
        generation_ms=round((time.perf_counter() - started_at) * 1000),
        total_ms=round((time.perf_counter() - started_at) * 1000),
    )
    yield _sse({"type": "done", "conversation_id": conversation_id})


__all__ = ["DIRECT_RUNNER_VERSION", "run_direct_response_stream"]
