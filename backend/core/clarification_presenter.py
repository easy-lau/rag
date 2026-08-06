"""Stream natural clarification wording from the configured answer model.

The structured clarification contract is authoritative.  Model output is a
presentation layer only: it cannot add choices, identifiers or permissions.
If generation fails, callers keep serving the structured state and emit no
synthetic business sentence.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator

from config import get_settings
from core.clarification import ClarificationContract
from core.llm_stream import stream_with_retry_before_first_delta
from core.openai_client import get_client
from core.rag_trace import content_fields, trace_event
from core.structured_output import create_stream_completion


_PRESENTER_MAX_TOKENS = 384


def _prompt(contract: ClarificationContract, original_query: str) -> list[dict[str, str]]:
    public_contract = contract.to_dict(public=True)
    payload = {
        "original_query": str(original_query or "").strip()[:4000],
        "clarification": public_contract,
    }
    return [
        {
            "role": "system",
            "content": (
                "你负责把服务端给出的结构化澄清状态表达成自然、简洁的中文。"
                "只说明当前缺少的条件，并自然邀请用户选择或补充；不得回答原问题，"
                "不得新增、删除、合并、排序或推测候选项，不得输出知识库、文档或权限标识。"
                "有候选项时不要重复打印编号清单，因为界面会单独展示按钮；没有候选项时"
                "只提出一个具体且可回答的补充问题。不要使用固定套话，不要解释系统内部实现。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        },
    ]


async def stream_clarification_text(
    *,
    contract: ClarificationContract,
    original_query: str,
    trace_id: str,
) -> AsyncGenerator[str, None]:
    settings = get_settings()
    model = settings.chat_model
    messages = _prompt(contract, original_query)
    timeout_seconds = min(
        float(settings.llm_request_timeout_seconds),
        float(getattr(settings, "rag_v2_generation_workflow_timeout_seconds", 60.0)),
    )
    started_at = time.perf_counter()
    trace_event(
        "clarification.presentation.started",
        trace_id=trace_id,
        model=model,
        adapter=contract.adapter,
        dimension=contract.dimension,
        reason_code=contract.reason_code,
        choice_count=len(contract.choices),
        **content_fields("original_query", original_query),
    )
    emitted_chars = 0
    try:
        client = get_client().with_options(max_retries=0)
        request = {
            "model": model,
            "messages": messages,
            "temperature": min(max(float(settings.temperature), 0.0), 0.3),
            "max_tokens": min(max(int(settings.max_tokens), 1), _PRESENTER_MAX_TOKENS),
            "stream": True,
            "timeout": timeout_seconds,
        }

        async def open_stream():
            opened_stream, _ = await create_stream_completion(
                client,
                request=request,
                provider_identity=getattr(settings, "llm_base_url", ""),
                model=model,
            )
            return opened_stream

        stream = stream_with_retry_before_first_delta(
            open_stream,
            model=model,
            prompt_chars=sum(len(message["content"]) for message in messages),
            timeout_seconds=timeout_seconds,
            max_attempts=settings.llm_max_attempts,
            retry_base_delay_seconds=settings.llm_retry_base_delay_seconds,
        )
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(anext(stream), timeout=timeout_seconds)
                except StopAsyncIteration:
                    break
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = str(
                    getattr(getattr(choices[0], "delta", None), "content", "") or ""
                )
                if not delta:
                    continue
                emitted_chars += len(delta)
                yield delta
        finally:
            await stream.aclose()
    except Exception as exc:
        trace_event(
            "clarification.presentation.failed",
            trace_id=trace_id,
            model=model,
            adapter=contract.adapter,
            dimension=contract.dimension,
            emitted_chars=emitted_chars,
            error=exc,
        )
        return
    trace_event(
        "clarification.presentation.completed",
        trace_id=trace_id,
        model=model,
        adapter=contract.adapter,
        dimension=contract.dimension,
        emitted_chars=emitted_chars,
        generation_ms=round((time.perf_counter() - started_at) * 1000),
    )


__all__ = ["stream_clarification_text"]
