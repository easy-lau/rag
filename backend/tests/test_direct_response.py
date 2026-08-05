import json
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

from core.direct_response import run_direct_response_stream
from core.query_route_compiler import CompiledAnswerRequirement, RagTaskContract
from core.query_route_compiler import TaskContractDispatchError
from core.query_route_contract import RouteClarification


def _contract(response_mode: str) -> RagTaskContract:
    action = {
        "general_chat": "chat",
        "writing": "writing",
        "platform_help": "system_help",
    }[response_mode]
    return RagTaskContract(
        schema_version="rag_task_contract.v1",
        route_schema_version="rag_route_decision.v1",
        readiness="ready",
        intent_code=action,
        intent_name=action,
        action=action,
        confidence=0.99,
        source="local",
        relation="new",
        evidence_scope="current_input",
        query_mode="current",
        context_turn_keys=(),
        response_mode=response_mode,
        retrieval_policy="skip",
        need_retrieval=False,
        dispatch_authorized=True,
        decision_reason="test_direct",
        selected_kb_count=0,
        requirements=(
            CompiledAnswerRequirement(
                id="r1",
                role="answer",
                origin="user_text",
                description="完成当前请求",
                importance="required",
                source="explicit",
            ),
        ),
        clarification=RouteClarification(question=""),
    )


def _settings():
    return SimpleNamespace(
        chat_model="test-chat",
        temperature=0,
        max_tokens=128,
        llm_request_timeout_seconds=1,
        llm_max_attempts=1,
        llm_retry_base_delay_seconds=0,
        rag_v2_generation_workflow_timeout_seconds=1,
    )


class _CompletionStream:
    def __init__(self, text: str):
        self.text = text

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        yield SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content=self.text),
                finish_reason="stop",
            )],
            usage=None,
        )
        yield SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(
                prompt_tokens=12,
                completion_tokens=3,
                total_tokens=15,
            ),
        )


class _Completions:
    def __init__(self, text: str):
        self.text = text
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _CompletionStream(self.text)


class _Client:
    def __init__(self, text: str = "直接回答"):
        self.completions = _Completions(text)
        self.chat = SimpleNamespace(completions=self.completions)

    def with_options(self, **_kwargs):
        return self


def _payloads(chunks: list[str]) -> list[dict]:
    return [
        json.loads(chunk.removeprefix("data: ").strip())
        for chunk in chunks
    ]


class DirectResponseRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def _run(
        self,
        response_mode: str,
        *,
        history=None,
        is_followup: bool = False,
    ):
        client = _Client()
        with (
            patch("core.direct_response.get_settings", return_value=_settings()),
            patch("core.direct_response.get_client", return_value=client),
            patch("core.direct_response.trace_event"),
        ):
            chunks = [
                chunk
                async for chunk in run_direct_response_stream(
                    question="请处理这个请求",
                    kb_ids=[],
                    search_config={},
                    conversation_id=str(uuid.uuid4()),
                    db=SimpleNamespace(),
                    task_contract=_contract(response_mode),
                    conversation_history=history,
                    is_followup=is_followup,
                    trace_id="trace-direct",
                )
            ]
        return _payloads(chunks), client

    async def test_direct_modes_skip_every_retrieval_surface(self) -> None:
        for response_mode in ("general_chat", "writing", "platform_help"):
            with self.subTest(response_mode=response_mode):
                payloads, client = await self._run(response_mode)
                search = next(
                    item for item in payloads if item["type"] == "search_results"
                )
                self.assertFalse(search["retrieval_executed"])
                self.assertEqual(search["evidence_status"], "skipped")
                self.assertEqual(search["results"], [])
                self.assertEqual(search["answer_sources"], [])
                process = next(
                    item for item in payloads if item["type"] == "search_process"
                )
                self.assertEqual(process["execution_path"], "direct")
                self.assertEqual(
                    [step["key"] for step in process["steps"]],
                    ["analyze", "generate"],
                )
                active_steps = [
                    item["step"]
                    for item in payloads
                    if item["type"] == "search_step" and item["status"] == "active"
                ]
                self.assertEqual(active_steps, ["analyze", "generate"])
                self.assertEqual(
                    [item["content"] for item in payloads if item["type"] == "text_delta"],
                    ["直接回答"],
                )
                self.assertEqual(len(client.completions.calls), 1)

    async def test_history_is_sent_only_for_confirmed_followup(self) -> None:
        history = [
            {"role": "user", "content": "上一问"},
            {"role": "assistant", "content": "上一答"},
        ]
        _, new_client = await self._run(
            "general_chat",
            history=history,
            is_followup=False,
        )
        _, followup_client = await self._run(
            "general_chat",
            history=history,
            is_followup=True,
        )

        new_messages = new_client.completions.calls[0]["messages"]
        followup_messages = followup_client.completions.calls[0]["messages"]
        self.assertEqual([item["role"] for item in new_messages], ["system", "user"])
        self.assertEqual(
            [item["role"] for item in followup_messages],
            ["system", "user", "assistant", "user"],
        )

    async def test_retrieval_contract_is_rejected_instead_of_silently_skipped(self) -> None:
        contract = _contract("general_chat")
        retrieval_contract = RagTaskContract(
            **{
                **contract.__dict__,
                "response_mode": "writing",
                "retrieval_policy": "required",
                "need_retrieval": True,
            }
        )

        with self.assertRaises(TaskContractDispatchError):
            _ = [
                chunk
                async for chunk in run_direct_response_stream(
                    question="根据知识库写摘要",
                    kb_ids=[],
                    search_config={},
                    conversation_id=str(uuid.uuid4()),
                    db=SimpleNamespace(),
                    task_contract=retrieval_contract,
                )
            ]


if __name__ == "__main__":
    unittest.main()
