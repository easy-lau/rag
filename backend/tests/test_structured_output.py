import asyncio
import unittest
from types import SimpleNamespace

from core.structured_output import (
    clear_structured_output_capability_cache,
    create_stream_completion,
    create_structured_completion,
)


class _Unsupported(Exception):
    status_code = 400

    def __init__(self, detail):
        super().__init__(detail)
        self.body = {"error": {"message": detail}}


class _Completions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        mode = kwargs.get("response_format", {}).get("type")
        if mode == "json_schema":
            raise _Unsupported("This response_format type is unavailable now")
        if mode == "json_object":
            raise _Unsupported("response_format json_object is not supported")
        return SimpleNamespace(ok=True)


class _Client:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_Completions())


class _JsonKeywordCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        mode = kwargs.get("response_format", {}).get("type")
        if mode == "json_schema":
            raise _Unsupported("This response_format type is unavailable now")
        prompt = "\n".join(
            str(message.get("content") or "")
            for message in kwargs.get("messages", [])
            if isinstance(message, dict)
        ).casefold()
        if mode == "json_object" and "json" not in prompt:
            raise _Unsupported(
                "Prompt must contain the word 'json' in some form to use "
                "response_format of type json_object"
            )
        return SimpleNamespace(ok=True)


class _JsonKeywordClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_JsonKeywordCompletions())


class _AxonHubUnsupported(Exception):
    def __init__(self):
        super().__init__(
            "Bad Request, error: This response_format type is unavailable now, "
            "code: invalid_request_error, type: invalid_request_error"
        )
        self.body = {
            "error": {
                "message": "This response_format type is unavailable now",
                "code": "invalid_request_error",
            }
        }


class _AxonHubCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("response_format", {}).get("type") == "json_schema":
            raise _AxonHubUnsupported()
        return SimpleNamespace(ok=True)


class _AxonHubClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_AxonHubCompletions())


class _SlowSchemaCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("response_format", {}).get("type") == "json_schema":
            await asyncio.sleep(0.2)
        return SimpleNamespace(ok=True)


class _SlowSchemaClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_SlowSchemaCompletions())


class _SuccessfulCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(ok=True)


class _SuccessfulClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_SuccessfulCompletions())


class _ThinkingUnsupportedCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if "thinking" in kwargs.get("extra_body", {}):
            raise _Unsupported("unknown parameter: thinking")
        return SimpleNamespace(ok=True)


class _ThinkingUnsupportedClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_ThinkingUnsupportedCompletions())


class StructuredOutputTests(unittest.TestCase):
    def setUp(self):
        clear_structured_output_capability_cache()

    def test_negotiates_to_plain_json_and_reuses_capability(self):
        async def run():
            client = _Client()
            first = await create_structured_completion(
                client,
                request={"model": "model-a", "messages": [{"role": "user", "content": "json"}]},
                strict_response_format={"type": "json_schema", "json_schema": {"name": "x"}},
                timeout_seconds=1,
                provider_identity="https://llm.example/v1",
                model="model-a",
            )
            second = await create_structured_completion(
                client,
                request={"model": "model-a", "messages": [{"role": "user", "content": "json"}]},
                strict_response_format={"type": "json_schema", "json_schema": {"name": "x"}},
                timeout_seconds=1,
                provider_identity="https://llm.example/v1",
                model="model-a",
            )
            return first, second, client.chat.completions.calls

        first, second, calls = asyncio.run(run())
        self.assertEqual(first.mode, "plain_json")
        self.assertEqual(first.attempted_modes, ("json_schema", "json_object", "plain_json"))
        self.assertEqual(second.mode, "plain_json")
        self.assertEqual(len(calls), 4)
        self.assertNotIn("response_format", calls[-1])

    def test_json_object_fallback_adds_json_instruction_for_gateway_requirement(self):
        async def run():
            client = _JsonKeywordClient()
            result = await create_structured_completion(
                client,
                request={
                    "model": "model-a",
                    "messages": [{"role": "system", "content": "Return the contract."}],
                },
                strict_response_format={"type": "json_schema", "json_schema": {"name": "x"}},
                timeout_seconds=1,
                provider_identity="https://llm.example/v1",
                model="model-a",
            )
            return result, client.chat.completions.calls

        result, calls = asyncio.run(run())
        self.assertEqual(result.mode, "json_object")
        self.assertEqual(result.attempted_modes, ("json_schema", "json_object"))
        fallback_messages = calls[-1]["messages"]
        self.assertEqual(calls[-1]["response_format"], {"type": "json_object"})
        self.assertTrue(
            any(
                message.get("role") == "system"
                and "json" in str(message.get("content") or "").casefold()
                for message in fallback_messages
                if isinstance(message, dict)
            )
        )

    def test_axonhub_wrapped_400_schema_error_downgrades(self):
        async def run():
            client = _AxonHubClient()
            result = await create_structured_completion(
                client,
                request={
                    "model": "model-a",
                    "messages": [{"role": "system", "content": "Return JSON."}],
                },
                strict_response_format={"type": "json_schema", "json_schema": {"name": "x"}},
                timeout_seconds=1,
                provider_identity="https://llm.example/v1",
                model="model-a",
            )
            return result, client.chat.completions.calls

        result, calls = asyncio.run(run())
        self.assertEqual(result.mode, "json_object")
        self.assertEqual(result.attempted_modes, ("json_schema", "json_object"))
        self.assertEqual(calls[-1]["response_format"], {"type": "json_object"})

    def test_schema_timeout_retries_with_json_object_budget(self):
        async def run():
            client = _SlowSchemaClient()
            result = await create_structured_completion(
                client,
                request={
                    "model": "model-a",
                    "messages": [{"role": "system", "content": "Return JSON."}],
                },
                strict_response_format={"type": "json_schema", "json_schema": {"name": "x"}},
                timeout_seconds=0.1,
                provider_identity="https://llm.example/v1",
                model="model-a",
            )
            return result, client.chat.completions.calls

        result, calls = asyncio.run(run())
        self.assertEqual(result.mode, "json_object")
        self.assertEqual(result.attempted_modes, ("json_schema", "json_object"))
        self.assertEqual(calls[-1]["response_format"], {"type": "json_object"})

    def test_declared_capability_disables_thinking_for_structured_request(self):
        async def run():
            client = _SuccessfulClient()
            result = await create_structured_completion(
                client,
                request={
                    "model": "model-with-thinking-control",
                    "messages": [{"role": "system", "content": "Return JSON."}],
                    "extra_body": {"trace": {"enabled": True}},
                },
                strict_response_format={"type": "json_schema", "json_schema": {"name": "x"}},
                timeout_seconds=1,
                provider_identity="https://llm.example/v1",
                model="model-with-thinking-control",
                disable_thinking=True,
            )
            return result, client.chat.completions.calls

        result, calls = asyncio.run(run())
        self.assertTrue(result.thinking_disabled)
        self.assertEqual(
            calls[0]["extra_body"],
            {
                "trace": {"enabled": True},
                "thinking": {"type": "disabled"},
            },
        )

    def test_undeclared_capability_does_not_receive_thinking_control(self):
        async def run():
            client = _SuccessfulClient()
            result = await create_structured_completion(
                client,
                request={
                    "model": "gpt-4.1-mini",
                    "messages": [{"role": "system", "content": "Return JSON."}],
                },
                strict_response_format={"type": "json_schema", "json_schema": {"name": "x"}},
                timeout_seconds=1,
                provider_identity="https://llm.example/v1",
                model="gpt-4.1-mini",
            )
            return result, client.chat.completions.calls

        result, calls = asyncio.run(run())
        self.assertFalse(result.thinking_disabled)
        self.assertNotIn("extra_body", calls[0])

    def test_thinking_rejection_retries_same_mode_and_caches_capability(self):
        async def run():
            client = _ThinkingUnsupportedClient()
            request = {
                "model": "model-with-thinking-control",
                "messages": [{"role": "system", "content": "Return JSON."}],
            }
            first = await create_structured_completion(
                client,
                request=request,
                strict_response_format={"type": "json_schema", "json_schema": {"name": "x"}},
                timeout_seconds=1,
                provider_identity="https://llm.example/v1",
                model="model-with-thinking-control",
                disable_thinking=True,
            )
            second = await create_structured_completion(
                client,
                request=request,
                strict_response_format={"type": "json_schema", "json_schema": {"name": "x"}},
                timeout_seconds=1,
                provider_identity="https://llm.example/v1",
                model="model-with-thinking-control",
                disable_thinking=True,
            )
            return first, second, client.chat.completions.calls

        first, second, calls = asyncio.run(run())
        self.assertEqual(len(calls), 3)
        self.assertEqual(
            calls[0]["extra_body"],
            {"thinking": {"type": "disabled"}},
        )
        self.assertNotIn("extra_body", calls[1])
        self.assertNotIn("extra_body", calls[2])
        self.assertEqual(first.mode, "json_schema")
        self.assertEqual(first.attempted_modes, ("json_schema",))
        self.assertFalse(first.thinking_disabled)
        self.assertEqual(second.mode, "json_schema")
        self.assertFalse(second.thinking_disabled)

    def test_stream_completion_uses_only_declared_thinking_capability(self):
        async def run():
            declared_client = _SuccessfulClient()
            other_client = _SuccessfulClient()
            declared_stream, declared_disabled = await create_stream_completion(
                declared_client,
                request={
                    "model": "model-with-thinking-control",
                    "messages": [{"role": "user", "content": "你好"}],
                    "stream": True,
                    "extra_body": {"trace": {"enabled": True}},
                },
                provider_identity="https://llm.example/v1",
                model="model-with-thinking-control",
                disable_thinking=True,
            )
            other_stream, other_disabled = await create_stream_completion(
                other_client,
                request={
                    "model": "gpt-5.6-luna",
                    "messages": [{"role": "user", "content": "你好"}],
                    "stream": True,
                },
                provider_identity="https://llm.example/v1",
                model="gpt-5.6-luna",
            )
            return (
                declared_stream,
                declared_disabled,
                declared_client.chat.completions.calls,
                other_stream,
                other_disabled,
                other_client.chat.completions.calls,
            )

        declared_stream, declared_disabled, declared_calls, other_stream, other_disabled, other_calls = asyncio.run(run())
        self.assertTrue(declared_stream.ok)
        self.assertTrue(declared_disabled)
        self.assertEqual(
            declared_calls[0]["extra_body"],
            {
                "trace": {"enabled": True},
                "thinking": {"type": "disabled"},
            },
        )
        self.assertTrue(other_stream.ok)
        self.assertFalse(other_disabled)
        self.assertNotIn("extra_body", other_calls[0])

    def test_stream_thinking_rejection_retries_once_and_caches_capability(self):
        async def run():
            client = _ThinkingUnsupportedClient()
            request = {
                "model": "model-with-thinking-control",
                "messages": [{"role": "user", "content": "你好"}],
                "stream": True,
            }
            first_stream, first_disabled = await create_stream_completion(
                client,
                request=request,
                provider_identity="https://llm.example/v1",
                model="model-with-thinking-control",
                disable_thinking=True,
            )
            second_stream, second_disabled = await create_stream_completion(
                client,
                request=request,
                provider_identity="https://llm.example/v1",
                model="model-with-thinking-control",
                disable_thinking=True,
            )
            return (
                first_stream,
                first_disabled,
                second_stream,
                second_disabled,
                client.chat.completions.calls,
            )

        first_stream, first_disabled, second_stream, second_disabled, calls = asyncio.run(run())
        self.assertTrue(first_stream.ok)
        self.assertTrue(second_stream.ok)
        self.assertFalse(first_disabled)
        self.assertFalse(second_disabled)
        self.assertEqual(len(calls), 3)
        self.assertEqual(
            calls[0]["extra_body"],
            {"thinking": {"type": "disabled"}},
        )
        self.assertNotIn("extra_body", calls[1])
        self.assertNotIn("extra_body", calls[2])


if __name__ == "__main__":
    unittest.main()
