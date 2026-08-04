import asyncio
import unittest
from types import SimpleNamespace

from core.structured_output import (
    clear_structured_output_capability_cache,
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


class StructuredOutputTests(unittest.TestCase):
    def setUp(self):
        clear_structured_output_capability_cache()

    def test_negotiates_to_plain_json_and_reuses_capability(self):
        async def run():
            client = _Client()
            first = await create_structured_completion(
                client,
                request={"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "json"}]},
                strict_response_format={"type": "json_schema", "json_schema": {"name": "x"}},
                timeout_seconds=1,
                provider_identity="https://llm.example/v1",
                model="deepseek-v4-flash",
            )
            second = await create_structured_completion(
                client,
                request={"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "json"}]},
                strict_response_format={"type": "json_schema", "json_schema": {"name": "x"}},
                timeout_seconds=1,
                provider_identity="https://llm.example/v1",
                model="deepseek-v4-flash",
            )
            return first, second, client.chat.completions.calls

        first, second, calls = asyncio.run(run())
        self.assertEqual(first.mode, "plain_json")
        self.assertEqual(first.attempted_modes, ("json_schema", "json_object", "plain_json"))
        self.assertEqual(second.mode, "plain_json")
        self.assertEqual(len(calls), 4)
        self.assertNotIn("response_format", calls[-1])


if __name__ == "__main__":
    unittest.main()
