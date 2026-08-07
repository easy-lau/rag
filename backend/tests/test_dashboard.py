import unittest
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from api.dashboard import _REPORT_SYSTEM_PROMPT, dashboard_ai_report, dashboard_overview
from core.permissions import (
    DASHBOARD_READ,
    ROLE_TEMPLATES,
    derive_menus,
    normalize_assignable_capabilities,
)


class _Result:
    def __init__(self, *, scalar=None, rows=(), row=None):
        self.scalar = scalar
        self.rows = list(rows)
        self.row = row

    def scalar_one(self):
        if self.scalar is None:
            raise AssertionError("result has no scalar")
        return self.scalar

    def all(self):
        return list(self.rows)

    def first(self):
        return self.row


class _DashboardDB:
    def __init__(self, results):
        self.results = list(results)
        self.statements = []

    async def execute(self, statement):
        self.statements.append(statement)
        if not self.results:
            raise AssertionError("unexpected dashboard query")
        return self.results.pop(0)


class DashboardPermissionTests(unittest.TestCase):
    def test_dashboard_read_is_assignable_and_derives_menu(self):
        self.assertIn(DASHBOARD_READ, normalize_assignable_capabilities([DASHBOARD_READ]))
        self.assertIn("menu:dashboard", derive_menus({DASHBOARD_READ}))
        self.assertNotIn("menu:dashboard", derive_menus({}))

    def test_auditor_template_includes_dashboard_read(self):
        auditor = next(item for item in ROLE_TEMPLATES if item["code"] == "auditor")
        self.assertIn(DASHBOARD_READ, auditor["capabilities"])


class DashboardOverviewTests(unittest.IsolatedAsyncioTestCase):
    def _run_results(self):
        user_id = uuid.uuid4()
        day = datetime(2026, 8, 6, tzinfo=timezone.utc)
        return [
            _Result(scalar=3),              # 用户总数
            _Result(scalar=2),              # 活跃用户
            _Result(scalar=3),              # 知识库数
            _Result(scalar=14),             # 文档总数
            _Result(scalar=49),             # 分块总数
            _Result(scalar=5),              # 新增文档
            _Result(rows=[("ready", 10), ("draft", 3), ("inactive", 1)]),
            _Result(rows=[(day, 12, 1500.0), (day + timedelta(days=1), 8, 1000.0)]),
            _Result(rows=[(user_id, 8, 6, 1250.0, day + timedelta(hours=3))]),
            _Result(rows=[(user_id, "张三")]),
            _Result(rows=[
                ("hit", 5),
                ("partial", 2),
                ("needs_clarification", 1),
                ("no_hit", 1),
                ("insufficient_evidence", 1),
            ]),
            _Result(scalar=0),              # 失败次数
            _Result(row=(1200.0, 3000.0)),  # avg / p95
            _Result(row=(2.5, 1.0)),        # avg hit / avg kb
            _Result(rows=[
                (day, "generation.completed", 700, 300, 1000, 1),
                (day, "intent.model_result", 200, 50, 250, 1),
                (day + timedelta(days=1), "generation.completed", 500, 200, 700, 1),
            ]),                             # Token 日趋势与阶段构成
            _Result(row=(20, 2, 2, 1)),     # 登录成功/失败、登录账号、失败来源
            _Result(scalar=15),             # 操作总数
            _Result(rows=[("doc.update", 8), ("doc.create", 5)]),
        ]

    async def test_overview_aggregates_all_dimensions(self):
        db = _DashboardDB(self._run_results())
        overview = await dashboard_overview(days=7, db=db, _=None)

        self.assertEqual(overview["days"], 7)
        self.assertEqual(overview["scale"]["users"], 3)
        self.assertEqual(overview["scale"]["active_users"], 2)
        self.assertEqual(overview["scale"]["knowledge_bases"], 3)
        self.assertEqual(overview["scale"]["documents"], 14)
        self.assertEqual(overview["scale"]["chunks"], 49)
        self.assertEqual(overview["scale"]["new_documents"], 5)
        self.assertEqual(overview["scale"]["documents_by_status"], {"ready": 10, "draft": 3, "inactive": 1})

        self.assertEqual(overview["qa"]["total"], 10)
        self.assertEqual(len(overview["qa"]["daily"]), 2)
        self.assertEqual(overview["qa"]["daily"][0]["date"], "2026-08-06")
        self.assertEqual(overview["qa"]["daily"][0]["avg_duration_ms"], 1500)
        self.assertEqual(overview["qa"]["per_user"][0]["username"], "张三")
        self.assertEqual(overview["qa"]["per_user"][0]["count"], 8)
        self.assertEqual(overview["qa"]["per_user"][0]["hit_rate"], 0.75)
        self.assertEqual(overview["qa"]["per_user"][0]["avg_duration_ms"], 1250)
        self.assertEqual(
            overview["qa"]["per_user"][0]["last_active_at"],
            datetime(2026, 8, 6, 3, tzinfo=timezone.utc).isoformat(),
        )

        self.assertEqual(overview["quality"]["by_evidence"]["hit"], 5)
        self.assertEqual(overview["quality"]["hit_rate"], 0.7)
        self.assertEqual(overview["quality"]["clarify_rate"], 0.1)
        self.assertEqual(overview["quality"]["no_answer_rate"], 0.2)
        self.assertEqual(overview["quality"]["error_count"], 0)

        self.assertEqual(overview["performance"]["avg_duration_ms"], 1200)
        self.assertEqual(overview["performance"]["p95_duration_ms"], 3000)
        self.assertEqual(overview["performance"]["avg_hit_count"], 2.5)
        self.assertEqual(overview["performance"]["avg_selected_kb_count"], 1.0)

        self.assertEqual(overview["tokens"]["prompt_tokens"], 1400)
        self.assertEqual(overview["tokens"]["completion_tokens"], 550)
        self.assertEqual(overview["tokens"]["total_tokens"], 1950)
        self.assertEqual(overview["tokens"]["measured_calls"], 3)
        self.assertEqual(overview["tokens"]["avg_tokens_per_qa"], 195)
        self.assertEqual(len(overview["tokens"]["daily"]), 2)
        self.assertEqual(overview["tokens"]["daily"][0]["total_tokens"], 1250)
        self.assertEqual(overview["tokens"]["by_stage"][0]["stage"], "generation.completed")
        self.assertEqual(overview["tokens"]["by_stage"][0]["total_tokens"], 1700)

        self.assertEqual(overview["security"], {
            "login_success": 20,
            "login_failed": 2,
            "login_users": 2,
            "failed_sources": 1,
        })
        self.assertTrue(any(
            "login_logs.attempt_count" in str(statement)
            for statement in db.statements
        ))
        self.assertEqual(overview["operations"]["total"], 15)
        self.assertEqual(overview["operations"]["top_actions"][0], {"action": "doc.update", "count": 8})

        cutoff_params = []
        for statement in db.statements:
            for value in dict(statement.compile().params).values():
                if isinstance(value, datetime):
                    cutoff_params.append(value)
        self.assertTrue(cutoff_params)
        expected_cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        for value in cutoff_params:
            self.assertAlmostEqual(value.timestamp(), expected_cutoff.timestamp(), delta=60)

    async def test_overview_empty_window_returns_zero_rates(self):
        db = _DashboardDB([
            _Result(scalar=1),              # 用户总数
            _Result(scalar=0),              # 活跃用户
            _Result(scalar=1),              # 知识库数
            _Result(scalar=0),              # 文档总数
            _Result(scalar=0),              # 分块总数
            _Result(scalar=0),              # 新增文档
            _Result(rows=[]),               # 文档状态
            _Result(rows=[]),               # 问答日趋势
            _Result(rows=[]),               # 用户问答排名
            _Result(rows=[]),               # 证据状态
            _Result(scalar=0),              # 失败次数
            _Result(row=(None, None)),       # avg / p95
            _Result(row=(None, None)),       # avg hit / avg kb
            _Result(rows=[]),               # Token
            _Result(row=(0, 0, 0, 0)),      # 登录安全
            _Result(scalar=0),              # 操作总数
            _Result(rows=[]),               # 管理操作 Top
        ])
        overview = await dashboard_overview(days=30, db=db, _=None)
        self.assertEqual(overview["qa"]["total"], 0)
        self.assertEqual(overview["quality"]["hit_rate"], 0.0)
        self.assertEqual(overview["quality"]["clarify_rate"], 0.0)
        self.assertEqual(overview["quality"]["no_answer_rate"], 0.0)
        self.assertIsNone(overview["performance"]["avg_duration_ms"])
        self.assertIsNone(overview["performance"]["p95_duration_ms"])
        self.assertIsNone(overview["performance"]["avg_hit_count"])
        self.assertEqual(overview["tokens"]["total_tokens"], 0)
        self.assertEqual(overview["tokens"]["daily"], [])
        self.assertEqual(overview["tokens"]["by_stage"], [])
        self.assertEqual(overview["security"], {
            "login_success": 0,
            "login_failed": 0,
            "login_users": 0,
            "failed_sources": 0,
        })
        self.assertEqual(overview["operations"]["total"], 0)


class DashboardAiReportTests(unittest.IsolatedAsyncioTestCase):
    async def test_report_returns_markdown_from_llm(self):
        from api import dashboard as module

        db = _DashboardDB(DashboardOverviewTests()._run_results())
        settings = module.get_settings()
        original_client = module.get_client
        original_settings = settings.llm_api_key, settings.llm_base_url, settings.chat_model
        settings.llm_api_key = "k"
        settings.llm_base_url = "https://example.test/v1"
        settings.chat_model = "mock-model"
        try:
            with patch.object(module, "get_client") as client_factory:
                client = AsyncMock()
                choice = AsyncMock()
                choice.message.content = "## 总体概况\n系统运行正常。"
                response = AsyncMock()
                response.choices = [choice]
                client.chat.completions.create = AsyncMock(return_value=response)
                client_factory.return_value = client

                result = await dashboard_ai_report(days=7, db=db, _=None)
        finally:
            settings.llm_api_key, settings.llm_base_url, settings.chat_model = original_settings

        self.assertEqual(result["days"], 7)
        self.assertIn("系统运行正常", result["report"])
        call_kwargs = client.chat.completions.create.await_args.kwargs
        self.assertEqual(call_kwargs["model"], "mock-model")
        self.assertEqual(call_kwargs["messages"][0]["role"], "system")
        self.assertEqual(call_kwargs["messages"][0]["content"], _REPORT_SYSTEM_PROMPT)
        self.assertIn("聚合数据", call_kwargs["messages"][1]["content"])

    async def test_report_fails_without_llm_credentials(self):
        from api import dashboard as module

        settings = module.get_settings()
        original = settings.llm_api_key, settings.llm_base_url, settings.chat_model
        settings.llm_api_key = ""
        settings.llm_base_url = ""
        settings.chat_model = ""
        try:
            with self.assertRaises(Exception) as ctx:
                await dashboard_ai_report(days=7, db=_DashboardDB([]), _=None)
        finally:
            settings.llm_api_key, settings.llm_base_url, settings.chat_model = original
        self.assertIn("503", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
