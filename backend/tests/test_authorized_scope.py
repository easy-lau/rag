import unittest
import uuid

from core.authorized_scope import (
    auto_select_single_authorized_scope,
    resolve_authorized_scope_clarification,
)
from core.query_constraints import ApplicabilityScope
from core.rag_v2.contracts import AnswerRequirementV2, QueryPlanV2


class _Rows:
    def __init__(self, rows):
        self._rows = list(rows)

    def all(self):
        return list(self._rows)


class _CatalogDB:
    def __init__(self, rows):
        self.rows = list(rows)
        self.execute_count = 0

    async def execute(self, _statement):
        self.execute_count += 1
        return _Rows(self.rows)


def _row(*, kb_id, product, version):
    doc_id = uuid.uuid4()
    return (
        doc_id,
        kb_id,
        f"{product}-{version}-配置.md",
        [],
        f"所属产品：{product}\n产品版本：{version}",
        {"product": product, "version": version},
    )


def _plan(*, product="平台A", version=None, answer_shape="process"):
    scope = ApplicabilityScope(
        product=product,
        version=version,
        explicit_version=version is not None,
        extraction_reason="authorized-scope test",
    )
    requirement = AnswerRequirementV2(
        id="a1",
        description="登录参数修改方式",
        coverage_mode="collection",
        coverage_contract="structured_collection",
        depends_on_requirement_ids=(),
        augmentation_requirement_ids=(),
        applicability_scope=scope,
    )
    return QueryPlanV2(
        original_query="平台A的登录参数应该怎么修改",
        answer_shape=answer_shape,
        retrieval_queries=("平台A 登录参数",),
        requirements=(requirement,),
        confidence=0.9,
        source="model",
        reason="authorized-scope test",
    )


class AutoSelectSingleAuthorizedScopeTests(unittest.TestCase):
    def test_single_document_across_versions_is_auto_selected(self) -> None:
        doc_id = str(uuid.uuid4())
        choices = {
            ("平台a", "6"): {
                "product": "平台A",
                "version": "6",
                "kb_ids": {"kb-1"},
                "doc_ids": {doc_id},
            },
            ("平台a", "7"): {
                "product": "平台A",
                "version": "7",
                "kb_ids": {"kb-1"},
                "doc_ids": {doc_id},
            },
        }

        self.assertTrue(auto_select_single_authorized_scope(choices))

    def test_multiple_distinct_documents_remain_ambiguous(self) -> None:
        choices = {
            ("平台a", "6"): {
                "product": "平台A",
                "version": "6",
                "kb_ids": {"kb-1"},
                "doc_ids": {str(uuid.uuid4())},
            },
            ("平台a", "7"): {
                "product": "平台A",
                "version": "7",
                "kb_ids": {"kb-1"},
                "doc_ids": {str(uuid.uuid4())},
            },
        }

        self.assertFalse(auto_select_single_authorized_scope(choices))

    def test_single_version_single_document_is_auto_selected(self) -> None:
        choices = {
            ("平台a", "6"): {
                "product": "平台A",
                "version": "6",
                "kb_ids": {"kb-1"},
                "doc_ids": {str(uuid.uuid4())},
            },
        }

        self.assertTrue(auto_select_single_authorized_scope(choices))


class AuthorizedScopeClarificationTests(unittest.IsolatedAsyncioTestCase):
    async def test_multiple_authorized_versions_require_clarification(self):
        kb_id = uuid.uuid4()
        db = _CatalogDB([
            _row(kb_id=kb_id, product="平台A", version="6"),
            _row(kb_id=kb_id, product="平台A", version="7"),
        ])

        result = await resolve_authorized_scope_clarification(
            db,
            plan=_plan(),
            query="平台A的登录参数应该怎么修改",
            kb_ids=[kb_id],
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.dimension, "product_version")
        self.assertEqual([choice.version for choice in result.choices], ["6", "7"])
        contract = result.to_contract()
        self.assertEqual(contract.adapter, "semantic")
        self.assertEqual(contract.selection_mode, "choice")
        self.assertNotIn(str(kb_id), str(contract.to_dict(public=True)))

    async def test_single_version_and_unselected_rows_do_not_clarify(self):
        selected_kb_id = uuid.uuid4()
        unselected_kb_id = uuid.uuid4()
        db = _CatalogDB([
            _row(kb_id=selected_kb_id, product="平台A", version="6"),
            _row(kb_id=unselected_kb_id, product="平台A", version="7"),
        ])

        result = await resolve_authorized_scope_clarification(
            db,
            plan=_plan(),
            query="平台A的登录参数应该怎么修改",
            kb_ids=[selected_kb_id],
        )

        self.assertIsNone(result)

    async def test_multiple_versions_on_one_document_do_not_clarify(self) -> None:
        """All version choices resolving to the same authorized doc auto-select.

        The server owns the only candidate: asking the user to pick a version
        would offer no real document choice, so the version picker is skipped.
        """

        kb_id = uuid.uuid4()
        shared_doc_id = uuid.uuid4()
        rows = [
            (
                shared_doc_id,
                kb_id,
                f"平台A-6-配置.md",
                [],
                f"所属产品：平台A\n产品版本：6",
                {"product": "平台A", "version": "6"},
            ),
            (
                shared_doc_id,
                kb_id,
                f"平台A-7-配置.md",
                [],
                f"所属产品：平台A\n产品版本：7",
                {"product": "平台A", "version": "7"},
            ),
        ]
        db = _CatalogDB(rows)

        result = await resolve_authorized_scope_clarification(
            db,
            plan=_plan(),
            query="平台A的登录参数应该怎么修改",
            kb_ids=[kb_id],
        )

        self.assertIsNone(result)

    async def test_catalog_metadata_extends_product_recognition_without_alias_code(self):
        kb_id = uuid.uuid4()
        db = _CatalogDB([
            _row(kb_id=kb_id, product="平台A", version="6"),
            _row(kb_id=kb_id, product="平台A", version="7"),
        ])

        result = await resolve_authorized_scope_clarification(
            db,
            plan=_plan(product=None),
            query="平台A的登录参数应该怎么修改",
            kb_ids=[kb_id],
        )

        self.assertIsNotNone(result)
        self.assertEqual(
            [choice.label for choice in result.choices],
            ["平台A 版本 6", "平台A 版本 7"],
        )

    async def test_explicit_all_versions_and_overview_bypass_picker(self):
        kb_id = uuid.uuid4()
        rows = [
            _row(kb_id=kb_id, product="平台A", version="6"),
            _row(kb_id=kb_id, product="平台A", version="7"),
        ]
        cases = (
            (_plan(version="7"), "平台A 7 的登录参数怎么修改"),
            (_plan(), "对比平台A的全部版本"),
            (_plan(answer_shape="overview"), "介绍一下平台A"),
        )
        for plan, query in cases:
            with self.subTest(query=query):
                db = _CatalogDB(rows)
                result = await resolve_authorized_scope_clarification(
                    db,
                    plan=plan,
                    query=query,
                    kb_ids=[kb_id],
                )
                self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
