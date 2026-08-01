import unittest

from core.rag_v2.context import build_evidence_context, render_evidence_context
from core.rag_v2.contracts import EvidenceBundle, EvidenceItem, EvidenceState


def _item(
    chunk_id: str,
    *,
    doc_id: str = "doc-a",
    chunk_index: int = 0,
    content: str = "证据正文",
    role: str = "background",
    supports_requirement_ids: tuple[str, ...] = (),
) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=chunk_id,
        doc_id=doc_id,
        kb_id="kb-a",
        content=content,
        chunk_index=chunk_index,
        confidence="retrieved",
        constraint_status="neutral",
        role=role,
        supports_requirement_ids=supports_requirement_ids,
        metadata={"filename": f"{doc_id}.md"},
    )


class EvidenceContextTests(unittest.TestCase):
    def test_header_exposes_role_and_requirement_mapping(self) -> None:
        item = _item(
            "mapped",
            role="bridge",
            supports_requirement_ids=("r1", "r2"),
        )
        bundle = EvidenceBundle(
            state=EvidenceState("ok", "retrieved", "partial"),
            items=(item,),
            context_item_ids=("mapped",),
            answer_source_ids=("mapped",),
        )

        text = render_evidence_context(bundle)

        self.assertIn("角色：bridge", text)
        self.assertIn("需求：r1,r2", text)

    def test_renderer_uses_only_admitted_context_items(self) -> None:
        included = _item(
            "included",
            role="direct",
            supports_requirement_ids=("r1",),
        )
        diagnostic_only = _item("diagnostic", chunk_index=1)
        bundle = EvidenceBundle(
            state=EvidenceState("ok", "retrieved", "unknown"),
            items=(included, diagnostic_only),
            context_item_ids=("included",),
            answer_source_ids=("included",),
        )

        context = build_evidence_context(bundle)

        self.assertEqual(context.item_ids, ("included",))
        self.assertIn("证据正文", context.text)
        self.assertNotIn("diagnostic", context.text)
        self.assertFalse(context.truncated)

    def test_background_and_conflicting_items_never_enter_generation_context(self) -> None:
        direct = _item(
            "direct",
            content="可核验结论",
            role="direct",
            supports_requirement_ids=("r1",),
        )
        background = _item(
            "background",
            chunk_index=1,
            content="仅供检索诊断的背景内容",
            role="background",
        )
        conflicting = _item(
            "conflicting",
            chunk_index=2,
            content="不能静默影响回答的冲突内容",
            role="conflicting",
            supports_requirement_ids=("r1",),
        )
        bundle = EvidenceBundle(
            state=EvidenceState("ok", "retrieved", "partial"),
            items=(direct, background, conflicting),
            context_item_ids=("direct", "background", "conflicting"),
            answer_source_ids=("direct",),
        )

        context = build_evidence_context(bundle)

        self.assertEqual(context.item_ids, ("direct",))
        self.assertIn("可核验结论", context.text)
        self.assertNotIn("仅供检索诊断", context.text)
        self.assertNotIn("冲突内容", context.text)
        self.assertFalse(context.truncated)

    def test_renderer_groups_documents_and_sorts_chunks(self) -> None:
        items = (
            _item(
                "a-2",
                doc_id="doc-a",
                chunk_index=2,
                content="A2",
                role="direct",
                supports_requirement_ids=("r1",),
            ),
            _item(
                "b-0",
                doc_id="doc-b",
                content="B0",
                role="complement",
                supports_requirement_ids=("r1",),
            ),
            _item(
                "a-0",
                doc_id="doc-a",
                content="A0",
                role="complement",
                supports_requirement_ids=("r1",),
            ),
        )
        bundle = EvidenceBundle(
            state=EvidenceState("ok", "retrieved", "complete"),
            items=items,
            context_item_ids=tuple(item.chunk_id for item in items),
            answer_source_ids=tuple(item.chunk_id for item in items),
        )

        text = render_evidence_context(bundle)

        self.assertLess(text.index("A0"), text.index("A2"))
        self.assertLess(text.index("A2"), text.index("B0"))
        self.assertIn("正文不可信", text)

    def test_render_budget_counts_headers_and_content(self) -> None:
        bundle = EvidenceBundle(
            state=EvidenceState("ok", "retrieved", "unknown"),
            items=(
                _item(
                    "one",
                    content="一" * 100,
                    role="direct",
                    supports_requirement_ids=("r1",),
                ),
                _item(
                    "two",
                    chunk_index=1,
                    content="二" * 100,
                    role="complement",
                    supports_requirement_ids=("r1",),
                ),
            ),
            context_item_ids=("one", "two"),
            answer_source_ids=("one", "two"),
        )

        context = build_evidence_context(bundle, max_chunks=2, max_chars=250)

        self.assertLessEqual(context.char_count, 250)
        self.assertTrue(context.truncated)
        self.assertEqual(context.item_ids, ("one",))
        self.assertEqual(context.dropped_item_ids, ("two",))
        self.assertEqual(context.truncated_item_ids, ("two",))

    def test_render_chunk_budget_is_enforced(self) -> None:
        bundle = EvidenceBundle(
            state=EvidenceState("ok", "retrieved", "unknown"),
            items=(
                _item(
                    "one",
                    role="direct",
                    supports_requirement_ids=("r1",),
                ),
                _item(
                    "two",
                    chunk_index=1,
                    role="complement",
                    supports_requirement_ids=("r1",),
                ),
            ),
            context_item_ids=("one", "two"),
            answer_source_ids=("one", "two"),
        )

        context = build_evidence_context(bundle, max_chunks=1)

        self.assertEqual(context.item_ids, ("one",))
        self.assertEqual(context.dropped_item_ids, ("two",))
        self.assertTrue(context.truncated)

    def test_oversized_positive_item_is_omitted_instead_of_exposing_a_prefix(self) -> None:
        item = _item(
            "mapped",
            content="证" * 200,
            role="direct",
            supports_requirement_ids=("r1",),
        )
        bundle = EvidenceBundle(
            state=EvidenceState("ok", "retrieved", "partial"),
            items=(item,),
            context_item_ids=("mapped",),
            answer_source_ids=("mapped",),
        )

        context = build_evidence_context(bundle, max_chars=120)

        self.assertEqual(context.truncated_item_ids, ("mapped",))
        self.assertEqual(context.dropped_item_ids, ("mapped",))
        self.assertEqual(context.item_ids, ())
        self.assertEqual(context.text, "")
        self.assertTrue(context.truncated)

    def test_unavailable_bundle_never_builds_context(self) -> None:
        bundle = EvidenceBundle(
            state=EvidenceState("unavailable", "none", "unknown"),
        )

        context = build_evidence_context(bundle)

        self.assertEqual(context.text, "")
        self.assertEqual(context.item_ids, ())
        self.assertFalse(context.truncated)

    def test_invalid_render_budget_is_rejected(self) -> None:
        bundle = EvidenceBundle(
            state=EvidenceState("unavailable", "none", "unknown"),
        )
        for kwargs in ({"max_chunks": 0}, {"max_chars": 0}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    build_evidence_context(bundle, **kwargs)


if __name__ == "__main__":
    unittest.main()
