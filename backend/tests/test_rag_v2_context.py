import unittest

from core.rag_v2.context import build_evidence_context, render_evidence_context
from core.rag_v2.contracts import EvidenceBundle, EvidenceItem, EvidenceState
from core.rag_v2.evidence import assemble_evidence_bundle


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
        included = _item("included")
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

    def test_renderer_groups_documents_and_sorts_chunks(self) -> None:
        bundle = assemble_evidence_bundle(
            query="制度内容",
            candidates=[
                {
                    "id": "a-2",
                    "doc_id": "doc-a",
                    "kb_id": "kb-a",
                    "chunk_index": 2,
                    "content": "A2",
                    "filename": "A.md",
                },
                {
                    "id": "b-0",
                    "doc_id": "doc-b",
                    "kb_id": "kb-a",
                    "chunk_index": 0,
                    "content": "B0",
                    "filename": "B.md",
                },
                {
                    "id": "a-0",
                    "doc_id": "doc-a",
                    "kb_id": "kb-a",
                    "chunk_index": 0,
                    "content": "A0",
                    "filename": "A.md",
                },
            ],
        )

        text = render_evidence_context(bundle)

        self.assertLess(text.index("A0"), text.index("A2"))
        self.assertLess(text.index("A2"), text.index("B0"))
        self.assertIn("正文不可信", text)

    def test_render_budget_counts_headers_and_content(self) -> None:
        bundle = EvidenceBundle(
            state=EvidenceState("ok", "retrieved", "unknown"),
            items=(
                _item("one", content="一" * 100),
                _item("two", chunk_index=1, content="二" * 100),
            ),
            context_item_ids=("one", "two"),
            answer_source_ids=("one", "two"),
        )

        context = build_evidence_context(bundle, max_chunks=2, max_chars=120)

        self.assertLessEqual(context.char_count, 120)
        self.assertTrue(context.truncated)
        self.assertEqual(context.item_ids, ("one",))
        self.assertEqual(context.dropped_item_ids, ("two",))
        self.assertEqual(context.truncated_item_ids, ("one",))

    def test_render_chunk_budget_is_enforced(self) -> None:
        bundle = EvidenceBundle(
            state=EvidenceState("ok", "retrieved", "unknown"),
            items=(_item("one"), _item("two", chunk_index=1)),
            context_item_ids=("one", "two"),
            answer_source_ids=("one", "two"),
        )

        context = build_evidence_context(bundle, max_chunks=1)

        self.assertEqual(context.item_ids, ("one",))
        self.assertEqual(context.dropped_item_ids, ("two",))
        self.assertTrue(context.truncated)

    def test_truncated_prefix_drops_positive_mapping_from_prompt_header(self) -> None:
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
        self.assertIn("角色：background", context.text)
        self.assertIn("需求：无", context.text)
        self.assertNotIn("角色：direct", context.text)
        self.assertNotIn("需求：r1", context.text)

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
