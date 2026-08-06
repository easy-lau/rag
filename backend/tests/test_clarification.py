import uuid
import unittest

from core.clarification import (
    CLARIFICATION_EVENT_SCHEMA,
    CLARIFICATION_STATE_SCHEMA,
    ClarificationContract,
    build_clarification_state,
    public_clarification_event,
    resolve_clarification_reply,
    validate_clarification_state,
)


class ClarificationContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kb_id = uuid.uuid4()
        self.doc_1 = uuid.uuid4()
        self.doc_2 = uuid.uuid4()
        self.contract = ClarificationContract(
            adapter="evidence",
            dimension="product_version",
            reason_code="multiple_authorized_versions",
            selection_mode="choice",
            choices=(
                {
                    "key": "scope1",
                    "label": "云枢 版本 6",
                    "value": "6",
                    "products": ["云枢"],
                    "versions": ["6"],
                    "kb_ids": [str(self.kb_id)],
                    "doc_ids": [str(self.doc_1)],
                },
                {
                    "key": "scope2",
                    "label": "云枢 版本 6.0.1",
                    "value": "6.0.1",
                    "products": ["云枢"],
                    "versions": ["6.0.1"],
                    "kb_ids": [str(self.kb_id)],
                    "doc_ids": [str(self.doc_2)],
                },
            ),
        )
        self.state = build_clarification_state(
            contract=self.contract,
            original_query="我想修改云枢的默认密码",
            selected_kb_ids=[self.kb_id],
            base_user_message_id=uuid.uuid4(),
            clarification_message_id=uuid.uuid4(),
        )

    def test_one_state_schema_validates_both_adapters(self) -> None:
        self.assertEqual(self.state["schema_version"], CLARIFICATION_STATE_SCHEMA)
        self.assertEqual(validate_clarification_state(self.state), self.state)

        semantic = ClarificationContract(
            adapter="semantic",
            dimension="reference",
            reason_code="unresolved_reference",
            selection_mode="refine",
        )
        semantic_state = build_clarification_state(
            contract=semantic,
            original_query="这个怎么修改",
            selected_kb_ids=[self.kb_id],
            base_user_message_id=uuid.uuid4(),
            clarification_message_id=uuid.uuid4(),
        )
        self.assertEqual(
            semantic_state["schema_version"],
            self.state["schema_version"],
        )

    def test_all_supported_ordinal_forms_resolve_through_one_parser(self) -> None:
        for reply in ("1", "第一个", "第一项", "选1", "选择第一个吧", "scope1"):
            with self.subTest(reply=reply):
                resolution = resolve_clarification_reply(reply, self.state)
                self.assertEqual(resolution.action, "single")
                self.assertEqual(resolution.choices[0]["key"], "scope1")

    def test_label_and_version_value_resolve_same_choice(self) -> None:
        for reply in ("云枢 版本 6.0.1", "6.0.1", "我要云枢6.0.1版本"):
            with self.subTest(reply=reply):
                resolution = resolve_clarification_reply(reply, self.state)
                self.assertEqual(resolution.action, "single")
                self.assertEqual(resolution.choices[0]["key"], "scope2")

    def test_all_cancel_new_question_and_invalid_choice_are_distinct(self) -> None:
        self.assertEqual(
            resolve_clarification_reply("全部版本", self.state).action,
            "all",
        )
        self.assertEqual(
            resolve_clarification_reply("取消", self.state).action,
            "cancel",
        )
        self.assertEqual(
            resolve_clarification_reply("为什么密码会过期？", self.state).action,
            "new_question",
        )
        self.assertEqual(
            resolve_clarification_reply("第九个", self.state).action,
            "repeat",
        )

    def test_active_event_exposes_labels_but_not_resource_ids(self) -> None:
        event = public_clarification_event(
            self.state,
            route_state_revision=4,
            conversation_id=uuid.uuid4(),
            persisted=True,
        )
        self.assertEqual(event["type"], "clarification_state")
        self.assertEqual(event["schema_version"], CLARIFICATION_EVENT_SCHEMA)
        self.assertEqual(event["status"], "active")
        self.assertTrue(event["persisted"])
        self.assertEqual(event["choices"][0]["label"], "云枢 版本 6")
        self.assertNotIn("kb_ids", event["choices"][0])
        self.assertNotIn("doc_ids", event["choices"][0])


if __name__ == "__main__":
    unittest.main()
