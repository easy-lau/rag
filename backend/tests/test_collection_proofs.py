import unittest

from core.rag_v2.collection_proofs import (
    derive_source_collection_closure_proofs,
    has_explicit_collection_closure,
)
from core.rag_v2.contracts import AnswerRequirementV2, EvidenceItem


def _requirement(
    question: str,
    *,
    coverage_contract: str = "structured_collection",
) -> AnswerRequirementV2:
    return AnswerRequirementV2(
        id="r1",
        description=question,
        coverage_mode="collection",
        coverage_contract=coverage_contract,  # type: ignore[arg-type]
    )


def _item(content: str) -> EvidenceItem:
    return EvidenceItem(
        chunk_id="source",
        kb_id="kb",
        doc_id="doc",
        content=content,
        role="direct",
        contribution_kind="answer_claim",
        supports_requirement_ids=("r1",),
    )


class CollectionSourceProofTests(unittest.TestCase):
    def _proves(
        self,
        question: str,
        content: str,
        *,
        coverage_contract: str = "structured_collection",
    ) -> bool:
        requirement = _requirement(
            question,
            coverage_contract=coverage_contract,
        )
        return has_explicit_collection_closure(
            _item(content),
            requirement=requirement,
            requirements=(requirement,),
        )

    def _proves_ordered_steps(self, question: str, content: str) -> bool:
        return self._proves(
            question,
            content,
            coverage_contract="ordered_steps",
        )

    def test_accepts_target_bound_local_process_block(self):
        self.assertTrue(self._proves_ordered_steps(
            "采购申请流程是什么",
            "采购申请流程如下：\n1. 提交申请。\n2. 负责人审批。\n3. 系统归档。",
        ))

    def test_accepts_exclusive_singleton_with_an_actual_member(self):
        self.assertTrue(self._proves(
            "系统支持的登录方式有哪些",
            "系统支持的登录方式仅包括密码登录。",
        ))

    def test_accepts_an_all_inclusive_marker_without_self_rejecting_it(self):
        self.assertTrue(self._proves(
            "系统支持的登录方式有哪些",
            "系统支持的登录方式全部包括密码登录、单点登录。",
        ))

    def test_does_not_treat_a_business_action_as_an_external_reference(self):
        self.assertTrue(self._proves_ordered_steps(
            "采购申请流程是什么",
            "采购申请流程仅包括提交申请、审核申请、查看审批状态。",
        ))

    def test_rejects_local_facet_and_external_reference(self):
        self.assertFalse(self._proves(
            "公司出差标准是什么",
            "公司出差标准：交通如下：飞机、高铁。住宿标准另见表。",
        ))

    def test_rejects_anchor_followed_by_attachment_list(self):
        self.assertFalse(self._proves_ordered_steps(
            "采购申请流程是什么",
            "采购申请流程如下：具体请查看附件。\n1. 附件清单\n2. 联系人",
        ))

    def test_rejects_a_list_after_an_intervening_same_line_statement(self):
        """Only the immediately following block can belong to the anchor."""

        self.assertFalse(self._proves_ordered_steps(
            "采购申请流程是什么",
            "采购申请流程如下。其他规则另行说明。\n1. 错误列表\n2. 不是流程",
        ))

    def test_rejects_bare_count_or_bare_closure_phrase(self):
        self.assertFalse(self._proves(
            "系统支持的登录方式有哪些",
            "系统支持的登录方式共3项，具体见附录。",
        ))
        self.assertFalse(self._proves(
            "系统支持的登录方式有哪些",
            "系统支持的登录方式全部如下。",
        ))

    def test_rejects_open_world_taxonomy_with_external_rules(self):
        self.assertFalse(self._proves(
            "供应商管理要求是什么",
            "供应商管理要求分为准入、履约、退出三类，具体条款另见制度正文。",
        ))

    def test_rejects_open_ended_members_even_after_an_exclusive_prefix(self):
        self.assertFalse(self._proves(
            "系统支持的登录方式有哪些",
            "系统支持的登录方式仅包括密码登录、单点登录等。",
        ))
        self.assertFalse(self._proves(
            "系统支持的登录方式有哪些",
            "系统支持的登录方式仅包括密码登录及其他方式。",
        ))

    def test_rejects_deferred_procedure_continuation(self):
        """A local transition is not proof that the whole procedure ends there."""

        self.assertFalse(self._proves_ordered_steps(
            "采购申请流程是什么",
            "采购申请流程：提交申请后由负责人审批，后续归档另行处理。",
        ))

    def test_rejects_a_plain_transition_as_a_complete_procedure(self):
        self.assertFalse(self._proves_ordered_steps(
            "采购申请流程是什么",
            "采购申请流程：提交申请后由负责人审批。",
        ))

    def test_accepts_an_explicit_inline_procedure_sequence(self):
        self.assertTrue(self._proves_ordered_steps(
            "采购申请流程是什么",
            "采购申请流程：提交申请、负责人审批、系统归档。",
        ))

    def test_keeps_a_closed_collection_when_a_later_sentence_redirects_elsewhere(self):
        """A later, unrelated sentence is not a qualifier of this declaration."""

        self.assertTrue(self._proves(
            "系统支持的登录方式有哪些",
            "系统支持的登录方式仅包括密码登录、单点登录。其他功能详见管理员手册。",
        ))

    def test_keeps_a_closed_collection_when_a_later_sentence_is_an_example(self):
        """An example of another statement must not reopen the prior list."""

        self.assertTrue(self._proves(
            "系统支持的登录方式有哪些",
            "系统支持的登录方式仅包括密码登录、单点登录。说明：例如首次登录可使用密码。",
        ))

    def test_rejects_a_redirect_attached_to_the_same_collection_declaration(self):
        self.assertFalse(self._proves(
            "系统支持的登录方式有哪些",
            "系统支持的登录方式仅包括密码登录、单点登录，具体规则详见管理员手册。",
        ))

    def test_rejects_a_deferred_continuation_attached_to_the_same_procedure(self):
        self.assertFalse(self._proves_ordered_steps(
            "采购申请流程是什么",
            "采购申请流程：提交申请、负责人审批，后续归档另行处理。",
        ))

    def test_keeps_a_closed_procedure_when_a_later_sentence_redirects_elsewhere(self):
        self.assertTrue(self._proves_ordered_steps(
            "采购申请流程是什么",
            "采购申请流程：提交申请、负责人审批、系统归档。其他功能详见管理员手册。",
        ))

    def test_proof_spans_exclude_a_later_unrelated_sentence(self):
        question = "系统支持的登录方式有哪些"
        content = "系统支持的登录方式仅包括密码登录、单点登录。其他功能详见管理员手册。"
        requirement = _requirement(question)
        proofs = derive_source_collection_closure_proofs(
            _item(content),
            requirement=requirement,
            requirements=(requirement,),
        )

        self.assertEqual(len(proofs), 1)
        proof = proofs[0]
        self.assertTrue(proof.is_closed)
        self.assertEqual(proof.anchor_span.text(content), "系统支持的登录方式仅包括密码登录、单点登录。")
        self.assertEqual(proof.member_block_span.text(content), "密码登录、单点登录")
        self.assertEqual(proof.attached_qualifier_spans, ())

    def test_proof_spans_expose_an_attached_redirect_as_the_rejection_reason(self):
        question = "系统支持的登录方式有哪些"
        content = "系统支持的登录方式仅包括密码登录、单点登录，具体规则详见管理员手册。"
        requirement = _requirement(question)
        proofs = derive_source_collection_closure_proofs(
            _item(content),
            requirement=requirement,
            requirements=(requirement,),
        )

        self.assertEqual(len(proofs), 1)
        self.assertFalse(proofs[0].is_closed)
        self.assertIn(
            "详见",
            "".join(span.text(content) for span in proofs[0].attached_qualifier_spans),
        )

    def test_adjacent_step_block_stays_closed_when_later_prose_redirects_elsewhere(self):
        self.assertTrue(self._proves_ordered_steps(
            "采购申请流程是什么",
            "采购申请流程如下：\n1. 提交申请\n2. 负责人审批\n3. 系统归档\n其他功能详见管理员手册。",
        ))

    def test_configuration_words_do_not_reclassify_the_declared_contract(self):
        """Only the semantic contract chooses member-list versus procedure."""

        question = "VPN配置项有哪些"
        member_source = "VPN配置项仅包括认证模式、服务端口。"
        self.assertTrue(self._proves(question, member_source))
        self.assertFalse(self._proves(
            question,
            member_source,
            coverage_contract="ordered_steps",
        ))

        ordered_source = "VPN配置项设置步骤：先选择认证模式，再填写服务端口。"
        self.assertTrue(self._proves(
            question,
            ordered_source,
            coverage_contract="ordered_steps",
        ))


if __name__ == "__main__":
    unittest.main()
