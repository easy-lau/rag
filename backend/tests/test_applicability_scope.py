import unittest
import uuid

from core.query_constraints import (
    ApplicabilityScope,
    admit_candidates_for_scopes,
    evaluate_candidate_constraints,
    extract_applicability_scope,
    extract_applicability_scopes,
    extract_document_constraint_identity,
)


def _candidate(
    *,
    product="云枢",
    project=None,
    version="8.2.75",
    global_scope=False,
):
    metadata = {
        "product": product,
        "version": version,
    }
    if project is not None:
        metadata["project"] = project
    if global_scope:
        metadata["scope_applicability"] = "global"
    return {
        "kb_id": str(uuid.uuid4()),
        "doc_id": str(uuid.uuid4()),
        "chunk_id": str(uuid.uuid4()),
        "content": "用于测试的候选正文，不能进入拒绝记录。",
        "metadata": metadata,
    }


class ApplicabilityScopeTests(unittest.TestCase):
    def test_named_project_scope_comes_from_exact_current_query_span(self):
        question = "中青建安项目的云枢8.2.75普通员工餐补标准是多少"

        scope = extract_applicability_scope(question)

        self.assertIsNone(scope.product)
        self.assertEqual(scope.version, "8.2.75")
        self.assertEqual(scope.project, "中青建安")
        self.assertIsNotNone(scope.project_source)
        self.assertEqual(scope.project_source.origin, "current_query")
        self.assertEqual(
            question[scope.project_source.start:scope.project_source.end],
            scope.project_source.span,
        )
        self.assertEqual(scope.project_source.span, "中青建安")

    def test_possessive_prefix_is_not_promoted_to_project_scope(self):
        question = "中青建安的云枢8.2.75普通员工餐补标准是多少"

        scope = extract_applicability_scope(question)

        self.assertIsNone(scope.project)
        self.assertIsNone(scope.project_source)

    def test_comparison_returns_independent_source_scopes(self):
        question = "比较 CloudPivot 6 和 CloudPivot 7 的安全配置"

        scopes = extract_applicability_scopes(question)

        self.assertEqual(
            {(scope.product, scope.version) for scope in scopes},
            {("CloudPivot", "6"), ("CloudPivot", "7")},
        )
        self.assertTrue(all(scope.version_source is not None for scope in scopes))

    def test_project_mismatch_is_rejected_without_candidate_content(self):
        scope = extract_applicability_scope(
            "中青建安项目的云枢8.2.75餐补标准是多少"
        )
        wrong_project = _candidate(project="华东示范项目")

        evaluation = evaluate_candidate_constraints(scope, wrong_project)
        admission = admit_candidates_for_scopes([wrong_project], (scope,))

        self.assertEqual(evaluation.status, "mismatch")
        self.assertIn("project", evaluation.mismatch_dimensions)
        self.assertEqual(admission.candidates, ())
        self.assertEqual(len(admission.rejections), 1)
        rejection = admission.rejections[0]
        self.assertIn("project", rejection.mismatch_dimensions)
        self.assertNotIn("content", rejection.to_dict())
        self.assertNotIn("候选正文", repr(rejection))

    def test_explicit_global_clause_is_compatible_but_not_project_exact(self):
        scope = extract_applicability_scope(
            "中青建安项目的云枢8.2.75餐补标准是多少"
        )
        global_candidate = _candidate(global_scope=True)

        evaluation = evaluate_candidate_constraints(scope, global_candidate)
        admission = admit_candidates_for_scopes([global_candidate], (scope,))

        self.assertEqual(evaluation.status, "compatible")
        self.assertEqual(evaluation.scope_applicability, "global_compatible")
        self.assertEqual(len(admission.candidates), 1)
        self.assertEqual(
            admission.candidates[0]["metadata"]["scope_applicability"],
            "global_compatible",
        )

    def test_unknown_project_identity_fails_closed_for_explicit_project_scope(self):
        scope = extract_applicability_scope(
            "中青建安项目的云枢8.2.75餐补标准是多少"
        )
        unknown_project = _candidate(project=None)

        evaluation = evaluate_candidate_constraints(scope, unknown_project)

        self.assertEqual(evaluation.status, "unknown")
        self.assertIn("project", evaluation.mismatch_dimensions)

    def test_project_exact_candidate_is_admitted_as_exact_scope(self):
        scope = extract_applicability_scope(
            "中青建安项目的云枢8.2.75餐补标准是多少"
        )
        candidate = _candidate(project="中青建安")

        admission = admit_candidates_for_scopes([candidate], (scope,))

        self.assertEqual(len(admission.candidates), 1)
        self.assertEqual(admission.rejections, ())
        metadata = admission.candidates[0]["metadata"]
        self.assertEqual(metadata["scope_applicability"], "exact")
        self.assertEqual(metadata["scope_fingerprint"], scope.fingerprint)

    def test_global_clause_cannot_escape_product_or_version_mismatch(self):
        scope = extract_applicability_scope(
            "中青建安项目的云枢8.2.75餐补标准是多少"
        )
        global_wrong_version = _candidate(version="7", global_scope=True)

        evaluation = evaluate_candidate_constraints(scope, global_wrong_version)
        admission = admit_candidates_for_scopes([global_wrong_version], (scope,))

        self.assertEqual(evaluation.status, "mismatch")
        self.assertIn("version", evaluation.mismatch_dimensions)
        self.assertEqual(admission.candidates, ())
        self.assertEqual(len(admission.rejections), 1)
        self.assertIn("version", admission.rejections[0].mismatch_dimensions)

    def test_scope_union_only_rejects_when_candidate_fails_every_scope(self):
        scopes = extract_applicability_scopes(
            "比较 CloudPivot 6 和 CloudPivot 7 的安全配置"
        )
        candidate_6 = _candidate(product="CloudPivot", version="6")
        candidate_8 = _candidate(product="CloudPivot", version="8")

        admission = admit_candidates_for_scopes([candidate_6, candidate_8], scopes)

        self.assertEqual(len(admission.candidates), 1)
        self.assertEqual(admission.candidates[0]["metadata"]["scope_match_status"], "exact")
        # ``candidate_8`` fails both source-anchored comparison ranges, so
        # there is one content-free diagnostic per failed answer scope.
        self.assertEqual(len(admission.rejections), 2)
        self.assertTrue(all(
            item.reason_code.startswith("scope_mismatch")
            for item in admission.rejections
        ))

    def test_unproven_project_value_cannot_create_a_hard_boundary(self):
        unproven = ApplicabilityScope(
            project="中青建安",
            explicit_project=True,
        )

        self.assertFalse(unproven.has_project_constraint)
        self.assertFalse(unproven.has_scope_constraint)

    def test_project_noun_inside_the_question_is_not_fabricated_as_scope(self):
        # ``项目等级`` is the subject of a relationship, not a named project.
        # A project parser that accepts it would fail-close valid evidence and
        # make ordinary multi-hop questions look like cross-project conflicts.
        for question in (
            "审批额度由项目等级决定是多少",
            "确认项目等级对应的适用分类",
        ):
            with self.subTest(question=question):
                scope = extract_applicability_scope(question)
                self.assertFalse(scope.has_project_constraint)
                self.assertIsNone(scope.project)

    def test_document_version_field_does_not_absorb_business_amounts(self):
        candidate = {
            "content": (
                "所属产品：CloudPivot；产品版本：6。"
                "住宿标准：A级不超过1200元/天。"
            ),
        }

        identity = extract_document_constraint_identity(candidate)
        evaluation = evaluate_candidate_constraints(
            extract_applicability_scope("CloudPivot 6 的住宿标准"),
            candidate,
        )

        self.assertEqual(identity.versions, ("6",))
        self.assertNotIn("1200", identity.versions)
        self.assertEqual(evaluation.status, "exact")
        self.assertEqual(evaluation.candidate_versions, ("6",))

    def test_document_version_field_accepts_only_controlled_version_list_entries(self):
        candidate = {
            "content": (
                "所属产品：CloudPivot；产品版本：6、7 / 8.2.75，"
                "住宿标准1200元/天。"
            ),
        }

        identity = extract_document_constraint_identity(candidate)

        self.assertEqual(identity.versions, ("6", "7", "8.2.75"))
        self.assertNotIn("1200", identity.versions)

    def test_generic_product_version_field_uses_the_same_bounded_parser(self):
        candidate = {
            "content": (
                "所属产品：CloudPivot；版本：CloudPivot 6。"
                "住宿标准：A级不超过1200元/天。"
            ),
        }

        evaluation = evaluate_candidate_constraints(
            extract_applicability_scope("CloudPivot 6 的住宿标准"),
            candidate,
        )

        self.assertEqual(evaluation.status, "exact")
        self.assertEqual(evaluation.candidate_versions, ("6",))


if __name__ == "__main__":
    unittest.main()
