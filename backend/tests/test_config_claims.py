import unittest

from core.rag_v2.bridge_resolution import adjudicate_answer_claims
from core.rag_v2.config_claims import (
    extract_config_assignments,
    matching_config_assignments,
)


CONFIG_CHUNK = """【云枢6配置参数说明 › 四、解决方案】
```yaml cloudpivot: organization: defaultPwd: Authine@123456 #默认密码配置 login: error_reply_same: true #解决登录用户名枚举 accountpassword&verificationcode: true #开启短信双重校验 switch: force_change_default_password: true #默认密码强制修改 ```
"""


class ConfigClaimTests(unittest.TestCase):
    def test_extracts_inline_flattened_yaml_assignment_with_comment(self):
        claims = extract_config_assignments(CONFIG_CHUNK)
        target = [
            item for item in claims
            if item.path[-1] == "force_change_default_password"
        ]

        self.assertEqual(len(target), 1)
        self.assertIs(target[0].value, True)
        self.assertEqual(target[0].meaning, "默认密码强制修改")
        self.assertEqual(
            target[0].normalized_assignment,
            "force_change_default_password=true",
        )

    def test_selects_specific_assignment_instead_of_neighboring_default_password(self):
        claims = matching_config_assignments(
            "默认密码强制修改应该如何配置",
            CONFIG_CHUNK,
        )

        self.assertEqual(
            [item.path[-1] for item in claims],
            ["force_change_default_password"],
        )

    def test_broad_password_question_keeps_close_related_assignments(self):
        claims = matching_config_assignments(
            "云枢6如何修改默认密码",
            CONFIG_CHUNK,
        )

        self.assertEqual(
            {item.path[-1] for item in claims},
            {"defaultPwd", "force_change_default_password"},
        )
        self.assertEqual(
            {item.meaning for item in claims},
            {"默认密码配置", "默认密码强制修改"},
        )

    def test_configuration_assignment_closes_generic_answer_claim(self):
        assertions = adjudicate_answer_claims(
            "我现在想让云枢登录强制修改密码应该怎么办",
            CONFIG_CHUNK,
        )

        config_assertions = [
            item for item in assertions
            if item.result_kind == "config_assignment"
        ]
        self.assertEqual(len(config_assertions), 1)
        self.assertEqual(
            config_assertions[0].normalized_result,
            "force_change_default_password=true",
        )

    def test_concept_question_does_not_treat_configuration_as_definition(self):
        assertions = adjudicate_answer_claims(
            "登录用户名枚举是什么",
            CONFIG_CHUNK,
        )

        self.assertFalse(any(
            item.result_kind == "config_assignment" for item in assertions
        ))

    def test_standard_yaml_preserves_full_path_and_detects_conflicting_value(self):
        enabled = """cloudpivot:
  switch:
    force_change_default_password: true # 默认密码强制修改
"""
        disabled = enabled.replace("true", "false")

        left = matching_config_assignments("默认密码强制修改怎么配置", enabled)
        right = matching_config_assignments("默认密码强制修改怎么配置", disabled)

        self.assertEqual(
            left[0].normalized_assignment,
            "cloudpivot.switch.force_change_default_password=true",
        )
        self.assertEqual(
            right[0].normalized_assignment,
            "cloudpivot.switch.force_change_default_password=false",
        )
        self.assertEqual(left[0].normalized_path, right[0].normalized_path)


if __name__ == "__main__":
    unittest.main()
