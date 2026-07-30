import unittest

from core.document_parser import _is_placeholder_only_section, parse_markdown_content


class MarkdownParserShortContentTests(unittest.TestCase):
    def test_preserves_short_cjk_content(self) -> None:
        chunks = parse_markdown_content("测试", "测试文档")

        self.assertEqual(len(chunks), 1)
        self.assertIn("测试", chunks[0]["content"])

    def test_preserves_short_latin_content(self) -> None:
        chunks = parse_markdown_content("OK", "Status")

        self.assertEqual(len(chunks), 1)
        self.assertIn("OK", chunks[0]["content"])

    def test_preserves_short_meaningful_content(self) -> None:
        chunks = parse_markdown_content("同意", "审批结论")

        self.assertEqual(len(chunks), 1)
        self.assertIn("同意", chunks[0]["content"])

    def test_preserves_heading_only_document(self) -> None:
        chunks = parse_markdown_content("# 测试", "测试文档")

        self.assertEqual(len(chunks), 1)
        self.assertIn("测试", chunks[0]["content"])

    def test_blank_content_remains_empty(self) -> None:
        self.assertEqual(parse_markdown_content("  \n\n\t", "空文档"), [])


class MarkdownParserPlaceholderTests(unittest.TestCase):
    def test_recognizes_exact_placeholder_values(self) -> None:
        for value in ("无", "暂无", "无内容", "不涉及", "N/A", "n/a", "-"):
            with self.subTest(value=value):
                self.assertTrue(_is_placeholder_only_section(value))

    def test_recognizes_blockquoted_template_placeholder(self) -> None:
        self.assertTrue(_is_placeholder_only_section("> 无"))
        self.assertTrue(_is_placeholder_only_section("> 暂无\n> 不涉及"))

    def test_does_not_match_meaningful_short_or_containing_text(self) -> None:
        for value in ("测试", "同意", "OK", "没有权限", "暂无处理方案"):
            with self.subTest(value=value):
                self.assertFalse(_is_placeholder_only_section(value))

    def test_filters_placeholder_only_structured_sections(self) -> None:
        content = """# 一、基本信息
所属产品：云枢

# 二、问题描述
> 无

# 三、原因分析
暂无

# 四、解决方案
开启统一失败提示。
"""

        chunks = parse_markdown_content(content, "配置说明")

        combined = "\n".join(chunk["content"] for chunk in chunks)
        self.assertIn("所属产品：云枢", combined)
        self.assertIn("开启统一失败提示", combined)
        self.assertNotIn("二、问题描述", combined)
        self.assertNotIn("三、原因分析", combined)

    def test_placeholder_only_document_does_not_use_raw_fallback(self) -> None:
        content = "# 二、问题描述\n> 无\n\n# 三、原因分析\nN/A"

        self.assertEqual(parse_markdown_content(content, "空模板"), [])

    def test_preserves_fenced_and_indented_code(self) -> None:
        for content in ("```text\nN/A\n```", "    N/A"):
            with self.subTest(content=content):
                chunks = parse_markdown_content(content, "代码示例")
                self.assertEqual(len(chunks), 1)
                self.assertIn("N/A", chunks[0]["content"])

    def test_preserves_markdown_table_with_placeholder_cell(self) -> None:
        content = """# 状态表
| 字段 | 值 |
| --- | --- |
| 说明 | 无 |
"""

        chunks = parse_markdown_content(content, "表格说明")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["metadata"]["type"], "table")
        self.assertIn("| 说明 | 无 |", chunks[0]["content"])


if __name__ == "__main__":
    unittest.main()
