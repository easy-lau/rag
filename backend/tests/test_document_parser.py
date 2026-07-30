import unittest

from core.document_parser import parse_markdown_content


class MarkdownParserShortContentTests(unittest.TestCase):
    def test_preserves_short_cjk_content(self) -> None:
        chunks = parse_markdown_content("测试", "测试文档")

        self.assertEqual(len(chunks), 1)
        self.assertIn("测试", chunks[0]["content"])

    def test_preserves_short_latin_content(self) -> None:
        chunks = parse_markdown_content("OK", "Status")

        self.assertEqual(len(chunks), 1)
        self.assertIn("OK", chunks[0]["content"])

    def test_preserves_heading_only_document(self) -> None:
        chunks = parse_markdown_content("# 测试", "测试文档")

        self.assertEqual(len(chunks), 1)
        self.assertIn("测试", chunks[0]["content"])

    def test_blank_content_remains_empty(self) -> None:
        self.assertEqual(parse_markdown_content("  \n\n\t", "空文档"), [])


if __name__ == "__main__":
    unittest.main()
