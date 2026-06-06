import unittest

from services.qa_service import _format_rag_sources


class QaSourceFormattingTests(unittest.TestCase):
    def test_format_rag_sources_shows_text_excerpt(self) -> None:
        sources = [
            {
                "source": "demo.pdf 第 2 页",
                "block_type": "text",
                "page": 2,
                "content": "Word2vec 包含 CBOW 和 Skip-gram 两种模型。",
            }
        ]
        output = _format_rag_sources(sources)
        self.assertIn("内容摘录", output)
        self.assertIn("CBOW", output)

    def test_format_rag_sources_shows_image_description(self) -> None:
        sources = [
            {
                "source": "demo.pdf 第 4 页 图片 2",
                "block_type": "image",
                "page": 4,
                "description": "一张展示中文词语与余弦距离的表格截图。",
                "content": "图片内容",
                "image_markdown": "![检索命中的图片](/tmp/demo.png)",
            }
        ]
        output = _format_rag_sources(sources)
        self.assertIn("类型：图片", output)
        self.assertIn("余弦距离", output)
        self.assertIn("![检索命中的图片]", output)


if __name__ == "__main__":
    unittest.main()
