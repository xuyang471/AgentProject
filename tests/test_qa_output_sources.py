import unittest
from unittest.mock import patch

from services.qa_service import answer_question


class QaOutputSourcesTests(unittest.TestCase):
    def test_rag_output_contains_sources_section(self) -> None:
        parsed_documents = [
            {
                "file_name": "demo.pdf",
                "file_type": "pdf",
                "blocks": [
                    {"type": "text", "content": "产品最大功率为 120W。", "source": "demo.pdf 第 1 页", "page": 1},
                ],
            }
        ]

        fake_result = {
            "mode": "rag",
            "reason": "问题偏向文本事实检索，优先走 RAG。",
            "answer": "产品最大功率为 120W。",
            "sources": ["demo.pdf | demo.pdf 第 1 页 | page=1 | type=text"],
        }

        with patch("services.qa_service.answer_with_workflow", return_value=fake_result):
            output = answer_question("最大功率是多少", parsed_documents)

        self.assertIn("检索来源", output)
        self.assertIn("demo.pdf", output)
        self.assertIn("结果说明", output)


if __name__ == "__main__":
    unittest.main()
