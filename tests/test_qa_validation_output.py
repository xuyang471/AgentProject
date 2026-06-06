import unittest

from services.qa_service import answer_question


class QaValidationOutputTests(unittest.TestCase):
    def test_answer_output_contains_validation_section(self) -> None:
        parsed_documents = [
            {
                "file_name": "demo.docx",
                "file_type": "docx",
                "blocks": [
                    {
                        "type": "text",
                        "content": "产品最大功率是 120W。",
                        "source": "demo.docx 第 1 段",
                        "page": None,
                    }
                ],
            }
        ]

        answer = answer_question("最大功率是多少", parsed_documents)

        self.assertIn("答案校验", answer)
        self.assertIn("可信度", answer)


if __name__ == "__main__":
    unittest.main()
