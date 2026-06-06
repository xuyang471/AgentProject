import unittest
from unittest.mock import patch

from services.qa_service import answer_question


class QaServiceTests(unittest.TestCase):
    def test_answer_question_returns_contextual_text(self) -> None:
        parsed_documents = [
            {
                "file_name": "demo.docx",
                "file_type": "docx",
                "blocks": [
                    {"type": "text", "content": "产品最大功率是 120W。", "source": "demo.docx 第 1 段", "page": None},
                ],
            }
        ]

        answer = answer_question("最大功率是多少", parsed_documents)

        self.assertIn("120W", answer)
        self.assertIn("回答", answer)

    def test_agent_answer_output_contains_react_trace(self) -> None:
        parsed_documents = [
            {
                "file_name": "demo.pdf",
                "file_type": "pdf",
                "blocks": [
                    {"type": "text", "content": "设备功率为 120W。", "source": "demo.pdf 第 1 页", "page": 1},
                ],
            }
        ]

        with patch("langchain_app.agent.create_qwen_chat_model", return_value=None):
            answer = answer_question("请计算 (120 * 3) / 1000 等于多少？", parsed_documents)

        self.assertIn("### ReAct 轨迹", answer)
        self.assertIn("Thought:", answer)
        self.assertIn("Action: calculator", answer)
        self.assertIn("Final Answer:", answer)


if __name__ == "__main__":
    unittest.main()
