import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_app.agent import answer_with_agent
from langchain_core.tools import StructuredTool


MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
    b"\x00\x00\x03\x01\x01\x00\xc9\xfe\x92\xef\x00\x00\x00\x00IEND\xaeB`\x82"
)


class AgentToolPlanTests(unittest.TestCase):
    def test_agent_uses_tool_plan_when_llm_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "figure.png"
            image_path.write_bytes(MINIMAL_PNG)
            parsed_documents = [
                {
                    "file_name": "demo.pdf",
                    "file_type": "pdf",
                    "blocks": [
                        {
                            "type": "image",
                            "content": "图1 神经语言模型结构示意图",
                            "source": "demo.pdf 第 2 页 图注区域裁切 1",
                            "image_index": 1,
                            "image_path": str(image_path),
                            "ocr_text": "图1 神经语言模型",
                            "description": "神经语言模型结构示意图",
                            "page": 2,
                        }
                    ],
                }
            ]

            with patch("langchain_app.agent.create_qwen_chat_model", return_value=None):
                result = answer_with_agent("图1 表达了什么？", parsed_documents)

        self.assertIsNotNone(result)
        self.assertIn("answer", result)
        self.assertTrue(result.get("steps"))
        self.assertEqual(result["steps"][0]["tool"], "search_document_blocks")
        self.assertIn("Thought:", result.get("react_trace", ""))
        self.assertIn("Action:", result.get("react_trace", ""))
        self.assertIn("Observation:", result.get("react_trace", ""))
        self.assertIn("Final Answer:", result.get("react_trace", ""))

    def test_agent_uses_calculator_tool_plan_for_math_question(self) -> None:
        parsed_documents = [
            {
                "file_name": "demo.pdf",
                "file_type": "pdf",
                "blocks": [
                    {
                        "type": "text",
                        "page": 1,
                        "content": "设备功率为 120W，连续工作 3 小时。",
                        "source": "demo.pdf 第 1 页",
                    }
                ],
            }
        ]

        with patch("langchain_app.agent.create_qwen_chat_model", return_value=None):
            result = answer_with_agent("请计算 (120 * 3) / 1000 等于多少？", parsed_documents)

        self.assertIsNotNone(result)
        self.assertIn("结果：0.36", result["answer"])
        self.assertEqual(result["steps"][0]["tool"], "calculator")
        self.assertIn("Action: calculator", result.get("react_trace", ""))

    def test_agent_uses_web_search_tool_plan_for_external_question(self) -> None:
        def fake_web_search(query: str, top_k: int = 5) -> str:
            return (
                '[{"title":"Qwen 官方文档","url":"https://example.com/qwen",'
                '"snippet":"这是官方 API 说明。"}]'
            )

        parsed_documents = [
            {
                "file_name": "demo.pdf",
                "file_type": "pdf",
                "blocks": [
                    {
                        "type": "text",
                        "page": 1,
                        "content": "本文介绍 Qwen 模型。",
                        "source": "demo.pdf 第 1 页",
                    }
                ],
            }
        ]

        fake_tools = [
            StructuredTool.from_function(
                func=fake_web_search,
                name="web_search",
                description="联网查询外部资料。",
            )
        ]

        with patch("langchain_app.agent.create_qwen_chat_model", return_value=None):
            with patch("langchain_app.agent.create_document_tools", return_value=fake_tools):
                result = answer_with_agent("请联网查询 Qwen 最新官方文档", parsed_documents)

        self.assertIsNotNone(result)
        self.assertIn("联网查询工具", result["answer"])
        self.assertEqual(result["steps"][0]["tool"], "web_search")
        self.assertIn("Action: web_search", result.get("react_trace", ""))


if __name__ == "__main__":
    unittest.main()
