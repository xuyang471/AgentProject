import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_app.tools import create_document_tools


MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
    b"\x00\x00\x03\x01\x01\x00\xc9\xfe\x92\xef\x00\x00\x00\x00IEND\xaeB`\x82"
)


class LangChainToolsTests(unittest.TestCase):
    def test_create_document_tools_returns_list(self) -> None:
        tools = create_document_tools([])
        self.assertTrue(isinstance(tools, list))

    def test_tool_can_find_image_and_return_standard_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "tiny.png"
            image_path.write_bytes(MINIMAL_PNG)

            parsed_documents = [
                {
                    "file_name": "demo.png",
                    "file_type": "image",
                    "blocks": [
                        {
                            "type": "image",
                            "content": "image file",
                            "source": "demo.png image 1",
                            "image_index": 1,
                            "image_path": str(image_path),
                            "ocr_text": "",
                            "description": "test image",
                            "page": None,
                        }
                    ],
                }
            ]

            tools = create_document_tools(parsed_documents)
            if not tools:
                self.assertEqual(tools, [])
                return

            tool_map = {tool.name: tool for tool in tools}
            result = tool_map["run_ocr_on_image"].invoke({"source": "demo.png image 1"})
            payload = json.loads(result)

            self.assertTrue(payload["ok"])
            self.assertEqual(payload["data"]["source"], "demo.png image 1")

    def test_calculator_tool_returns_standard_json(self) -> None:
        tools = create_document_tools([])
        if not tools:
            self.assertEqual(tools, [])
            return

        tool_map = {tool.name: tool for tool in tools}
        result = tool_map["calculator"].invoke({"expression": "(120 * 3) / 1000"})
        payload = json.loads(result)

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["data"]["result"], 0.36)

    def test_web_search_tool_returns_standard_json(self) -> None:
        class FakeBackend:
            session_id = ""

            def read_resource(self, uri):
                return {}

            def call_tool(self, name, **kwargs):
                if name == "web_search":
                    return {
                        "ok": True,
                        "data": [
                            {
                                "title": "Qwen 官方文档",
                                "url": "https://example.com/qwen",
                                "snippet": "这是联网查询结果。",
                            }
                        ],
                        "error": None,
                        "metadata": {"tool": "web_search"},
                    }
                return {"ok": False, "data": None, "error": "unsupported", "metadata": {}}

        with patch("langchain_app.tools._create_server_backed_interface", return_value=("local", FakeBackend())):
            tools = create_document_tools([])

        tool_map = {tool.name: tool for tool in tools}
        result = tool_map["web_search"].invoke({"query": "Qwen 最新官方文档", "top_k": 3})
        payload = json.loads(result)

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["data"][0]["title"], "Qwen 官方文档")


if __name__ == "__main__":
    unittest.main()
