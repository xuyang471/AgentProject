import unittest
from pathlib import Path
from unittest.mock import patch

from mcp_server import LocalMCPServer
from services.state_service import (
    create_analysis_session,
    register_uploaded_files,
    save_parsed_documents,
    save_qa_record,
    save_report_record,
)
from storage.db import DB_PATH, init_database


class McpServerTests(unittest.TestCase):
    def setUp(self) -> None:
        if DB_PATH.exists():
            DB_PATH.unlink()
        init_database()

    def test_server_can_read_report_and_history_resources(self) -> None:
        session_id = create_analysis_session("mcp-demo")
        upload_path = Path("D:/AgentProject/output/temp/test/mcp_demo.pdf")
        register_uploaded_files(session_id, [upload_path])
        save_parsed_documents(
            session_id,
            [
                {
                    "file_name": "mcp_demo.pdf",
                    "file_type": "pdf",
                    "blocks": [
                        {
                            "type": "text",
                            "page": 1,
                            "content": "Word2vec 包含 CBOW 和 Skip-gram 两种训练模型。",
                            "source": "mcp_demo.pdf 第 1 页",
                        }
                    ],
                }
            ],
        )
        save_report_record(
            session_id,
            Path("D:/AgentProject/output/reports/mcp_demo/report.md"),
            "# 文档总结报告\n## 1. 核心摘要\n这是 MCP 资源测试报告。",
        )
        save_qa_record(
            session_id=session_id,
            question="Word2vec 有哪些模型？",
            answer="包含 CBOW 和 Skip-gram。",
            route_type="rag",
            sources=[{"source": "mcp_demo.pdf 第 1 页"}],
        )

        server = LocalMCPServer(session_id=session_id)
        report = server.read_resource(f"session://{session_id}/report")
        history = server.read_resource(f"session://{session_id}/qa-history")

        self.assertIn("核心摘要", report.get("content", ""))
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["question"], "Word2vec 有哪些模型？")

    def test_server_search_tool_returns_matches(self) -> None:
        parsed_documents = [
            {
                "file_name": "demo.pdf",
                "file_type": "pdf",
                "blocks": [
                    {
                        "type": "text",
                        "page": 1,
                        "content": "系统采用 Redis 缓存热点数据并结合 Quartz 定时任务。",
                        "source": "demo.pdf 第 1 页",
                    }
                ],
            }
        ]
        server = LocalMCPServer(parsed_documents=parsed_documents)
        result = server.call_tool("search_document_blocks", query="Redis")

        self.assertTrue(result["ok"])
        self.assertIn("Redis", result["data"][0]["content"])

    def test_server_calculator_tool_returns_result(self) -> None:
        server = LocalMCPServer()
        result = server.call_tool("calculator", expression="(120 * 3) / 1000")

        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["result"], 0.36)

    def test_server_web_search_tool_returns_matches(self) -> None:
        server = LocalMCPServer()
        with patch(
            "mcp_server.tools.search_web",
            return_value=[{"title": "Qwen API", "url": "https://example.com", "snippet": "官方文档"}],
        ):
            result = server.call_tool("web_search", query="Qwen 最新 API 文档", top_k=3)

        self.assertTrue(result["ok"])
        self.assertEqual(result["data"][0]["title"], "Qwen API")

    def test_server_supports_standard_analysis_resource_uri(self) -> None:
        session_id = create_analysis_session("mcp-uri-demo")
        server = LocalMCPServer(session_id=session_id)

        resources = server.list_resources()
        latest = server.read_resource("analysis://sessions/latest")

        self.assertIn(f"analysis://sessions/{session_id}/report", resources)
        self.assertEqual(latest["id"], session_id)


if __name__ == "__main__":
    unittest.main()
