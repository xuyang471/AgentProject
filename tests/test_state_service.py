import unittest
from pathlib import Path

from services.state_service import (
    SESSION_STATUS_QA_AVAILABLE,
    create_analysis_session,
    get_latest_session,
    list_sessions,
    load_session_context,
    register_uploaded_files,
    save_parsed_documents,
    save_qa_record,
    save_report_record,
)
from storage.db import DB_PATH, init_database


class StateServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        if DB_PATH.exists():
            DB_PATH.unlink()
        init_database()

    def test_state_persistence_round_trip(self) -> None:
        session_id = create_analysis_session("demo-session")
        upload_path = Path("D:/AgentProject/output/temp/test/demo.pdf")
        register_uploaded_files(session_id, [upload_path])

        parsed_documents = [
            {
                "file_name": "demo.pdf",
                "file_type": "pdf",
                "blocks": [
                    {
                        "type": "text",
                        "page": 1,
                        "content": "Word2vec 包含 CBOW 和 Skip-gram 两种训练模型。",
                        "source": "demo.pdf 第 1 页",
                    },
                    {
                        "type": "image",
                        "page": 2,
                        "image_index": 1,
                        "image_path": "D:/AgentProject/output/temp/test/demo_image.png",
                        "description": "一张展示神经语言模型结构的示意图。",
                        "content": "一张展示神经语言模型结构的示意图。",
                        "source": "demo.pdf 第 2 页 图片 1",
                    },
                ],
            }
        ]
        save_parsed_documents(session_id, parsed_documents)

        report_path = Path("D:/AgentProject/output/reports/demo/report.md")
        report_markdown = "# 文档总结报告\n## 1. 核心摘要\n这是一个关于 Word2vec 的测试报告。"
        save_report_record(session_id, report_path, report_markdown)
        save_qa_record(
            session_id=session_id,
            question="Word2vec 有哪些训练模型？",
            answer="Word2vec 包含 CBOW 和 Skip-gram 两种训练模型。",
            route_type="rag",
            sources=[{"source": "demo.pdf 第 1 页", "block_type": "text"}],
        )

        latest_session = get_latest_session()
        self.assertIsNotNone(latest_session)
        self.assertEqual(latest_session["id"], session_id)
        self.assertEqual(latest_session["status"], SESSION_STATUS_QA_AVAILABLE)

        context = load_session_context(session_id)
        self.assertIsNotNone(context)
        self.assertEqual(len(context["documents"]), 1)
        self.assertEqual(context["documents"][0]["file_name"], "demo.pdf")
        self.assertEqual(len(context["documents"][0]["blocks"]), 2)
        self.assertEqual(context["report"]["report_path"], str(report_path))
        self.assertEqual(
            context["latest_qa"]["answer"],
            "Word2vec 包含 CBOW 和 Skip-gram 两种训练模型。",
        )
        self.assertEqual(len(context["latest_qa"]["sources"]), 1)

    def test_list_sessions_returns_latest_first(self) -> None:
        first_session = create_analysis_session("session-a")
        second_session = create_analysis_session("session-b")

        sessions = list_sessions(limit=10)

        self.assertGreaterEqual(len(sessions), 2)
        self.assertEqual(sessions[0]["id"], second_session)
        self.assertEqual(sessions[1]["id"], first_session)


if __name__ == "__main__":
    unittest.main()
