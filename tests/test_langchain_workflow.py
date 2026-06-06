import unittest
from unittest.mock import patch

from langchain_app.graph_workflow import answer_with_langgraph_workflow
from langchain_app.workflow import answer_with_workflow


class LangChainWorkflowTests(unittest.TestCase):
    def test_answer_with_workflow_returns_none_without_documents(self) -> None:
        result = answer_with_workflow("最大功率是多少", [])
        self.assertIsNone(result)

    def test_langgraph_workflow_returns_none_without_documents(self) -> None:
        result = answer_with_langgraph_workflow("总结一下", [])
        self.assertIsNone(result)

    def test_workflow_falls_back_when_langgraph_returns_none(self) -> None:
        parsed_documents = [
            {
                "file_name": "demo.pdf",
                "file_type": "pdf",
                "blocks": [{"type": "text", "content": "Redis cache", "source": "demo.pdf 第 1 页"}],
            }
        ]
        with patch("langchain_app.workflow.answer_with_langgraph_workflow", return_value=None):
            with patch("langchain_app.workflow.answer_with_langchain", return_value={"answer": "ok", "sources": []}):
                result = answer_with_workflow("总结一下", parsed_documents)

        self.assertIsNotNone(result)
        self.assertEqual(result["workflow_engine"], "function")


if __name__ == "__main__":
    unittest.main()
