import unittest

from langchain_app.retrievers import create_hybrid_retriever


class LangChainRetrieverFallbackTests(unittest.TestCase):
    def test_create_hybrid_retriever_handles_empty_documents(self) -> None:
        retriever = create_hybrid_retriever([], top_k=3)
        self.assertIsNone(retriever)


if __name__ == "__main__":
    unittest.main()
