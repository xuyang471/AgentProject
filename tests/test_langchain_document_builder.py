import unittest

from langchain_app.document_builder import build_langchain_documents


class LangChainDocumentBuilderTests(unittest.TestCase):
    def test_build_langchain_documents_handles_missing_dependency(self) -> None:
        parsed_documents = [
            {
                "file_name": "demo.pdf",
                "file_type": "pdf",
                "blocks": [
                    {
                        "type": "text",
                        "content": "最大功率为 120W。",
                        "source": "demo.pdf 第 1 页",
                        "page": 1,
                    }
                ],
            }
        ]

        documents = build_langchain_documents(parsed_documents)

        self.assertTrue(isinstance(documents, list))


if __name__ == "__main__":
    unittest.main()
