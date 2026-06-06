import unittest

from langchain_app.mcp_client import StdioMCPClient


class McpClientTests(unittest.TestCase):
    def test_client_can_start_ping_and_list_tools(self) -> None:
        client = StdioMCPClient(parsed_documents=[], session_id="")
        try:
            ping_result = client.ping()
            tools = client.list_tools()
        finally:
            client.close()

        self.assertTrue(ping_result.get("pong"))
        self.assertIn("search_document_blocks", tools)
        self.assertIn("calculator", tools)
        self.assertIn("web_search", tools)


if __name__ == "__main__":
    unittest.main()
