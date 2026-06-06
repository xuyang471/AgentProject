import json
import unittest
from unittest.mock import patch

from services.web_search_service import search_web


class _FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None


class WebSearchServiceTests(unittest.TestCase):
    def test_search_web_parses_tavily_style_results(self) -> None:
        payload = {
            "results": [
                {
                    "title": "Qwen API 文档",
                    "url": "https://example.com/qwen",
                    "content": "Qwen 提供文本与多模态接口。",
                }
            ]
        }

        with patch.dict("os.environ", {"TAVILY_API_KEY": "demo-key"}, clear=False):
            with patch("services.web_search_service.urlopen", return_value=_FakeResponse(payload)):
                results = search_web("Qwen API 文档", top_k=3)

        self.assertEqual(results[0]["title"], "Qwen API 文档")
        self.assertEqual(results[0]["url"], "https://example.com/qwen")
        self.assertIn("多模态", results[0]["snippet"])

    def test_search_web_requires_api_key(self) -> None:
        with patch.dict("os.environ", {"TAVILY_API_KEY": ""}, clear=False):
            with self.assertRaises(RuntimeError):
                search_web("latest qwen docs")


if __name__ == "__main__":
    unittest.main()
