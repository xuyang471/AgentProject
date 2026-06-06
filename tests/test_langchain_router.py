import unittest

from langchain_app.router import route_question


class LangChainRouterTests(unittest.TestCase):
    def test_route_question_to_agent_for_image_question(self) -> None:
        result = route_question("图2中的曲线代表什么？")
        self.assertEqual(result["route"], "agent")
        self.assertEqual(result["intent"], "image")

    def test_route_question_to_rag_for_text_question(self) -> None:
        result = route_question("产品最大功率是多少？")
        self.assertEqual(result["route"], "rag")
        self.assertEqual(result["intent"], "text")

    def test_route_question_to_agent_for_calculation_question(self) -> None:
        result = route_question("请计算 120 * 3 / 1000 等于多少？")
        self.assertEqual(result["route"], "agent")
        self.assertEqual(result["intent"], "calculation")

    def test_route_question_to_agent_for_web_search_question(self) -> None:
        result = route_question("请联网查询 Qwen 最新官方文档")
        self.assertEqual(result["route"], "agent")
        self.assertEqual(result["intent"], "web_search")


if __name__ == "__main__":
    unittest.main()
