import unittest

from services.answer_validation_service import validate_answer


class AnswerValidationServiceTests(unittest.TestCase):
    def test_validate_answer_with_sources_returns_grounded_result(self) -> None:
        result = validate_answer(
            "Word2vec 包含 CBOW 和 Skip-gram 两种训练模型。",
            route_type="rag",
            sources=[
                {
                    "source": "demo.pdf 第 1 页",
                    "content": "Word2vec 包含 CBOW 和 Skip-gram 两种训练模型。",
                    "block_type": "text",
                }
            ],
        )

        self.assertTrue(result["grounded"])
        self.assertIn(result["confidence"], {"中", "高"})

    def test_validate_answer_without_sources_returns_low_confidence(self) -> None:
        result = validate_answer(
            "这可能是一个非常复杂的系统设计。",
            route_type="agent",
            steps=[],
            sources=[],
        )

        self.assertEqual(result["confidence"], "低")
        self.assertFalse(result["grounded"])


if __name__ == "__main__":
    unittest.main()
