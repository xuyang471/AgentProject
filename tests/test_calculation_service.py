import unittest

from services.calculation_service import calculate_expression, normalize_expression


class CalculationServiceTests(unittest.TestCase):
    def test_normalize_expression_handles_percent_and_symbols(self) -> None:
        normalized = normalize_expression("120 × 30%")

        self.assertEqual(normalized, "120 * (30/100)")

    def test_calculate_expression_supports_basic_math(self) -> None:
        result = calculate_expression("(120 * 3) / 1000")

        self.assertEqual(result, 0.36)

    def test_calculate_expression_rejects_unsafe_syntax(self) -> None:
        with self.assertRaises(ValueError):
            calculate_expression("__import__('os').system('calc')")


if __name__ == "__main__":
    unittest.main()
