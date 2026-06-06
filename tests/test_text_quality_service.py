import unittest

from services.text_quality_service import clean_source_excerpt, clean_text_artifacts, is_low_quality_text


class TextQualityServiceTests(unittest.TestCase):
    def test_clean_text_artifacts_removes_repeated_digits(self) -> None:
        text = "444444444444444444444444 Word2vec 的工作原理"
        cleaned = clean_text_artifacts(text)

        self.assertNotIn("444444444", cleaned)
        self.assertIn("Word2vec", cleaned)

    def test_low_quality_text_detects_pdf_glyph_noise(self) -> None:
        text = "/G1/G1/G1/G1/G1 /c1/c2/c3/c4/c5/c6/c7"

        self.assertTrue(is_low_quality_text(text))

    def test_clean_source_excerpt_hides_low_quality_excerpt(self) -> None:
        text = "/G1/G1/G1/G1/G1 /c1/c2/c3/c4/c5/c6/c7"
        excerpt = clean_source_excerpt(text)

        self.assertIn("噪声", excerpt)


if __name__ == "__main__":
    unittest.main()
