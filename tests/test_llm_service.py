from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from services.llm_service import analyze_image


MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
    b"\x00\x00\x03\x01\x01\x00\xc9\xfe\x92\xef\x00\x00\x00\x00IEND\xaeB`\x82"
)


class LlmServiceTests(unittest.TestCase):
    def test_analyze_image_falls_back_without_dashscope_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "tiny.png"
            image_path.write_bytes(MINIMAL_PNG)

            with patch.dict("os.environ", {"DASHSCOPE_API_KEY": ""}, clear=False):
                result = analyze_image(image_path, local_ocr_text="示例文字")

        self.assertEqual(result["ocr_text"], "示例文字")
        self.assertIn("未进行云端视觉理解", result["description"])


if __name__ == "__main__":
    unittest.main()
