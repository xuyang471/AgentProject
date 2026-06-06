from pathlib import Path
import tempfile
import unittest

from services.parser_service import parse_files


class ParserServiceTests(unittest.TestCase):
    def test_unsupported_file_returns_placeholder_block(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "demo.txt"
            path.write_text("hello", encoding="utf-8")

            parsed = parse_files([path])

        self.assertEqual(parsed[0]["file_type"], "unsupported")
        self.assertEqual(parsed[0]["blocks"][0]["content"], "文件格式不支持。")


if __name__ == "__main__":
    unittest.main()
