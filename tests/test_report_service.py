import tempfile
import unittest
from pathlib import Path

from services.report_service import _parse_markdown_table, build_markdown_report


MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
    b"\x00\x00\x03\x01\x01\x00\xc9\xfe\x92\xef\x00\x00\x00\x00IEND\xaeB`\x82"
)


class ReportServiceTests(unittest.TestCase):
    def test_build_markdown_report_contains_valid_local_image_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "demo.png"
            image_path.write_bytes(MINIMAL_PNG)

            parsed_documents = [
                {
                    "file_name": "demo.pdf",
                    "file_type": "pdf",
                    "blocks": [
                        {
                            "type": "text",
                            "content": "The system uses Redis cache for hot data.",
                            "source": "demo.pdf page 1",
                            "page": 1,
                        },
                        {
                            "type": "image",
                            "content": "Architecture image.",
                            "description": "Architecture diagram.",
                            "source": "demo.pdf image 1",
                            "page": 1,
                            "image_path": str(image_path),
                        },
                        {
                            "type": "table",
                            "content": "| parameter | value |\n| --- | --- |\n| vocabulary | 348863 |",
                            "source": "demo.pdf table 1",
                            "page": 2,
                            "table_index": 1,
                            "table_image_path": str(image_path),
                        },
                    ],
                }
            ]

            report = build_markdown_report(parsed_documents)

        self.assertIn("![detected-image](", report)
        self.assertIn("![table-preview](", report)
        self.assertNotIn("| parameter | value |", report)
        self.assertNotIn("![detected-image](", report.replace("](", "]missing("))
        self.assertIn(str(image_path.resolve()), report)

    def test_parse_markdown_table_removes_separator_row(self) -> None:
        rows = _parse_markdown_table("| parameter | value |\n| --- | --- |\n| vocabulary | 348863 |")

        self.assertEqual(rows, [["parameter", "value"], ["vocabulary", "348863"]])


if __name__ == "__main__":
    unittest.main()
