from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from parsers import parse_docx, parse_image, parse_pdf


SUPPORTED_EXTENSIONS = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".png": parse_image,
    ".jpg": parse_image,
    ".jpeg": parse_image,
}


def parse_files(file_paths: Iterable[Path]) -> List[dict]:
    parsed_documents: List[dict] = []

    for file_path in file_paths:
        parser = SUPPORTED_EXTENSIONS.get(file_path.suffix.lower())
        if not parser:
            parsed_documents.append(
                {
                    "file_name": file_path.name,
                    "file_type": "unsupported",
                    "blocks": [
                        {
                            "type": "text",
                            "page": None,
                            "content": "文件格式不支持。",
                            "source": file_path.name,
                        }
                    ],
                }
            )
            continue

        parsed_documents.append(parser(file_path))

    return parsed_documents
