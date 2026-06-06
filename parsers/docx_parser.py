from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from services.file_service import ensure_directory

from .image_parser import build_image_block


def parse_docx(file_path: Path) -> Dict:
    try:
        from docx import Document
    except ModuleNotFoundError:
        return {
            "file_name": file_path.name,
            "file_type": "docx",
            "blocks": [
                {
                    "type": "text",
                    "page": None,
                    "content": "未安装 python-docx，暂无法解析 DOCX 内容。",
                    "source": file_path.name,
                }
            ],
        }

    document = Document(str(file_path))
    blocks: List[Dict] = []
    extracted_dir = ensure_directory(file_path.parent / "_extracted" / file_path.stem)

    for index, paragraph in enumerate(document.paragraphs, start=1):
        text = paragraph.text.strip()
        if text:
            blocks.append(
                {
                    "type": "text",
                    "page": None,
                    "content": text,
                    "source": f"{file_path.name} 第 {index} 段",
                }
            )

    image_index = 1
    for rel in document.part.rels.values():
        rel_type = getattr(rel, "reltype", "")
        if "image" not in rel_type:
            continue

        target_part = getattr(rel, "target_part", None)
        if target_part is None:
            continue

        image_bytes = getattr(target_part, "blob", b"")
        image_name = getattr(target_part, "partname", f"image_{image_index}.png")
        if not image_bytes:
            continue

        image_path = extracted_dir / f"docx_{image_index}_{Path(str(image_name)).name}"
        image_path.write_bytes(image_bytes)
        blocks.append(
            build_image_block(
                file_path=image_path,
                source=f"{file_path.name} 嵌入图片 {image_index}",
                page=None,
                image_index=image_index,
            )
        )
        image_index += 1

    return {
        "file_name": file_path.name,
        "file_type": "docx",
        "blocks": blocks,
    }
