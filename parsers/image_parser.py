from __future__ import annotations

from pathlib import Path
from typing import Dict

from services.llm_service import analyze_image
from services.ocr_service import extract_text_from_image


def build_image_block(file_path: Path, source: str, page: int | None = None, image_index: int = 1) -> Dict:
    try:
        from PIL import Image
    except ModuleNotFoundError:
        return {
            "type": "image",
            "page": page,
            "image_index": image_index,
            "ocr_text": "",
            "description": "未安装 Pillow，暂无法读取图片尺寸与元信息。",
            "content": "未安装 Pillow，暂无法读取图片尺寸与元信息。",
            "source": source,
        }

    with Image.open(file_path) as image:
        width, height = image.size
        image_format = image.format or file_path.suffix.replace(".", "").upper()

    base_description = f"图片文件，尺寸约为 {width}x{height}，格式为 {image_format}。"
    local_ocr_text = extract_text_from_image(file_path)
    analysis = analyze_image(file_path, local_ocr_text=local_ocr_text)
    model_description = analysis["description"]
    ocr_text = analysis["ocr_text"]
    description = f"{base_description} {model_description}".strip()
    content = description if not ocr_text else f"{description}\nOCR文本：{ocr_text}"

    return {
        "type": "image",
        "page": page,
        "image_index": image_index,
        "image_path": str(file_path),
        "image_width": width,
        "image_height": height,
        "ocr_text": ocr_text,
        "description": description,
        "content": content,
        "source": source,
    }


def parse_image(file_path: Path) -> Dict:
    block = build_image_block(
        file_path=file_path,
        source=f"{file_path.name} 图片 1",
        page=None,
        image_index=1,
    )

    return {
        "file_name": file_path.name,
        "file_type": "image",
        "blocks": [block],
    }
