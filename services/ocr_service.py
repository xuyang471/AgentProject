from __future__ import annotations

from pathlib import Path


def extract_text_from_image(image_path: str | Path) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ModuleNotFoundError:
        return ""

    try:
        with Image.open(image_path) as image:
            text = pytesseract.image_to_string(image, lang="chi_sim+eng")
            return text.strip()
    except Exception:
        return ""
