from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
import re
from typing import Dict, List, Tuple

from services.file_service import ensure_directory
from services.logging_service import configure_runtime_logging
from services.ocr_service import extract_text_from_image
from services.text_quality_service import clean_text_artifacts, is_low_quality_text

from .image_parser import build_image_block

configure_runtime_logging()


def _clean_cell(value) -> str:
    if value is None:
        return ""
    return str(value).replace("\n", " ").strip()


def _table_to_markdown(table_rows: List[List[str]]) -> str:
    non_empty_rows = []
    for row in table_rows:
        cleaned_row = [_clean_cell(cell) for cell in row]
        if any(cell for cell in cleaned_row):
            non_empty_rows.append(cleaned_row)

    if not non_empty_rows:
        return ""

    max_cols = max(len(row) for row in non_empty_rows)
    normalized = [row + [""] * (max_cols - len(row)) for row in non_empty_rows]
    header = normalized[0]
    body = normalized[1:] or [[]]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * max_cols) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _extract_pdf_tables(file_path: Path) -> Dict[int, List[Dict]]:
    try:
        import pdfplumber
    except ModuleNotFoundError:
        return {}

    table_map: Dict[int, List[Dict]] = {}
    try:
        with pdfplumber.open(str(file_path)) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                page_tables = []
                find_tables = getattr(page, "find_tables", None)
                if callable(find_tables):
                    for table in find_tables() or []:
                        extracted_rows = table.extract() or []
                        markdown = _table_to_markdown(extracted_rows)
                        if markdown:
                            page_tables.append(
                                {
                                    "content": markdown,
                                    "bbox": _normalize_bbox(table.bbox) if getattr(table, "bbox", None) else None,
                                }
                            )
                else:
                    for table in page.extract_tables() or []:
                        markdown = _table_to_markdown(table)
                        if markdown:
                            page_tables.append({"content": markdown, "bbox": None})
                if page_tables:
                    table_map[page_index] = page_tables
    except Exception:
        return {}

    return table_map


def _bbox_area(bbox: Dict[str, float] | None) -> float:
    if not bbox:
        return 0.0
    width = max(0.0, float(bbox.get("x1", 0.0)) - float(bbox.get("x0", 0.0)))
    height = max(0.0, float(bbox.get("y1", 0.0)) - float(bbox.get("y0", 0.0)))
    return width * height


def _bbox_overlap_ratio(source_bbox: Dict[str, float] | None, target_bbox: Dict[str, float] | None) -> float:
    if not source_bbox or not target_bbox:
        return 0.0
    source_area = _bbox_area(source_bbox)
    if source_area <= 0:
        return 0.0

    x0 = max(float(source_bbox.get("x0", 0.0)), float(target_bbox.get("x0", 0.0)))
    y0 = max(float(source_bbox.get("y0", 0.0)), float(target_bbox.get("y0", 0.0)))
    x1 = min(float(source_bbox.get("x1", 0.0)), float(target_bbox.get("x1", 0.0)))
    y1 = min(float(source_bbox.get("y1", 0.0)), float(target_bbox.get("y1", 0.0)))

    if x1 <= x0 or y1 <= y0:
        return 0.0

    intersection_area = (x1 - x0) * (y1 - y0)
    return intersection_area / source_area


def _bbox_horizontal_overlap_ratio(source_bbox: Dict[str, float] | None, target_bbox: Dict[str, float] | None) -> float:
    if not source_bbox or not target_bbox:
        return 0.0
    source_width = max(0.0, float(source_bbox.get("x1", 0.0)) - float(source_bbox.get("x0", 0.0)))
    if source_width <= 0:
        return 0.0

    x0 = max(float(source_bbox.get("x0", 0.0)), float(target_bbox.get("x0", 0.0)))
    x1 = min(float(source_bbox.get("x1", 0.0)), float(target_bbox.get("x1", 0.0)))
    if x1 <= x0:
        return 0.0
    return (x1 - x0) / source_width


def _bbox_vertical_gap(anchor_bbox: Dict[str, float] | None, candidate_bbox: Dict[str, float] | None) -> float:
    if not anchor_bbox or not candidate_bbox:
        return float("inf")
    anchor_top = float(anchor_bbox.get("y0", 0.0))
    anchor_bottom = float(anchor_bbox.get("y1", 0.0))
    candidate_top = float(candidate_bbox.get("y0", 0.0))
    candidate_bottom = float(candidate_bbox.get("y1", 0.0))
    if candidate_top >= anchor_bottom:
        return candidate_top - anchor_bottom
    if candidate_bottom <= anchor_top:
        return anchor_top - candidate_bottom
    return 0.0


def _filter_layout_blocks_against_tables(layout_blocks: List[Dict], page_tables: List[Dict]) -> List[Dict]:
    if not layout_blocks or not page_tables:
        return layout_blocks

    filtered_blocks: List[Dict] = []
    for block in layout_blocks:
        if block.get("is_caption"):
            filtered_blocks.append(block)
            continue
        block_bbox = block.get("bbox")
        overlaps_table = any(_bbox_overlap_ratio(block_bbox, table.get("bbox")) >= 0.45 for table in page_tables)
        if not overlaps_table:
            filtered_blocks.append(block)
    return filtered_blocks


def _extract_pdf_text_fallback(file_path: Path) -> Dict[int, str]:
    try:
        import pdfplumber
    except ModuleNotFoundError:
        return {}

    text_map: Dict[int, str] = {}
    try:
        with pdfplumber.open(str(file_path)) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                text = clean_text_artifacts((page.extract_text() or "").strip())
                if text:
                    text_map[page_index] = text
    except Exception:
        return {}

    return text_map


def _normalize_bbox(bbox: Tuple[float, float, float, float] | List[float]) -> Dict[str, float]:
    x0, y0, x1, y1 = bbox
    return {
        "x0": round(float(x0), 2),
        "y0": round(float(y0), 2),
        "x1": round(float(x1), 2),
        "y1": round(float(y1), 2),
    }


def _normalize_layout_text_blocks(block_items: List[Dict]) -> List[Dict]:
    normalized_blocks: List[Dict] = []
    for block_item in block_items:
        cleaned = clean_text_artifacts(block_item.get("content", ""))
        if not cleaned or is_low_quality_text(cleaned):
            continue
        if len(cleaned) < 6:
            continue
        if any(_is_near_duplicate_text(existing.get("content", ""), cleaned) for existing in normalized_blocks):
            continue
        normalized_blocks.append(
            {
                "content": cleaned,
                "bbox": block_item.get("bbox"),
                "is_caption": bool(block_item.get("is_caption", False)),
            }
        )
    return normalized_blocks


def _extract_pdf_layout_text_blocks(file_path: Path) -> Dict[int, List[Dict]]:
    try:
        import fitz
    except ModuleNotFoundError:
        return {}

    layout_map: Dict[int, List[Dict]] = {}
    try:
        with fitz.open(str(file_path)) as pdf:
            for page_index, page in enumerate(pdf, start=1):
                raw_blocks = []
                for block in page.get_text("blocks") or []:
                    if len(block) < 5:
                        continue
                    x0, y0, x1, y1, text = block[:5]
                    text = str(text or "").strip()
                    if text:
                        raw_blocks.append(
                            {
                                "content": text,
                                "bbox": _normalize_bbox((x0, y0, x1, y1)),
                                "is_caption": _contains_figure_caption(text),
                            }
                        )
                normalized_blocks = _normalize_layout_text_blocks(raw_blocks)
                if normalized_blocks:
                    layout_map[page_index] = normalized_blocks
    except Exception:
        return {}

    return layout_map


def _contains_figure_caption(text: str) -> bool:
    if not text:
        return False
    if re.search("\u56fe\\s*\\d+|\u56fe\u8868\\s*\\d+", text, re.IGNORECASE):
        return True
    patterns = [
        r"图\s*\d+",
        r"图表\s*\d+",
        r"图\s*\d+",
        r"图表\s*\d+",
        r"fig\.?\s*\d+",
        r"figure\s*\d+",
    ]
    lowered = text.lower()
    return any(re.search(pattern, lowered, re.IGNORECASE) for pattern in patterns)


def _extract_caption_blocks(layout_blocks: List[Dict]) -> List[Dict]:
    return [block for block in layout_blocks if block.get("is_caption")]


def _normalize_text_for_compare(text: str) -> str:
    normalized = clean_text_artifacts(text).lower()
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"[^\w\u4e00-\u9fff]", "", normalized)
    return normalized


def _is_near_duplicate_text(primary_text: str, fallback_text: str) -> bool:
    primary_normalized = _normalize_text_for_compare(primary_text)
    fallback_normalized = _normalize_text_for_compare(fallback_text)
    if not primary_normalized or not fallback_normalized:
        return False
    if primary_normalized == fallback_normalized:
        return True
    if primary_normalized in fallback_normalized or fallback_normalized in primary_normalized:
        shorter = min(len(primary_normalized), len(fallback_normalized))
        longer = max(len(primary_normalized), len(fallback_normalized))
        return longer > 0 and shorter / longer >= 0.78
    return SequenceMatcher(None, primary_normalized, fallback_normalized).ratio() >= 0.9


def _merge_page_text(primary_text: str, fallback_text: str) -> str:
    primary_cleaned = clean_text_artifacts(primary_text)
    fallback_cleaned = clean_text_artifacts(fallback_text)
    primary_normalized = _normalize_text_for_compare(primary_cleaned)
    fallback_normalized = _normalize_text_for_compare(fallback_cleaned)

    if not primary_cleaned:
        return fallback_cleaned
    if not fallback_cleaned:
        return primary_cleaned
    if is_low_quality_text(primary_cleaned) and not is_low_quality_text(fallback_cleaned):
        return fallback_cleaned
    if is_low_quality_text(fallback_cleaned) and not is_low_quality_text(primary_cleaned):
        return primary_cleaned
    if primary_normalized and fallback_normalized:
        if primary_normalized in fallback_normalized:
            return fallback_cleaned
        if fallback_normalized in primary_normalized:
            return primary_cleaned
    if _is_near_duplicate_text(primary_cleaned, fallback_cleaned):
        return primary_cleaned if len(primary_cleaned) >= len(fallback_cleaned) else fallback_cleaned

    # Keep both extractions when they appear complementary.
    return f"{primary_cleaned}\n{fallback_cleaned}"


def _caption_crop_bounds(
    page_width: float,
    page_height: float,
    caption_bbox: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """Build a figure crop around a caption instead of using a whole-page screenshot."""
    caption_x0, caption_y0, caption_x1, caption_y1 = caption_bbox
    margin_x = max(18.0, page_width * 0.04)
    margin_y = max(18.0, page_height * 0.025)
    figure_window_height = min(max(220.0, page_height * 0.36), max(caption_y0, 1.0))

    x0 = max(0.0, min(caption_x0 - margin_x, page_width * 0.04))
    x1 = min(page_width, max(caption_x1 + margin_x, page_width * 0.96))
    y0 = max(0.0, caption_y0 - figure_window_height)
    y1 = min(page_height, caption_y1 + margin_y)

    max_crop_height = page_height * 0.62
    if y1 - y0 > max_crop_height:
        y0 = max(0.0, y1 - max_crop_height)
    return x0, y0, x1, y1


def _is_noise_image(block: Dict) -> bool:
    width = int(block.get("image_width", 0) or 0)
    height = int(block.get("image_height", 0) or 0)
    area = width * height
    ocr_text = str(block.get("ocr_text", "")).strip()
    description = str(block.get("description", "")).strip()

    if width and height and area <= 12000 and not ocr_text:
        if "纯白色背景" in description or "无明显内容" in description or "无任何可辨识" in description:
            return True
    return False


def _looks_like_scanned_page(
    merged_text: str,
    layout_blocks: List[Dict],
    page_images: List,
    raw_text: str = "",
    fallback_text: str = "",
) -> bool:
    merged_cleaned = clean_text_artifacts(merged_text)
    raw_cleaned = clean_text_artifacts(raw_text)
    fallback_cleaned = clean_text_artifacts(fallback_text)

    if layout_blocks and len(layout_blocks) >= 2:
        return False
    if merged_cleaned and not is_low_quality_text(merged_cleaned) and len(merged_cleaned) >= 120:
        return False
    if fallback_cleaned and not is_low_quality_text(fallback_cleaned) and len(fallback_cleaned) >= 120:
        return False
    if raw_cleaned and not is_low_quality_text(raw_cleaned) and len(raw_cleaned) >= 120:
        return False

    return bool(page_images) or not merged_cleaned or is_low_quality_text(merged_cleaned) or len(merged_cleaned) < 60


def _render_pdf_page(file_path: Path, page_index: int, output_dir: Path) -> Path | None:
    try:
        import fitz
    except ModuleNotFoundError:
        return None

    try:
        with fitz.open(str(file_path)) as pdf:
            page = pdf.load_page(page_index - 1)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image_path = output_dir / f"page_{page_index}_snapshot.png"
            pix.save(str(image_path))
            return image_path
    except Exception:
        return None


def _extract_page_ocr_text(file_path: Path, page_index: int, output_dir: Path) -> tuple[str, Path | None]:
    snapshot_path = _render_pdf_page(file_path, page_index, output_dir)
    if snapshot_path is None:
        return "", None
    ocr_text = clean_text_artifacts(extract_text_from_image(snapshot_path))
    return ocr_text, snapshot_path


def _render_pdf_table_crop(
    file_path: Path,
    page_index: int,
    table_bbox: Dict[str, float] | None,
    output_dir: Path,
    table_index: int,
) -> Path | None:
    if not table_bbox:
        return None

    try:
        import fitz
    except ModuleNotFoundError:
        return None

    try:
        with fitz.open(str(file_path)) as pdf:
            page = pdf.load_page(page_index - 1)
            margin_x = max(12.0, page.rect.width * 0.01)
            margin_y = max(10.0, page.rect.height * 0.01)
            x0 = max(0.0, float(table_bbox.get("x0", 0.0)) - margin_x)
            y0 = max(0.0, float(table_bbox.get("y0", 0.0)) - margin_y)
            x1 = min(page.rect.width, float(table_bbox.get("x1", 0.0)) + margin_x)
            y1 = min(page.rect.height, float(table_bbox.get("y1", 0.0)) + margin_y)
            crop_rect = fitz.Rect(x0, y0, x1, y1)
            if crop_rect.is_empty or crop_rect.width < 40 or crop_rect.height < 24:
                return None
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=crop_rect, alpha=False)
            image_path = output_dir / f"page_{page_index}_table_{table_index}.png"
            pix.save(str(image_path))
            return image_path
    except Exception:
        return None


def _attach_caption_metadata(image_block: Dict, caption_block: Dict | None, *, caption_index: int | None = None, match_method: str = "") -> Dict:
    if not caption_block:
        return image_block
    image_block["caption_text"] = caption_block.get("content", "")
    image_block["caption_bbox"] = caption_block.get("bbox")
    image_block["caption_source"] = image_block.get("source", "")
    if caption_index is not None:
        image_block["matched_caption_index"] = caption_index
    if match_method:
        image_block["caption_match_method"] = match_method
    return image_block


def _find_nearby_text_block(caption_block: Dict | None, layout_blocks: List[Dict]) -> Dict | None:
    if not caption_block:
        return None

    caption_bbox = caption_block.get("bbox")
    best_match: Dict | None = None
    best_score: float | None = None
    for block in layout_blocks:
        if block.get("is_caption"):
            continue

        content = clean_text_artifacts(block.get("content", ""))
        if not content or len(content) < 8:
            continue

        candidate_bbox = block.get("bbox")
        horizontal_overlap = _bbox_horizontal_overlap_ratio(caption_bbox, candidate_bbox)
        vertical_gap = _bbox_vertical_gap(caption_bbox, candidate_bbox)
        if vertical_gap > 240:
            continue
        if horizontal_overlap < 0.12 and vertical_gap > 32:
            continue

        candidate_is_above = float(candidate_bbox.get("y1", 0.0)) <= float(caption_bbox.get("y0", 0.0))
        direction_penalty = 55.0 if candidate_is_above else 0.0
        overlap_penalty = max(0.0, 1.0 - horizontal_overlap) * 80.0
        score = vertical_gap + direction_penalty + overlap_penalty
        if best_score is None or score < best_score:
            best_match = block
            best_score = score

    return best_match


def _attach_nearby_text_metadata(image_block: Dict, nearby_text_block: Dict | None, *, match_method: str = "") -> Dict:
    if not nearby_text_block:
        return image_block
    image_block["nearby_text"] = nearby_text_block.get("content", "")
    image_block["nearby_text_bbox"] = nearby_text_block.get("bbox")
    image_block["nearby_text_source"] = nearby_text_block.get("source", "")
    image_block["nearby_text_block_index"] = nearby_text_block.get("text_block_index")
    if match_method:
        image_block["nearby_text_match_method"] = match_method
    return image_block


def _render_pdf_caption_crops(
    file_path: Path,
    page_index: int,
    output_dir: Path,
    caption_blocks: List[Dict] | None = None,
) -> List[Dict]:
    try:
        import fitz
    except ModuleNotFoundError:
        return []

    crop_entries: List[Dict] = []
    try:
        with fitz.open(str(file_path)) as pdf:
            page = pdf.load_page(page_index - 1)
            resolved_caption_blocks = caption_blocks or []
            if not resolved_caption_blocks:
                for block in page.get_text("blocks") or []:
                    if len(block) < 5:
                        continue
                    x0, y0, x1, y1, text = block[:5]
                    text = str(text or "").strip()
                    if _contains_figure_caption(text):
                        resolved_caption_blocks.append(
                            {
                                "content": clean_text_artifacts(text),
                                "bbox": _normalize_bbox((x0, y0, x1, y1)),
                                "is_caption": True,
                            }
                        )

            for caption_index, caption_block in enumerate(resolved_caption_blocks[:3], start=1):
                bbox = caption_block.get("bbox") or {}
                caption_bbox = (
                    float(bbox.get("x0", 0.0)),
                    float(bbox.get("y0", 0.0)),
                    float(bbox.get("x1", 0.0)),
                    float(bbox.get("y1", 0.0)),
                )
                crop_rect = fitz.Rect(*_caption_crop_bounds(page.rect.width, page.rect.height, caption_bbox))
                if crop_rect.is_empty or crop_rect.width < 80 or crop_rect.height < 80:
                    continue
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=crop_rect, alpha=False)
                image_path = output_dir / f"page_{page_index}_caption_crop_{caption_index}.png"
                pix.save(str(image_path))
                crop_entries.append(
                    {
                        "path": image_path,
                        "caption_index": caption_index,
                        "caption_block": caption_block,
                        "crop_bbox": _normalize_bbox((crop_rect.x0, crop_rect.y0, crop_rect.x1, crop_rect.y1)),
                    }
                )
    except Exception:
        return []

    return crop_entries


def parse_pdf(file_path: Path) -> Dict:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError:
        return {
            "file_name": file_path.name,
            "file_type": "pdf",
            "blocks": [
                {
                    "type": "text",
                    "page": None,
                    "content": "未安装 pypdf，暂无法解析 PDF 文本。",
                    "source": file_path.name,
                }
            ],
        }

    reader = PdfReader(str(file_path))
    blocks: List[Dict] = []
    extracted_dir = ensure_directory(file_path.parent / "_extracted" / file_path.stem)
    table_map = _extract_pdf_tables(file_path)
    fallback_text_map = _extract_pdf_text_fallback(file_path)
    layout_text_map = _extract_pdf_layout_text_blocks(file_path)

    for page_index, page in enumerate(reader.pages, start=1):
        raw_text = (page.extract_text() or "").strip()
        fallback_text = fallback_text_map.get(page_index, "")
        text = _merge_page_text(raw_text, fallback_text)
        layout_blocks = layout_text_map.get(page_index, [])
        page_tables = table_map.get(page_index, [])
        filtered_layout_blocks = _filter_layout_blocks_against_tables(layout_blocks, page_tables)
        indexed_layout_blocks: List[Dict] = []
        page_images = getattr(page, "images", [])
        scanned_page_ocr_text = ""
        scanned_page_snapshot_path: Path | None = None
        used_page_ocr_fallback = False
        if _looks_like_scanned_page(
            merged_text=text,
            layout_blocks=filtered_layout_blocks,
            page_images=page_images,
            raw_text=raw_text,
            fallback_text=fallback_text,
        ):
            scanned_page_ocr_text, scanned_page_snapshot_path = _extract_page_ocr_text(file_path, page_index, extracted_dir)
            if scanned_page_ocr_text and not is_low_quality_text(scanned_page_ocr_text):
                text = _merge_page_text(text, scanned_page_ocr_text)
                used_page_ocr_fallback = True
        if filtered_layout_blocks:
            for text_block_index, layout_text in enumerate(filtered_layout_blocks, start=1):
                indexed_layout_text = {
                    **layout_text,
                    "text_block_index": text_block_index,
                    "source": f"{file_path.name} 第 {page_index} 页 文本块 {text_block_index}",
                }
                indexed_layout_blocks.append(indexed_layout_text)
                blocks.append(
                    {
                        "type": "text",
                        "page": page_index,
                        "content": indexed_layout_text.get("content", ""),
                        "text_block_index": text_block_index,
                        "layout_bbox": indexed_layout_text.get("bbox"),
                        "is_caption": bool(indexed_layout_text.get("is_caption", False)),
                        "source": f"{file_path.name} 第 {page_index} 页 文本块 {text_block_index}",
                    }
                )
        elif text and not is_low_quality_text(text):
            blocks.append(
                {
                    "type": "text",
                    "page": page_index,
                    "content": text,
                    "ocr_text": scanned_page_ocr_text if used_page_ocr_fallback else "",
                    "ocr_fallback_used": used_page_ocr_fallback,
                    "ocr_source_image": str(scanned_page_snapshot_path) if scanned_page_snapshot_path and used_page_ocr_fallback else "",
                    "source": f"{file_path.name} 第 {page_index} 页",
                }
            )

        for table_index, table_entry in enumerate(page_tables, start=1):
            table_image_path = _render_pdf_table_crop(
                file_path=file_path,
                page_index=page_index,
                table_bbox=table_entry.get("bbox"),
                output_dir=extracted_dir,
                table_index=table_index,
            )
            blocks.append(
                {
                    "type": "table",
                    "page": page_index,
                    "table_index": table_index,
                    "content": table_entry.get("content", ""),
                    "table_bbox": table_entry.get("bbox"),
                    "table_image_path": str(table_image_path) if table_image_path else "",
                    "source": f"{file_path.name} 第 {page_index} 页 表格 {table_index}",
                }
            )

        caption_blocks = _extract_caption_blocks(indexed_layout_blocks or layout_blocks)
        valid_image_count = 0
        for image_index, image in enumerate(page_images, start=1):
            image_name = getattr(image, "name", f"page_{page_index}_image_{image_index}.png")
            image_bytes = getattr(image, "data", b"")
            if not image_bytes:
                continue

            image_path = extracted_dir / f"page_{page_index}_{image_index}_{Path(image_name).name}"
            image_path.write_bytes(image_bytes)
            image_block = build_image_block(
                file_path=image_path,
                source=f"{file_path.name} 第 {page_index} 页 图片 {image_index}",
                page=page_index,
                image_index=image_index,
            )
            if _is_noise_image(image_block):
                continue
            if image_index <= len(caption_blocks):
                image_block = _attach_caption_metadata(
                    image_block,
                    caption_blocks[image_index - 1],
                    caption_index=image_index,
                    match_method="page_order",
                )
                image_block = _attach_nearby_text_metadata(
                    image_block,
                    _find_nearby_text_block(caption_blocks[image_index - 1], indexed_layout_blocks),
                    match_method="caption_neighbor",
                )
            valid_image_count += 1
            blocks.append(image_block)

        needs_page_snapshot = _contains_figure_caption(text or raw_text or fallback_text) and valid_image_count == 0
        if needs_page_snapshot:
            crop_entries = _render_pdf_caption_crops(file_path, page_index, extracted_dir, caption_blocks=caption_blocks)
            for crop_index, crop_entry in enumerate(crop_entries, start=1):
                crop_block = build_image_block(
                    file_path=crop_entry["path"],
                    source=f"{file_path.name} 第 {page_index} 页 图注区域裁切 {crop_index}",
                    page=page_index,
                    image_index=crop_index,
                )
                crop_block["crop_bbox"] = crop_entry.get("crop_bbox")
                crop_block = _attach_caption_metadata(
                    crop_block,
                    crop_entry.get("caption_block"),
                    caption_index=crop_entry.get("caption_index"),
                    match_method="caption_crop",
                )
                crop_block = _attach_nearby_text_metadata(
                    crop_block,
                    _find_nearby_text_block(crop_entry.get("caption_block"), indexed_layout_blocks),
                    match_method="caption_neighbor",
                )
                blocks.append(crop_block)
            if crop_entries:
                continue
            snapshot_path = _render_pdf_page(file_path, page_index, extracted_dir)
            if snapshot_path is not None:
                snapshot_block = build_image_block(
                    file_path=snapshot_path,
                    source=f"{file_path.name} 第 {page_index} 页 页面图示 1",
                    page=page_index,
                    image_index=1,
                )
                if caption_blocks:
                    snapshot_block = _attach_caption_metadata(
                        snapshot_block,
                        caption_blocks[0],
                        caption_index=1,
                        match_method="page_snapshot",
                    )
                    snapshot_block = _attach_nearby_text_metadata(
                        snapshot_block,
                        _find_nearby_text_block(caption_blocks[0], indexed_layout_blocks),
                        match_method="caption_neighbor",
                    )
                blocks.append(snapshot_block)

    return {
        "file_name": file_path.name,
        "file_type": "pdf",
        "blocks": blocks,
    }
