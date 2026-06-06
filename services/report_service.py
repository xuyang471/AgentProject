from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Iterable, List

from .file_service import ensure_directory
from .llm_service import extract_report_insights, summarize_texts

BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_ASSETS_DIR = BASE_DIR / "output" / "report_assets"


def _collect_text_blocks(parsed_documents: Iterable[dict]) -> List[str]:
    texts: List[str] = []
    for document in parsed_documents:
        for block in document["blocks"]:
            if block["type"] in {"text", "image", "table"} and block["content"].strip():
                if block["type"] == "table":
                    texts.append(_table_to_plain_text(block["content"]))
                else:
                    texts.append(block["content"].strip())
    return texts


def _collect_image_blocks(parsed_documents: Iterable[dict]) -> List[dict]:
    return [
        block
        for document in parsed_documents
        for block in document["blocks"]
        if block["type"] == "image"
    ]


def _collect_table_blocks(parsed_documents: Iterable[dict]) -> List[dict]:
    return [
        block
        for document in parsed_documents
        for block in document["blocks"]
        if block["type"] == "table"
    ]


def _build_markdown_image(image_path: str) -> str:
    if not image_path:
        return ""
    path = Path(image_path)
    if not path.exists():
        return ""
    return f"![检测到的图片]({path.resolve()})"


def _sanitize_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\u4e00-\u9fff-]+", "_", value).strip("_") or "table_preview"


def _is_markdown_separator(cells: List[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)


def _parse_markdown_table(content: str) -> List[List[str]]:
    rows: List[List[str]] = []
    for line in str(content).splitlines():
        stripped = line.strip()
        if not stripped or "|" not in stripped:
            continue

        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if _is_markdown_separator(cells):
            continue
        if any(cells):
            rows.append(cells)

    if not rows:
        return []

    max_columns = max(len(row) for row in rows)
    return [row + [""] * (max_columns - len(row)) for row in rows]


def _wrap_cell_text(text: str, max_chars: int = 16) -> List[str]:
    compact = re.sub(r"\s+", " ", str(text)).strip()
    if not compact:
        return [""]
    return [compact[index : index + max_chars] for index in range(0, len(compact), max_chars)] or [""]


def _table_to_plain_text(content: str) -> str:
    rows = _parse_markdown_table(content)
    if not rows:
        return str(content).replace("|", " ").strip()

    header = rows[0]
    body = rows[1:] or rows[:1]
    entries: List[str] = []
    for row in body[:8]:
        pairs = [
            f"{header[index]}：{cell}" if index < len(header) and header[index] else cell
            for index, cell in enumerate(row)
            if str(cell).strip()
        ]
        if pairs:
            entries.append("；".join(pairs))
    return "表格内容：" + "。".join(entries)


def _render_table_preview(block: dict) -> str:
    table_image_path = str(block.get("table_image_path", "") or "").strip()
    if table_image_path:
        existing = Path(table_image_path)
        if existing.exists():
            return f"![table-preview]({existing.resolve()})"

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError:
        return ""

    content = str(block.get("content", "")).strip()
    if not content:
        return ""

    ensure_directory(REPORT_ASSETS_DIR)
    page = block.get("page", "x")
    table_index = block.get("table_index", "x")
    source = _sanitize_filename(str(block.get("source", "")))
    output_path = REPORT_ASSETS_DIR / f"table_p{page}_{table_index}_{source}.png"

    if output_path.exists():
        return f"![表格预览]({output_path.resolve()})"

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return ""

    try:
        font = ImageFont.truetype("simhei.ttf", 18)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            font = ImageFont.load_default()

    line_height = 28
    padding = 20
    max_chars = max(len(line) for line in lines[:12])
    width = min(max(700, max_chars * 12), 1800)
    height = min(padding * 2 + line_height * min(len(lines), 14), 900)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([(0, 0), (width - 1, height - 1)], outline="#AAB7C4", width=2)

    y = padding
    for index, line in enumerate(lines[:12]):
        fill = "#EAF3FF" if index in {0, 1} else "white"
        draw.rectangle([(10, y - 4), (width - 10, y + line_height - 2)], fill=fill)
        draw.text((20, y), line[:200], font=font, fill="black")
        y += line_height

    if len(lines) > 12:
        draw.text((20, y), "...", font=font, fill="#555555")

    image.save(output_path)
    return f"![表格预览]({output_path.resolve()})"


def _build_markdown_image(image_path: str) -> str:
    """Build a valid Markdown image reference for local report assets."""
    if not image_path:
        return ""
    path = Path(image_path)
    if not path.exists():
        return ""
    return f"![检测到的图片]({path.resolve()})"


def _extract_key_numbers(texts: Iterable[str]) -> List[str]:
    found: List[str] = []
    pattern = re.compile(
        r"\b\d+(?:\.\d+)?(?:ms|s|秒|分钟|小时|天|%|w|kw|mw|gb|mb|tb|人|个|门|次|年|月|日|周|v|a)?\b",
        re.IGNORECASE,
    )
    for text in texts:
        matches = pattern.findall(text.replace("：", " ").replace("，", " "))
        for token in matches:
            cleaned = token.strip()
            if not cleaned:
                continue
            if cleaned.isdigit() and len(cleaned) <= 2:
                continue
            found.append(cleaned)
        if len(found) >= 5:
            break
    deduped: List[str] = []
    for item in found:
        if item not in deduped:
            deduped.append(item)
    return deduped[:5]


def _fallback_theme(texts: List[str], image_blocks: List[dict]) -> str:
    merged = " ".join(texts[:4]).strip()
    if not merged and image_blocks:
        merged = " ".join(block.get("description", "") for block in image_blocks[:2]).strip()
    if not merged:
        return "信息不足"
    return merged[:24].strip("，。；： ")


def _fallback_conclusions(texts: List[str], image_blocks: List[dict]) -> List[str]:
    conclusions: List[str] = []
    for text in texts:
        sentence = text.strip().replace("\n", " ")
        if len(sentence) < 12:
            continue
        conclusions.append(sentence[:80].rstrip("，。；： ") + "。")
        if len(conclusions) >= 2:
            break

    if not conclusions and image_blocks:
        for block in image_blocks[:2]:
            description = block.get("description", "").strip()
            if description:
                conclusions.append(description[:80].rstrip("，。；： ") + "。")
    return conclusions[:2]


def _fallback_open_questions(parsed_documents: List[dict], texts: List[str]) -> List[str]:
    open_questions: List[str] = []
    if not texts:
        open_questions.append("当前文档可提取的文本内容较少，部分结论可能依赖图片说明。")
    if any(document.get("file_type") == "pdf" for document in parsed_documents):
        open_questions.append("若原始 PDF 含扫描页或未被正确抽取的版面，仍可能存在信息遗漏。")
    return open_questions[:2]


def _build_report_insights(parsed_documents: List[dict], texts: List[str], image_blocks: List[dict]) -> dict:
    llm_result = extract_report_insights(texts)
    if llm_result:
        return {
            "summary": str(llm_result.get("summary", "")).strip() or summarize_texts(texts),
            "theme": str(llm_result.get("theme", "")).strip() or _fallback_theme(texts, image_blocks),
            "main_conclusions": [
                str(item).strip()
                for item in llm_result.get("main_conclusions", [])
                if str(item).strip()
            ][:3]
            or _fallback_conclusions(texts, image_blocks),
            "indicators": [
                str(item).strip()
                for item in llm_result.get("indicators", [])
                if str(item).strip()
            ][:5]
            or _extract_key_numbers(texts),
            "open_questions": [
                str(item).strip()
                for item in llm_result.get("open_questions", [])
                if str(item).strip()
            ][:3]
            or _fallback_open_questions(parsed_documents, texts),
        }

    return {
        "summary": summarize_texts(texts),
        "theme": _fallback_theme(texts, image_blocks),
        "main_conclusions": _fallback_conclusions(texts, image_blocks),
        "indicators": _extract_key_numbers(texts),
        "open_questions": _fallback_open_questions(parsed_documents, texts),
    }


def _build_markdown_image(image_path: str) -> str:
    """Build a valid Markdown image reference for local report assets."""
    if not image_path:
        return ""
    path = Path(image_path)
    if not path.exists():
        return ""
    return f"![detected-image]({path.resolve()})"


def _render_table_preview(block: dict) -> str:
    """Render a parsed Markdown table as a real grid image for report preview."""
    table_image_path = str(block.get("table_image_path", "") or "").strip()
    if table_image_path:
        existing = Path(table_image_path)
        if existing.exists():
            return f"![table-preview]({existing.resolve()})"

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError:
        return ""

    rows = _parse_markdown_table(str(block.get("content", "")))
    if not rows:
        return ""

    ensure_directory(REPORT_ASSETS_DIR)
    page = block.get("page", "x")
    table_index = block.get("table_index", "x")
    source = _sanitize_filename(str(block.get("source", "")))
    output_path = REPORT_ASSETS_DIR / f"table_grid_p{page}_{table_index}_{source}.png"

    if output_path.exists():
        return f"![table-preview]({output_path.resolve()})"

    rows = [row[:6] for row in rows[:12]]
    column_count = max(len(row) for row in rows)
    rows = [row + [""] * (column_count - len(row)) for row in rows]

    try:
        font = ImageFont.truetype("simhei.ttf", 18)
        header_font = ImageFont.truetype("simhei.ttf", 18)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            header_font = ImageFont.truetype("arialbd.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
            header_font = font

    padding_x = 14
    padding_y = 10
    line_height = 24
    max_chars_per_cell = 18

    wrapped_rows = [
        [_wrap_cell_text(cell, max_chars=max_chars_per_cell) for cell in row]
        for row in rows
    ]

    column_widths: List[int] = []
    for column_index in range(column_count):
        longest = max(
            len(line)
            for row in wrapped_rows
            for line in row[column_index]
        )
        column_widths.append(min(max(120, longest * 11 + padding_x * 2), 280))

    row_heights = [
        max(line_height * max(len(cell_lines), 1) + padding_y * 2 for cell_lines in row)
        for row in wrapped_rows
    ]

    width = sum(column_widths) + 1
    height = sum(row_heights) + 1
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    y = 0
    for row_index, row in enumerate(wrapped_rows):
        x = 0
        row_fill = "#EAF3FF" if row_index == 0 else "#FFFFFF"
        text_fill = "#111827"
        active_font = header_font if row_index == 0 else font
        for column_index, cell_lines in enumerate(row):
            cell_width = column_widths[column_index]
            cell_height = row_heights[row_index]
            draw.rectangle(
                [(x, y), (x + cell_width, y + cell_height)],
                fill=row_fill,
                outline="#AAB7C4",
                width=1,
            )
            text_y = y + padding_y
            for line in cell_lines:
                draw.text((x + padding_x, text_y), line, font=active_font, fill=text_fill)
                text_y += line_height
            x += cell_width
        y += row_heights[row_index]

    image.save(output_path)
    return f"![table-preview]({output_path.resolve()})"


def build_markdown_report(parsed_documents: List[dict]) -> str:
    texts = _collect_text_blocks(parsed_documents)
    image_blocks = _collect_image_blocks(parsed_documents)
    table_blocks = _collect_table_blocks(parsed_documents)
    insights = _build_report_insights(parsed_documents, texts, image_blocks)
    summary = insights["summary"]
    theme = insights["theme"]
    conclusions = insights["main_conclusions"]
    key_numbers = insights["indicators"]
    open_questions = insights["open_questions"]

    report_lines = [
        "# 文档总结报告",
        "## 1. 核心摘要",
        summary,
        "## 2. 关键信息提取",
        f"- 主题/领域：{theme}",
        f"- 主要结论：{'；'.join(conclusions) if conclusions else '信息不足'}",
        f"- 数据/指标：{'、'.join(key_numbers) if key_numbers else '未提取到明确业务指标'}",
        f"- 表格信息：共识别 {len(table_blocks)} 个表格内容块" if table_blocks else "- 表格信息：未识别到明确表格",
        "## 3. 图表/图片说明",
    ]

    if image_blocks:
        for index, block in enumerate(image_blocks, start=1):
            description = block.get("description", "").strip() or "未提取到清晰图片说明。"
            source = block.get("source", "")
            image_ref = _build_markdown_image(block.get("image_path", ""))
            report_lines.append(f"- 图片{index}：{description}" + (f" 来源：{source}" if source else ""))
            if image_ref:
                report_lines.append(image_ref)
    else:
        report_lines.append("- 未检测到独立图片说明内容")

    if table_blocks:
        report_lines.append("## 4. 表格内容摘要")
        for index, block in enumerate(table_blocks[:5], start=1):
            parsed_table = _parse_markdown_table(block.get("content", ""))
            row_count = max(len(parsed_table) - 1, 0) if parsed_table else 0
            column_count = len(parsed_table[0]) if parsed_table else 0
            compact = f"识别到 {row_count} 行、{column_count} 列的表格内容，详见下方表格预览。"
            source = block.get("source", "")
            report_lines.append(f"- 表格{index}：{compact}" + (f" 来源：{source}" if source else ""))
            table_preview = _render_table_preview(block)
            if table_preview:
                report_lines.append(table_preview)
        report_lines.append("## 5. 待确认或疑问点")
    else:
        report_lines.append("## 4. 待确认或疑问点")

    if open_questions:
        for item in open_questions:
            report_lines.append(f"- {item}")
    else:
        report_lines.append("- 暂未发现明显待确认项。")

    return "\n".join(report_lines)


def save_report(markdown_text: str, target_dir: Path) -> Path:
    ensure_directory(target_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = target_dir / f"report_{timestamp}.md"
    report_path.write_text(markdown_text, encoding="utf-8")
    return report_path
