from __future__ import annotations

from datetime import datetime
import io
from pathlib import Path
import re
from typing import List

import streamlit as st

from services.file_service import save_uploaded_files
from services.logging_service import configure_runtime_logging
from services.parser_service import parse_files
from services.qa_service import answer_question
from services.report_service import build_markdown_report, save_report
from services.state_service import (
    create_analysis_session,
    get_latest_session,
    get_qa_history,
    list_sessions,
    load_session_context,
    register_uploaded_files,
    save_parsed_documents,
    save_report_record,
)


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
REPORTS_DIR = OUTPUT_DIR / "reports"
TEMP_DIR = OUTPUT_DIR / "temp"

configure_runtime_logging()


def _default_state() -> None:
    st.session_state.setdefault("current_session_id", "")
    st.session_state.setdefault("parsed_documents", [])
    st.session_state.setdefault("report_markdown", "")
    st.session_state.setdefault("report_path", "")
    st.session_state.setdefault("current_report_id", "")
    st.session_state.setdefault("history_saved", False)
    st.session_state.setdefault("pending_session_name", "")
    st.session_state.setdefault("pending_saved_paths", [])
    st.session_state.setdefault("developer_mode", False)
    st.session_state.setdefault("qa_history", [])
    st.session_state.setdefault("question_input", "")
    st.session_state.setdefault("source_focus", {})
    st.session_state.setdefault("follow_up_mode", True)
    st.session_state.setdefault("source_focus_index", 0)


def _load_session_into_state(session_id: str) -> bool:
    context = load_session_context(session_id)
    if not context:
        return False

    report = context.get("report") or {}
    st.session_state["current_session_id"] = session_id
    st.session_state["parsed_documents"] = context.get("documents", [])
    st.session_state["report_markdown"] = report.get("content", "")
    st.session_state["report_path"] = report.get("report_path", "")
    st.session_state["current_report_id"] = report.get("id", "")
    st.session_state["history_saved"] = True
    st.session_state["pending_session_name"] = ""
    st.session_state["pending_saved_paths"] = []
    st.session_state["qa_history"] = get_qa_history(session_id, limit=8)
    st.session_state["question_input"] = ""
    st.session_state["source_focus"] = {}
    st.session_state["source_focus_index"] = 0
    return True


def init_state() -> None:
    _default_state()


def validate_uploads(files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> str | None:
    if len(files) > 5:
        return "单次最多上传 5 个文件。"

    total_size = sum(file.size for file in files)
    if total_size > 20 * 1024 * 1024:
        return "上传文件总大小不能超过 20MB。"

    return None


def _render_markdown_with_local_images(markdown_text: str) -> None:
    image_pattern = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")
    buffer: list[str] = []
    table_buffer: list[str] = []

    def flush_markdown() -> None:
        if buffer:
            st.markdown("\n".join(buffer))
            buffer.clear()

    def flush_table() -> None:
        if not table_buffer:
            return
        table_text = "\n".join(table_buffer)
        if not _render_table_like_text(table_text):
            st.markdown(table_text)
        table_buffer.clear()

    for line in markdown_text.splitlines():
        match = image_pattern.fullmatch(line.strip())
        if not match:
            stripped = line.strip()
            is_table_line = "|" in stripped and stripped.count("|") >= 2
            if is_table_line:
                flush_markdown()
                table_buffer.append(line)
            else:
                flush_table()
                buffer.append(line)
            continue

        flush_table()
        flush_markdown()
        raw_path = match.group("path").strip().strip('"')
        image_path = Path(raw_path)
        if image_path.exists():
            st.image(str(image_path), caption=match.group("alt") or None, width="stretch")
        else:
            st.warning(f"图片文件不存在，无法预览：{raw_path}")

    flush_table()
    flush_markdown()


def _split_react_trace(markdown_text: str) -> tuple[str, str]:
    marker = "### ReAct 轨迹"
    if marker not in markdown_text:
        return markdown_text, ""
    main_content, react_section = markdown_text.split(marker, 1)
    return main_content.rstrip(), f"{marker}{react_section}".strip()


def _build_session_name(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> str:
    if not uploaded_files:
        return f"analysis-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    first_name = Path(uploaded_files[0].name).stem
    return f"{first_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _session_temp_dir(session_id: str) -> Path:
    return TEMP_DIR / session_id / "uploads"


def _session_report_dir(session_id: str) -> Path:
    return REPORTS_DIR / session_id


def _unsaved_run_id() -> str:
    return f"unsaved-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"


def _clear_current_analysis_state() -> None:
    st.session_state["current_session_id"] = ""
    st.session_state["parsed_documents"] = []
    st.session_state["report_markdown"] = ""
    st.session_state["report_path"] = ""
    st.session_state["current_report_id"] = ""
    st.session_state["history_saved"] = False
    st.session_state["pending_session_name"] = ""
    st.session_state["pending_saved_paths"] = []
    st.session_state["qa_history"] = []
    st.session_state["question_input"] = ""
    st.session_state["source_focus"] = {}
    st.session_state["source_focus_index"] = 0


def _inject_page_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }
        .hero-card {
            border: 1px solid rgba(120, 120, 120, 0.18);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            background: linear-gradient(135deg, rgba(245,248,252,0.95), rgba(255,255,255,0.98));
            margin-bottom: 1rem;
        }
        .section-hint {
            color: #5b6573;
            font-size: 0.95rem;
            margin-top: 0.35rem;
            margin-bottom: 0.1rem;
        }
        .source-chip {
            display: inline-block;
            padding: 0.15rem 0.55rem;
            border-radius: 999px;
            background: #eef3f8;
            color: #324256;
            font-size: 0.8rem;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }
        .source-card {
            border: 1px solid rgba(120, 120, 120, 0.16);
            border-radius: 14px;
            padding: 0.85rem 0.95rem 0.7rem 0.95rem;
            background: #fbfcfe;
            margin-bottom: 0.75rem;
        }
        .source-title {
            font-weight: 600;
            color: #223449;
            margin-bottom: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def clean_preview_text(text: str, limit: int = 140) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit].rstrip()}..."


def _linkify_text(text: str) -> str:
    return re.sub(r"(https?://[^\s)]+)", r"[\1](\1)", text)


def _collect_block_metrics(parsed_documents: List[dict]) -> dict:
    metrics = {"files": len(parsed_documents), "blocks": 0, "text": 0, "image": 0, "table": 0}
    for document in parsed_documents:
        for block in document.get("blocks", []):
            metrics["blocks"] += 1
            block_type = block.get("type", "text")
            metrics[block_type] = metrics.get(block_type, 0) + 1
    return metrics


def _render_overview_metrics(parsed_documents: List[dict]) -> None:
    metrics = _collect_block_metrics(parsed_documents)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("文件数", metrics["files"])
    col2.metric("内容块", metrics["blocks"])
    col3.metric("文本块", metrics["text"])
    col4.metric("图片块", metrics["image"])
    col5.metric("表格块", metrics["table"])


def _build_question_suggestions(parsed_documents: List[dict]) -> List[str]:
    has_image = any(block.get("type") == "image" for doc in parsed_documents for block in doc.get("blocks", []))
    has_table = any(block.get("type") == "table" for doc in parsed_documents for block in doc.get("blocks", []))
    suggestions = [
        "这份文档的主旨是什么？",
        "请提取文档中的关键结论和关键数据。",
    ]
    if has_image:
        suggestions.append("图 1 或图片 1 主要表达了什么？")
    if has_table:
        suggestions.append("表格里最重要的结论或指标是什么？")
    suggestions.append("有哪些值得进一步确认的疑问点？")
    return suggestions[:4]


def _render_parsed_document_overview(parsed_documents: List[dict]) -> None:
    for document in parsed_documents:
        block_count = len(document.get("blocks", []))
        with st.expander(f"{document['file_name']} · {block_count} 个内容块", expanded=False):
            block_types = {}
            for block in document.get("blocks", []):
                block_type = block.get("type", "text")
                block_types[block_type] = block_types.get(block_type, 0) + 1
            if block_types:
                chips = "".join(
                    f"<span class='source-chip'>{block_type}: {count}</span>"
                    for block_type, count in sorted(block_types.items())
                )
                st.markdown(chips, unsafe_allow_html=True)
            for block in document.get("blocks", [])[:6]:
                source = block.get("source", document["file_name"])
                content = clean_preview_text(block.get("content", ""))
                st.markdown(f"- `{block.get('type', 'text')}` · `{source}`")
                if block.get("type") == "table":
                    if not _render_table_like_text(block.get("content", "")):
                        st.caption(content)
                elif content:
                    st.caption(content)


def _append_qa_history(question: str, answer: str) -> None:
    history = st.session_state.get("qa_history", [])
    history.insert(
        0,
        {
            "question": question,
            "answer": answer,
            "created_at": datetime.now().strftime("%H:%M:%S"),
        },
    )
    st.session_state["qa_history"] = history[:8]


def _split_sources_section(markdown_text: str) -> tuple[str, str]:
    markers = ["### 检索来源", "### 妫€绱㈡潵婧?"]
    for marker in markers:
        if marker in markdown_text:
            main_content, source_section = markdown_text.split(marker, 1)
            return main_content.rstrip(), f"{marker}{source_section}".strip()
    return markdown_text, ""


def _parse_source_cards(source_section: str) -> List[dict]:
    if not source_section:
        return []

    normalized = source_section.strip()
    if normalized.startswith("### "):
        first_newline = normalized.find("\n")
        normalized = normalized[first_newline + 1 :] if first_newline != -1 else ""

    chunks = re.split(r"(?=^####\s+)", normalized, flags=re.MULTILINE)
    cards: List[dict] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        lines = [line.rstrip() for line in chunk.splitlines() if line.strip()]
        if not lines:
            continue

        title_line = lines[0]
        title = re.sub(r"^####\s*", "", title_line).strip()
        meta_lines: List[str] = []
        body_lines: List[str] = []
        image_lines: List[str] = []

        for line in lines[1:]:
            stripped = line.strip()
            if stripped.startswith("- "):
                meta_lines.append(stripped[2:].strip())
            elif stripped.startswith("!["):
                image_lines.append(stripped)
            else:
                body_lines.append(stripped)

        cards.append(
            {
                "title": title,
                "meta_lines": meta_lines,
                "body_lines": body_lines,
                "image_lines": image_lines,
            }
        )

    return cards


def _extract_page_number(meta_lines: List[str]) -> int | None:
    for meta_line in meta_lines:
        match = re.search(r"第\s*(\d+)\s*页", meta_line)
        if match:
            return int(match.group(1))
    return None


def _extract_source_location(meta_lines: List[str]) -> str:
    for meta_line in meta_lines:
        if "位置" in meta_line:
            location = meta_line.split("：", 1)[-1].strip()
            location = re.sub(r"（第\s*\d+\s*页）", "", location).strip()
            return location
    return ""


def _extract_block_type(meta_lines: List[str]) -> str:
    for meta_line in meta_lines:
        if "类型" in meta_line:
            return meta_line.split("：", 1)[-1].strip().lower()
    return ""


def _markdown_table_to_html(table_text: str) -> str:
    lines = [line.strip() for line in table_text.splitlines() if line.strip()]
    if len(lines) < 2:
        return ""
    if "|" not in lines[0] or "|" not in lines[1]:
        return ""

    def parse_row(row: str) -> List[str]:
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        return [cell for cell in cells]

    header = parse_row(lines[0])
    separator = parse_row(lines[1])
    if not header or not separator or not all(set(cell) <= {"-", ":"} for cell in separator if cell):
        return ""

    body_rows = [parse_row(line) for line in lines[2:]]
    header_html = "".join(
        "<th style='padding:0.55rem 0.65rem; text-align:left; background:#eef3f8; "
        "border-bottom:1px solid #d9e2ec;'>" + cell + "</th>"
        for cell in header
    )
    body_html = "".join(
        "<tr>"
        + "".join(
            "<td style='padding:0.5rem 0.65rem; border-bottom:1px solid #eef3f8; "
            "vertical-align:top;'>" + cell + "</td>"
            for cell in row
        )
        + "</tr>"
        for row in body_rows
        if any(cell for cell in row)
    )
    return (
        "<table style='width:100%; border-collapse:collapse; margin:0.35rem 0 0.6rem 0; "
        "border:1px solid #d9e2ec; border-radius:10px; overflow:hidden;'>"
        f"<thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table>"
    )


def _render_table_like_text(text: str) -> bool:
    html = _markdown_table_to_html(text)
    if not html:
        return False
    st.markdown(
        """
        <div style="font-size:0.86rem; color:#607184; margin:0.1rem 0 0.35rem 0;">
        HTML 表格预览
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(html, unsafe_allow_html=True)
    return True


def _resolve_focus_bbox(block: dict) -> dict | None:
    for key in ("layout_bbox", "table_bbox", "crop_bbox", "caption_bbox", "nearby_text_bbox"):
        bbox = block.get(key)
        if isinstance(bbox, dict) and {"x0", "y0", "x1", "y1"}.issubset(bbox.keys()):
            return bbox
    return None


def _render_pdf_highlight_preview(file_path: str, page_number: int, bbox: dict | None) -> bool:
    if not file_path or not page_number:
        return False

    pdf_path = Path(file_path)
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        return False

    try:
        import fitz
        from PIL import Image, ImageDraw
    except ModuleNotFoundError:
        return False

    try:
        with fitz.open(str(pdf_path)) as pdf:
            if page_number < 1 or page_number > len(pdf):
                return False
            page = pdf.load_page(page_number - 1)
            matrix = fitz.Matrix(1.6, 1.6)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            if bbox:
                scale_x = image.width / max(page.rect.width, 1)
                scale_y = image.height / max(page.rect.height, 1)
                rect = (
                    float(bbox.get("x0", 0.0)) * scale_x,
                    float(bbox.get("y0", 0.0)) * scale_y,
                    float(bbox.get("x1", 0.0)) * scale_x,
                    float(bbox.get("y1", 0.0)) * scale_y,
                )
                overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
                draw = ImageDraw.Draw(overlay)
                draw.rectangle(rect, outline=(220, 56, 44, 255), width=5, fill=(255, 225, 120, 70))
                image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
            st.image(image, caption=f"PDF 第 {page_number} 页定位预览", width="stretch")
            return True
    except Exception:
        return False


def _find_blocks_for_source(source_location: str, page: int | None, parsed_documents: List[dict], block_type: str = "") -> List[dict]:
    matches: List[dict] = []
    normalized_block_type = block_type.replace("图片", "image").replace("文本", "text").replace("表格", "table")
    for document in parsed_documents:
        for block in document.get("blocks", []):
            if source_location and block.get("source") != source_location:
                continue
            if page and block.get("page") != page:
                continue
            if normalized_block_type and normalized_block_type in {"image", "text", "table"}:
                if block.get("type") != normalized_block_type:
                    continue
            matches.append({"document": document, "block": block})
    if matches:
        return matches

    for document in parsed_documents:
        for block in document.get("blocks", []):
            if source_location and source_location in str(block.get("source", "")):
                if page and block.get("page") != page:
                    continue
                matches.append({"document": document, "block": block})
    return matches


def _render_source_preview(document: dict, block: dict) -> None:
    table_image_path = block.get("table_image_path", "")
    image_path = block.get("image_path", "")
    file_path = document.get("file_path", "")
    page = block.get("page")
    bbox = _resolve_focus_bbox(block)

    rendered_pdf_preview = False
    if file_path and page:
        rendered_pdf_preview = _render_pdf_highlight_preview(file_path, int(page), bbox)

    if rendered_pdf_preview:
        return
    if table_image_path and Path(table_image_path).exists():
        st.image(table_image_path, caption="表格局部预览", width="stretch")
        return
    if image_path and Path(image_path).exists():
        st.image(image_path, caption=block.get("caption_text") or "图片局部预览", width="stretch")


def _build_follow_up_question(question: str) -> str:
    if not st.session_state.get("follow_up_mode", True):
        return question
    history = st.session_state.get("qa_history", [])
    if not history:
        return question

    context_items = history[:2]
    context_lines = []
    for item in reversed(context_items):
        context_lines.append(f"上一轮问题：{item['question']}")
        context_lines.append(f"上一轮回答摘要：{clean_preview_text(item['answer'], 220)}")
    context_lines.append(f"当前问题：{question}")
    context_lines.append("请结合以上上下文，重点回答当前问题。")
    return "\n".join(context_lines)


def _render_source_focus_panel() -> None:
    focus = st.session_state.get("source_focus") or {}
    if not focus:
        return

    parsed_documents = st.session_state.get("parsed_documents", [])
    matches = _find_blocks_for_source(
        source_location=focus.get("source_location", ""),
        page=focus.get("page"),
        parsed_documents=parsed_documents,
        block_type=focus.get("block_type", ""),
    )
    if not matches:
        return

    st.markdown("### 来源定位面板")
    st.caption("这里会展示你刚刚点击的来源对应内容，方便快速核对页码、局部图片和表格原文。")
    total_matches = min(len(matches), 4)
    focus_index = int(st.session_state.get("source_focus_index", 0))
    focus_index = max(0, min(focus_index, total_matches - 1))
    st.session_state["source_focus_index"] = focus_index

    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 1, 2])
    with nav_col1:
        if st.button("关闭定位", key="close_source_focus", width="stretch"):
            st.session_state["source_focus"] = {}
            st.session_state["source_focus_index"] = 0
            st.rerun()
    with nav_col2:
        if st.button("上一条", key="source_focus_prev", width="stretch", disabled=focus_index <= 0):
            st.session_state["source_focus_index"] = max(0, focus_index - 1)
            st.rerun()
    with nav_col3:
        if st.button("下一条", key="source_focus_next", width="stretch", disabled=focus_index >= total_matches - 1):
            st.session_state["source_focus_index"] = min(total_matches - 1, focus_index + 1)
            st.rerun()
    with nav_col4:
        st.caption(f"当前定位：{focus_index + 1} / {total_matches}")

    item = matches[focus_index]
    document = item["document"]
    block = item["block"]
    with st.container(border=True):
        st.markdown(f"**文件**：`{document.get('file_name', '')}`")
        meta_bits = [
            f"页码：{block.get('page')}" if block.get("page") else "",
            f"类型：{block.get('type', '')}",
            f"来源：{block.get('source', '')}",
        ]
        st.caption(" | ".join(bit for bit in meta_bits if bit))
        _render_source_preview(document, block)

        table_rendered = False
        if block.get("type") == "table":
            table_rendered = _render_table_like_text(block.get("content", ""))
        if not table_rendered:
            st.markdown(_linkify_text(clean_preview_text(block.get("content", ""), 1200)))

        if block.get("caption_text"):
            st.caption(f"图注：{block.get('caption_text')}")
        if block.get("nearby_text"):
            st.caption(f"邻近正文：{clean_preview_text(block.get('nearby_text'), 220)}")

        summary_lines = [
            f"文件：{document.get('file_name', '')}",
            f"页码：{block.get('page')}" if block.get("page") else "页码：未知",
            f"类型：{block.get('type', '')}",
            f"来源：{block.get('source', '')}",
            f"内容摘要：{clean_preview_text(block.get('content', ''), 260)}",
        ]
        if block.get("caption_text"):
            summary_lines.append(f"图注：{block.get('caption_text')}")
        st.text_area(
            "可复制来源摘要",
            value="\n".join(summary_lines),
            height=140,
            key=f"source_summary_{focus_index}_{block.get('source', '')}",
        )

def _render_source_cards(source_section: str) -> None:
    cards = _parse_source_cards(source_section)
    if not cards:
        if source_section.strip():
            st.markdown(source_section)
        return

    st.markdown("### 检索来源")
    for card_index, card in enumerate(cards, start=1):
        source_location = _extract_source_location(card["meta_lines"])
        page = _extract_page_number(card["meta_lines"])
        block_type = _extract_block_type(card["meta_lines"])

        st.markdown(f"<div class='source-card'><div class='source-title'>{card['title']}</div></div>", unsafe_allow_html=True)
        action_col, meta_col = st.columns([0.28, 0.72])
        with action_col:
            button_label = f"跳到第 {page} 页" if page else "定位来源"
            if st.button(button_label, key=f"source_focus_{card_index}_{source_location}_{page}", width="stretch"):
                st.session_state["source_focus"] = {
                    "source_location": source_location,
                    "page": page,
                    "block_type": block_type,
                }
                st.session_state["source_focus_index"] = 0
        with meta_col:
            for meta_line in card["meta_lines"]:
                if "http://" in meta_line or "https://" in meta_line:
                    st.markdown(_linkify_text(meta_line))
                else:
                    st.markdown(f"<span class='source-chip'>{meta_line}</span>", unsafe_allow_html=True)

        for body_line in card["body_lines"]:
            if _render_table_like_text(body_line):
                continue
            st.markdown(_linkify_text(body_line))
        for image_line in card["image_lines"]:
            _render_markdown_with_local_images(image_line)


def _render_answer_block(answer_markdown: str) -> None:
    main_answer, react_trace = _split_react_trace(answer_markdown)
    main_answer, source_section = _split_sources_section(main_answer)
    _render_markdown_with_local_images(main_answer)
    if source_section:
        _render_source_cards(source_section)
    if react_trace and st.session_state.get("developer_mode", False):
        with st.expander("ReAct 轨迹（开发者）", expanded=False):
            st.markdown(react_trace)


def _render_qa_history() -> None:
    history = st.session_state.get("qa_history", [])
    if not history:
        st.caption("当前页面还没有问答记录。")
        return
    for index, item in enumerate(history):
        with st.expander(f"{item['created_at']} · {clean_preview_text(item['question'], 42)}", expanded=False):
            st.markdown(f"**问题**：{item['question']}")
            action_col1, action_col2 = st.columns([1, 3])
            with action_col1:
                if st.button("重新追问", key=f"reuse_question_{index}", width="stretch"):
                    st.session_state["question_input"] = item["question"]
                    st.rerun()
            with action_col2:
                st.caption("点击后会把这条历史问题带回输入框，方便继续追问或修改。")
            _render_answer_block(item["answer"])


def _save_current_analysis_to_history() -> bool:
    if st.session_state.get("history_saved"):
        return True

    parsed_documents = st.session_state.get("parsed_documents", [])
    report_markdown = st.session_state.get("report_markdown", "")
    saved_paths = [Path(path) for path in st.session_state.get("pending_saved_paths", [])]
    if not parsed_documents or not report_markdown:
        return False

    session_name = st.session_state.get("pending_session_name") or f"analysis-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    session_id = create_analysis_session(session_name)
    register_uploaded_files(session_id, saved_paths)
    save_parsed_documents(session_id, parsed_documents)
    report_path = Path(st.session_state.get("report_path", ""))
    report_id = save_report_record(session_id, report_path, report_markdown)

    st.session_state["current_session_id"] = session_id
    st.session_state["current_report_id"] = report_id
    st.session_state["history_saved"] = True
    st.session_state["pending_session_name"] = ""
    st.session_state["pending_saved_paths"] = []
    return True


def _render_history_sidebar() -> None:
    st.sidebar.subheader("历史会话")
    sessions = list_sessions(limit=20)
    if not sessions:
        st.sidebar.caption("暂无历史分析记录。")
        st.session_state["developer_mode"] = st.sidebar.toggle(
            "开发者模式",
            value=bool(st.session_state.get("developer_mode", False)),
            help="开启后可查看 Agent 的显式 ReAct 轨迹与更多调试信息。",
        )
        return

    if st.sidebar.button("加载最近一次会话", width="stretch"):
        latest_session = get_latest_session()
        if latest_session and _load_session_into_state(latest_session["id"]):
            st.sidebar.success("已加载最近一次历史会话。")
            st.rerun()
        st.sidebar.error("最近一次历史会话加载失败。")

    session_ids = [item["id"] for item in sessions]
    current_session_id = st.session_state.get("current_session_id", "")
    default_index = session_ids.index(current_session_id) if current_session_id in session_ids else 0

    def _format_session(session_id: str) -> str:
        record = next((item for item in sessions if item["id"] == session_id), None)
        if not record:
            return session_id
        session_name = record.get("session_name", session_id)
        status = record.get("status", "UNKNOWN")
        updated_at = record.get("updated_at", "")
        return f"{session_name} | {status} | {updated_at}"

    selected_session_id = st.sidebar.selectbox(
        "选择要查看的分析会话",
        options=session_ids,
        index=default_index,
        format_func=_format_session,
        key="history_session_selector",
    )

    selected_record = next((item for item in sessions if item["id"] == selected_session_id), None)
    if selected_record:
        st.sidebar.caption(f"会话名称：`{selected_record['session_name']}`")
        st.sidebar.caption(f"状态：`{selected_record['status']}`")
        st.sidebar.caption(f"更新时间：`{selected_record['updated_at']}`")

    if st.sidebar.button("加载所选会话", width="stretch"):
        if _load_session_into_state(selected_session_id):
            st.sidebar.success("已切换到所选历史会话。")
            st.rerun()
        st.sidebar.error("历史会话加载失败。")

    st.sidebar.divider()
    st.session_state["developer_mode"] = st.sidebar.toggle(
        "开发者模式",
        value=bool(st.session_state.get("developer_mode", False)),
        help="开启后可查看 Agent 的显式 ReAct 轨迹与更多调试信息。",
    )


def main() -> None:
    st.set_page_config(page_title="多模态图文智能分析助手", layout="wide")
    init_state()
    _inject_page_style()
    _render_history_sidebar()

    st.title("多模态图文智能分析助手")
    st.markdown(
        """
        <div class="hero-card">
            <div><strong>上传 PDF / Word / 图片</strong>，系统会自动完成图文解析、结构化总结与基于来源的问答。</div>
            <div class="section-hint">当前版本已接入 RAG、LangChain、Agent、MCP、历史会话与答案校验能力。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    current_session_id = st.session_state.get("current_session_id", "")
    if current_session_id:
        st.caption(f"当前分析会话：`{current_session_id}`")

    st.markdown("### 上传与分析")
    st.caption("支持最多 5 个文件，总大小不超过 20MB。建议优先上传包含正文、图表和图片的原始文档。")
    with st.form("upload_analysis_form", clear_on_submit=False):
        uploaded_files = st.file_uploader(
            "上传 PDF / DOCX / PNG / JPG 文件",
            type=["pdf", "docx", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )
        action_col1, action_col2 = st.columns([1, 1])
        with action_col1:
            analyze_clicked = st.form_submit_button("开始分析", type="primary", width="stretch")
        with action_col2:
            clear_clicked = st.form_submit_button("清空当前页面状态", width="stretch")

    if clear_clicked:
        _clear_current_analysis_state()
        st.rerun()

    if analyze_clicked:
        if not uploaded_files:
            st.warning("请先上传至少一个文件。")
        else:
            error = validate_uploads(uploaded_files)
            if error:
                st.error(error)
            else:
                run_id = _unsaved_run_id()
                session_name = _build_session_name(uploaded_files)
                session_temp_dir = _session_temp_dir(run_id)
                session_report_dir = _session_report_dir(run_id)
                st.session_state["current_session_id"] = ""

                with st.spinner("正在解析文档并生成报告，请稍候..."):
                    saved_paths = save_uploaded_files(uploaded_files, session_temp_dir)
                    parsed_documents = parse_files(saved_paths)
                    path_map = {path.name: str(path) for path in saved_paths}
                    for document in parsed_documents:
                        document["file_path"] = path_map.get(document.get("file_name", ""), document.get("file_path", ""))
                    report_markdown = build_markdown_report(parsed_documents)
                    report_path = save_report(report_markdown, session_report_dir)

                st.session_state["parsed_documents"] = parsed_documents
                st.session_state["report_markdown"] = report_markdown
                st.session_state["report_path"] = str(report_path)
                st.session_state["current_report_id"] = ""
                st.session_state["history_saved"] = False
                st.session_state["pending_session_name"] = session_name
                st.session_state["pending_saved_paths"] = [str(path) for path in saved_paths]
                st.session_state["qa_history"] = []
                st.session_state["question_input"] = ""
                st.success("分析完成。当前结果已生成到页面中，如需后续复用可保存到历史会话。")

    parsed_documents = st.session_state["parsed_documents"]
    if parsed_documents:
        _render_overview_metrics(parsed_documents)

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("总结报告")
        if st.session_state["report_markdown"]:
            report_tab, detail_tab = st.tabs(["报告预览", "解析概览"])
            with report_tab:
                _render_markdown_with_local_images(st.session_state["report_markdown"])
            with detail_tab:
                _render_parsed_document_overview(parsed_documents)

            if st.session_state.get("history_saved"):
                st.success("当前分析已保存到历史会话。")
            else:
                st.warning("当前分析仅保存在本次页面状态中，尚未写入历史会话。")
                if st.button("保存到历史会话", type="primary", width="stretch"):
                    if _save_current_analysis_to_history():
                        st.success("已保存到历史会话。")
                        st.rerun()
                    st.error("保存失败：当前没有可保存的分析结果。")

            st.download_button(
                "下载 Markdown 报告",
                data=st.session_state["report_markdown"],
                file_name=Path(st.session_state["report_path"]).name or "report.md",
                mime="text/markdown",
                width="stretch",
            )
            if st.session_state["report_path"]:
                st.caption(f"报告路径：`{st.session_state['report_path']}`")
        else:
            st.info("完成文档分析后，这里会显示结构化 Markdown 报告。")

    with right:
        st.subheader("问答交互")
        suggestions = _build_question_suggestions(parsed_documents) if parsed_documents else []
        if suggestions:
            st.caption("可以直接从这些常见问题开始：")
            suggestion_cols = st.columns(2)
            for index, suggestion in enumerate(suggestions):
                with suggestion_cols[index % 2]:
                    if st.button(suggestion, key=f"suggestion_{index}", width="stretch"):
                        st.session_state["question_input"] = suggestion

        with st.form("qa_form", clear_on_submit=False):
            st.checkbox("结合最近问答上下文理解当前追问", key="follow_up_mode")
            question = st.text_input(
                "请输入问题",
                key="question_input",
                placeholder="例如：图 2 展示了什么？",
            )
            ask_clicked = st.form_submit_button("提问", width="stretch")

        if ask_clicked:
            if not parsed_documents:
                st.warning("请先完成文档分析。")
            elif not question.strip():
                st.warning("请输入问题。")
            else:
                effective_question = _build_follow_up_question(question)
                with st.spinner("正在检索证据并生成回答..."):
                    answer = answer_question(
                        question=effective_question,
                        parsed_documents=parsed_documents,
                        session_id=st.session_state.get("current_session_id") or None,
                    )
                _append_qa_history(question, answer)
                _render_answer_block(answer)

        st.subheader("本页问答记录")
        _render_qa_history()
        _render_source_focus_panel()

        st.subheader("解析结果概览")
        if parsed_documents:
            for document in parsed_documents:
                st.write(f"- `{document['file_name']}`：{len(document['blocks'])} 个内容块")
        else:
            st.caption("暂无解析结果。")


if __name__ == "__main__":
    main()
