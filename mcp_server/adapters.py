from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from services.llm_service import analyze_image
from services.ocr_service import extract_text_from_image
from services.retrieval_service import retrieve_relevant_blocks
from services.state_service import get_latest_session, load_session_context


def resolve_session_id(session_id: str = "") -> str:
    if session_id:
        return session_id
    latest = get_latest_session()
    return latest["id"] if latest else ""


def load_session_payload(session_id: str = "") -> dict | None:
    resolved = resolve_session_id(session_id)
    if not resolved:
        return None
    return load_session_context(resolved)


def build_documents_from_context(parsed_documents: Optional[List[dict]], session_id: str = "") -> List[dict]:
    if parsed_documents:
        return parsed_documents
    context = load_session_payload(session_id)
    if not context:
        return []
    return context.get("documents", [])


def find_image_block(
    parsed_documents: List[dict],
    *,
    file_name: str = "",
    image_index: Optional[int] = None,
    source: str = "",
):
    for document in parsed_documents:
        for block in document.get("blocks", []):
            if block.get("type") != "image":
                continue
            if source and block.get("source") == source:
                return document, block
            if file_name and document.get("file_name") != file_name:
                continue
            if image_index is not None and block.get("image_index") != image_index:
                continue
            if file_name:
                return document, block
    return None, None


def iter_blocks(parsed_documents: List[dict], block_type: str = ""):
    for document in parsed_documents:
        for block in document.get("blocks", []):
            if block_type and block.get("type") != block_type:
                continue
            yield document, block


def search_document_blocks(query: str, parsed_documents: List[dict], block_type: str = "") -> List[dict]:
    items = []
    for document, block in iter_blocks(parsed_documents, block_type=block_type):
        content = str(block.get("content", "")).strip()
        if not content:
            continue
        items.append(
            {
                "content": content,
                "source": block.get("source", "未知来源"),
                "type": block.get("type", ""),
                "file_name": document.get("file_name", ""),
                "page": block.get("page"),
                "image_index": block.get("image_index"),
                "image_path": block.get("image_path", ""),
                "description": block.get("description", ""),
                "ocr_text": block.get("ocr_text", ""),
            }
        )
    return retrieve_relevant_blocks(query, items, top_k=3)


def run_image_ocr(parsed_documents: List[dict], *, file_name: str = "", image_index: Optional[int] = None, source: str = "") -> dict:
    document, block = find_image_block(
        parsed_documents,
        file_name=file_name,
        image_index=image_index,
        source=source,
    )
    if not document or not block:
        return {"error": "未找到目标图片块。"}

    image_path = block.get("image_path", "")
    if not image_path or not Path(image_path).exists():
        return {"error": "图片文件路径不存在，无法重新执行 OCR。"}

    text = extract_text_from_image(image_path)
    return {
        "file_name": document.get("file_name", ""),
        "source": block.get("source", ""),
        "image_index": block.get("image_index"),
        "ocr_text": text,
    }


def analyze_image_block(
    parsed_documents: List[dict],
    *,
    file_name: str = "",
    image_index: Optional[int] = None,
    source: str = "",
) -> dict:
    document, block = find_image_block(
        parsed_documents,
        file_name=file_name,
        image_index=image_index,
        source=source,
    )
    if not document or not block:
        return {"error": "未找到目标图片块。"}

    image_path = block.get("image_path", "")
    if not image_path or not Path(image_path).exists():
        return {"error": "图片文件路径不存在，无法重新执行图片分析。"}

    local_ocr_text = extract_text_from_image(image_path)
    analysis = analyze_image(image_path, local_ocr_text=local_ocr_text)
    return {
        "file_name": document.get("file_name", ""),
        "source": block.get("source", ""),
        "image_index": block.get("image_index"),
        "ocr_text": analysis.get("ocr_text", ""),
        "description": analysis.get("description", ""),
    }
