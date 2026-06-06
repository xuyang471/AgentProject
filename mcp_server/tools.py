from __future__ import annotations

from typing import List

from services.calculation_service import calculate_expression, normalize_expression
from services.web_search_service import search_web

from .adapters import analyze_image_block, build_documents_from_context, run_image_ocr, search_document_blocks
from .resources import read_report_resource


def _tool_response(data=None, *, ok: bool = True, error: str | None = None, metadata: dict | None = None) -> dict:
    return {
        "ok": ok,
        "data": data,
        "error": error,
        "metadata": metadata or {},
    }


def call_search_document_blocks(query: str, parsed_documents: List[dict] | None = None, session_id: str = "", block_type: str = "") -> dict:
    documents = build_documents_from_context(parsed_documents, session_id=session_id)
    if not documents:
        return _tool_response([], metadata={"query": query, "block_type": block_type, "match_count": 0})
    matches = search_document_blocks(query, documents, block_type=block_type)
    return _tool_response(
        matches,
        metadata={
            "query": query,
            "block_type": block_type,
            "match_count": len(matches),
        },
    )


def call_run_image_ocr(
    parsed_documents: List[dict] | None = None,
    session_id: str = "",
    *,
    file_name: str = "",
    image_index: int | None = None,
    source: str = "",
) -> dict:
    documents = build_documents_from_context(parsed_documents, session_id=session_id)
    payload = run_image_ocr(documents, file_name=file_name, image_index=image_index, source=source)
    if payload.get("error"):
        return _tool_response(None, ok=False, error=str(payload["error"]), metadata={"tool": "run_image_ocr"})
    return _tool_response(payload, metadata={"tool": "run_image_ocr", "source": payload.get("source", "")})


def call_analyze_image(
    parsed_documents: List[dict] | None = None,
    session_id: str = "",
    *,
    file_name: str = "",
    image_index: int | None = None,
    source: str = "",
) -> dict:
    documents = build_documents_from_context(parsed_documents, session_id=session_id)
    payload = analyze_image_block(documents, file_name=file_name, image_index=image_index, source=source)
    if payload.get("error"):
        return _tool_response(None, ok=False, error=str(payload["error"]), metadata={"tool": "analyze_image"})
    return _tool_response(payload, metadata={"tool": "analyze_image", "source": payload.get("source", "")})


def call_get_report_summary(session_id: str = "") -> dict:
    report = read_report_resource(session_id=session_id)
    summary = str(report.get("summary", "")).strip()
    content = str(report.get("content", "")).strip()
    return _tool_response(
        {
            "session_id": session_id,
            "summary": summary or content[:400],
            "content_preview": content[:400],
        },
        metadata={"tool": "get_report_summary"},
    )


def call_calculator(expression: str) -> dict:
    normalized = normalize_expression(expression)
    try:
        result = calculate_expression(expression)
    except Exception as exc:
        return _tool_response(
            None,
            ok=False,
            error=str(exc),
            metadata={"tool": "calculator", "expression": normalized},
        )

    return _tool_response(
        {
            "expression": normalized,
            "result": result,
        },
        metadata={"tool": "calculator", "expression": normalized},
    )


def call_web_search(query: str, top_k: int = 5) -> dict:
    normalized_query = str(query or "").strip()
    try:
        results = search_web(query=normalized_query, top_k=top_k)
    except Exception as exc:
        return _tool_response(
            None,
            ok=False,
            error=str(exc),
            metadata={"tool": "web_search", "query": normalized_query},
        )

    return _tool_response(
        results,
        metadata={
            "tool": "web_search",
            "query": normalized_query,
            "match_count": len(results),
        },
    )
