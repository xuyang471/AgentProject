from __future__ import annotations

from typing import List

from services.state_service import get_latest_session, get_qa_history, load_session_context

from .adapters import resolve_session_id


def read_latest_session_resource() -> dict:
    latest = get_latest_session()
    return latest or {}


def read_report_resource(session_id: str = "") -> dict:
    resolved = resolve_session_id(session_id)
    if not resolved:
        return {}
    context = load_session_context(resolved)
    if not context:
        return {}
    return context.get("report") or {}


def read_documents_resource(session_id: str = "") -> List[dict]:
    resolved = resolve_session_id(session_id)
    if not resolved:
        return []
    context = load_session_context(resolved)
    if not context:
        return []
    return context.get("documents", [])


def read_qa_history_resource(session_id: str = "", limit: int = 10) -> List[dict]:
    resolved = resolve_session_id(session_id)
    if not resolved:
        return []
    return get_qa_history(resolved, limit=limit)
