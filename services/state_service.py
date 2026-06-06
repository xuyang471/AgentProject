from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from storage import StateRepository


_repository = StateRepository()


SESSION_STATUS_CREATED = "CREATED"
SESSION_STATUS_FILES_UPLOADED = "FILES_UPLOADED"
SESSION_STATUS_PARSING = "PARSING"
SESSION_STATUS_PARSED = "PARSED"
SESSION_STATUS_REPORT_GENERATING = "REPORT_GENERATING"
SESSION_STATUS_REPORT_READY = "REPORT_READY"
SESSION_STATUS_QA_AVAILABLE = "QA_AVAILABLE"


def _extract_summary(markdown_text: str) -> str:
    match = re.search(
        r"##\s*1\.\s*核心摘要\s*\n(?P<summary>.+?)(?:\n##\s|\Z)",
        markdown_text,
        re.DOTALL,
    )
    if not match:
        return ""
    return " ".join(match.group("summary").split()).strip()


def create_analysis_session(session_name: str) -> str:
    return _repository.create_session(session_name=session_name, status=SESSION_STATUS_CREATED)


def mark_session_status(session_id: str, status: str) -> None:
    _repository.update_session_status(session_id, status)


def register_uploaded_files(session_id: str, file_paths: Iterable[Path]) -> None:
    _repository.register_uploaded_files(session_id, file_paths)
    mark_session_status(session_id, SESSION_STATUS_FILES_UPLOADED)


def save_parsed_documents(session_id: str, parsed_documents: list[dict]) -> None:
    _repository.save_parsed_documents(session_id, parsed_documents)
    mark_session_status(session_id, SESSION_STATUS_PARSED)


def save_report_record(session_id: str, report_path: Path, markdown_text: str) -> str:
    summary = _extract_summary(markdown_text)
    report_id = _repository.save_report(session_id, report_path, markdown_text, summary)
    mark_session_status(session_id, SESSION_STATUS_REPORT_READY)
    return report_id


def save_qa_record(
    session_id: str,
    question: str,
    answer: str,
    route_type: str,
    sources: list[dict] | list[str] | None,
) -> str:
    qa_id = _repository.save_qa_record(
        session_id=session_id,
        question=question,
        answer=answer,
        route_type=route_type,
        sources=sources,
    )
    mark_session_status(session_id, SESSION_STATUS_QA_AVAILABLE)
    return qa_id


def save_agent_run(
    session_id: str,
    question: str,
    status: str,
    intent: str,
    route: str,
    steps: list[dict] | None,
    final_answer: str,
    error_message: str = "",
) -> str:
    return _repository.save_agent_run(
        session_id=session_id,
        question=question,
        status=status,
        intent=intent,
        route=route,
        steps=steps,
        final_answer=final_answer,
        error_message=error_message,
    )


def get_latest_session() -> dict | None:
    return _repository.get_latest_session()


def list_sessions(limit: int = 20) -> list[dict]:
    return _repository.list_sessions(limit=limit)


def get_qa_history(session_id: str, limit: int = 10) -> list[dict]:
    return _repository.get_qa_history(session_id=session_id, limit=limit)


def load_session_context(session_id: str) -> dict | None:
    return _repository.get_session_context(session_id)
