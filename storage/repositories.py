from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from .db import get_connection, init_database


def _now() -> str:
    return datetime.now().isoformat(timespec="microseconds")


class StateRepository:
    def __init__(self) -> None:
        init_database()

    def create_session(self, session_name: str, status: str = "CREATED") -> str:
        session_id = uuid4().hex
        timestamp = _now()
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO analysis_session (id, session_name, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, session_name, status, timestamp, timestamp),
            )
        return session_id

    def update_session_status(self, session_id: str, status: str) -> None:
        with get_connection() as connection:
            connection.execute(
                """
                UPDATE analysis_session
                SET status = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, _now(), session_id),
            )

    def register_uploaded_files(self, session_id: str, file_paths: Iterable[Path]) -> None:
        timestamp = _now()
        with get_connection() as connection:
            for file_path in file_paths:
                document_id = uuid4().hex
                connection.execute(
                    """
                    INSERT INTO document_record (
                        id, session_id, file_name, file_type, file_path, parse_status, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        session_id,
                        file_path.name,
                        file_path.suffix.lower().lstrip(".") or "unknown",
                        str(file_path),
                        "UPLOADED",
                        timestamp,
                        timestamp,
                    ),
                )

    def save_parsed_documents(self, session_id: str, parsed_documents: list[dict]) -> None:
        timestamp = _now()
        with get_connection() as connection:
            document_rows = connection.execute(
                """
                SELECT id, file_name
                FROM document_record
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchall()
            document_id_map = {row["file_name"]: row["id"] for row in document_rows}

            for document in parsed_documents:
                file_name = document.get("file_name", "")
                document_id = document_id_map.get(file_name)
                if not document_id:
                    document_id = uuid4().hex
                    connection.execute(
                        """
                        INSERT INTO document_record (
                            id, session_id, file_name, file_type, file_path, parse_status, created_at, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            document_id,
                            session_id,
                            file_name,
                            document.get("file_type", "unknown"),
                            "",
                            "PARSED",
                            timestamp,
                            timestamp,
                        ),
                    )
                    document_id_map[file_name] = document_id
                else:
                    connection.execute(
                        """
                        UPDATE document_record
                        SET file_type = ?, parse_status = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (document.get("file_type", "unknown"), "PARSED", timestamp, document_id),
                    )

                connection.execute(
                    "DELETE FROM document_block WHERE document_id = ?",
                    (document_id,),
                )

                for block_index, block in enumerate(document.get("blocks", []), start=1):
                    metadata = {
                        key: value
                        for key, value in block.items()
                        if key not in {"type", "page", "content", "source", "image_path"}
                    }
                    connection.execute(
                        """
                        INSERT INTO document_block (
                            id, session_id, document_id, block_type, page, block_index, content,
                            source, image_path, metadata_json, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            uuid4().hex,
                            session_id,
                            document_id,
                            block.get("type", "text"),
                            block.get("page"),
                            block_index,
                            str(block.get("content", "")),
                            str(block.get("source", file_name)),
                            str(block.get("image_path", "")),
                            json.dumps(metadata, ensure_ascii=False),
                            timestamp,
                        ),
                    )

    def save_report(self, session_id: str, report_path: Path, markdown_text: str, summary: str) -> str:
        report_id = uuid4().hex
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO report_record (id, session_id, report_path, summary, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (report_id, session_id, str(report_path), summary, markdown_text, _now()),
            )
        return report_id

    def save_qa_record(
        self,
        session_id: str,
        question: str,
        answer: str,
        route_type: str,
        sources: list[dict] | list[str] | None,
    ) -> str:
        qa_id = uuid4().hex
        payload = json.dumps(sources or [], ensure_ascii=False)
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO qa_record (id, session_id, question, answer, route_type, sources_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (qa_id, session_id, question, answer, route_type, payload, _now()),
            )
        return qa_id

    def save_agent_run(
        self,
        session_id: str,
        question: str,
        status: str,
        intent: str,
        route: str,
        steps: list[dict] | None,
        final_answer: str,
        error_message: str = "",
    ) -> str:
        run_id = uuid4().hex
        timestamp = _now()
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO agent_run (
                    id, session_id, question, status, intent, route, steps_json,
                    final_answer, error_message, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    session_id,
                    question,
                    status,
                    intent,
                    route,
                    json.dumps(steps or [], ensure_ascii=False),
                    final_answer,
                    error_message,
                    timestamp,
                    timestamp,
                ),
            )
        return run_id

    def get_latest_session(self) -> dict | None:
        with get_connection() as connection:
            row = connection.execute(
                """
                SELECT id, session_name, status, created_at, updated_at
                FROM analysis_session
                ORDER BY updated_at DESC, created_at DESC, rowid DESC
                LIMIT 1
                """
            ).fetchone()
        return dict(row) if row else None

    def list_sessions(self, limit: int = 20) -> list[dict]:
        with get_connection() as connection:
            rows = connection.execute(
                """
                SELECT id, session_name, status, created_at, updated_at
                FROM analysis_session
                ORDER BY updated_at DESC, created_at DESC, rowid DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_session_context(self, session_id: str) -> dict | None:
        with get_connection() as connection:
            session_row = connection.execute(
                """
                SELECT id, session_name, status, created_at, updated_at
                FROM analysis_session
                WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
            if not session_row:
                return None

            document_rows = connection.execute(
                """
                SELECT id, file_name, file_type, file_path, parse_status
                FROM document_record
                WHERE session_id = ?
                ORDER BY created_at ASC
                """,
                (session_id,),
            ).fetchall()
            block_rows = connection.execute(
                """
                SELECT document_id, block_type, page, block_index, content, source, image_path, metadata_json
                FROM document_block
                WHERE session_id = ?
                ORDER BY document_id ASC, block_index ASC
                """,
                (session_id,),
            ).fetchall()
            report_row = connection.execute(
                """
                SELECT id, report_path, summary, content, created_at
                FROM report_record
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
            qa_row = connection.execute(
                """
                SELECT id, question, answer, route_type, sources_json, created_at
                FROM qa_record
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()

        documents_by_id: dict[str, dict] = {}
        for row in document_rows:
            documents_by_id[row["id"]] = {
                "file_name": row["file_name"],
                "file_type": row["file_type"],
                "file_path": row["file_path"],
                "parse_status": row["parse_status"],
                "blocks": [],
            }

        for row in block_rows:
            metadata = json.loads(row["metadata_json"] or "{}")
            block = {
                "type": row["block_type"],
                "page": row["page"],
                "content": row["content"],
                "source": row["source"],
            }
            if row["image_path"]:
                block["image_path"] = row["image_path"]
            block.update(metadata)
            document = documents_by_id.get(row["document_id"])
            if document is not None:
                document["blocks"].append(block)

        latest_qa = None
        if qa_row:
            latest_qa = dict(qa_row)
            latest_qa["sources"] = json.loads(qa_row["sources_json"] or "[]")

        return {
            "session": dict(session_row),
            "documents": list(documents_by_id.values()),
            "report": dict(report_row) if report_row else None,
            "latest_qa": latest_qa,
        }

    def get_qa_history(self, session_id: str, limit: int = 10) -> list[dict]:
        with get_connection() as connection:
            rows = connection.execute(
                """
                SELECT id, question, answer, route_type, sources_json, created_at
                FROM qa_record
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        history = []
        for row in rows:
            item = dict(row)
            item["sources"] = json.loads(row["sources_json"] or "[]")
            history.append(item)
        return history
