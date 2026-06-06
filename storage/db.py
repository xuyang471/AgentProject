from __future__ import annotations

import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
DB_PATH = OUTPUT_DIR / "system_state.db"


def get_connection() -> sqlite3.Connection:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_database() -> None:
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS analysis_session (
                id TEXT PRIMARY KEY,
                session_name TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS document_record (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                file_name TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                parse_status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES analysis_session(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS document_block (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                block_type TEXT NOT NULL,
                page INTEGER,
                block_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                image_path TEXT,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES analysis_session(id) ON DELETE CASCADE,
                FOREIGN KEY (document_id) REFERENCES document_record(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS report_record (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                report_path TEXT NOT NULL,
                summary TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES analysis_session(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS qa_record (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                route_type TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES analysis_session(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS agent_run (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                status TEXT NOT NULL,
                intent TEXT,
                route TEXT,
                steps_json TEXT NOT NULL,
                final_answer TEXT NOT NULL,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES analysis_session(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_document_record_session_id
            ON document_record(session_id);

            CREATE INDEX IF NOT EXISTS idx_document_block_document_id
            ON document_block(document_id);

            CREATE INDEX IF NOT EXISTS idx_report_record_session_id
            ON report_record(session_id);

            CREATE INDEX IF NOT EXISTS idx_qa_record_session_id
            ON qa_record(session_id);

            CREATE INDEX IF NOT EXISTS idx_agent_run_session_id
            ON agent_run(session_id);
            """
        )
