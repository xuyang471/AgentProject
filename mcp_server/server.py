from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, List, Tuple

from .resources import (
    read_documents_resource,
    read_latest_session_resource,
    read_qa_history_resource,
    read_report_resource,
)
from .tools import (
    call_analyze_image,
    call_calculator,
    call_get_report_summary,
    call_run_image_ocr,
    call_search_document_blocks,
    call_web_search,
)

MCP_CONTEXT_ENV = "AGENTPROJECT_MCP_CONTEXT_PATH"


def load_runtime_context_from_env() -> Tuple[List[dict], str]:
    context_path = os.getenv(MCP_CONTEXT_ENV, "").strip()
    if not context_path:
        return [], ""

    path = Path(context_path)
    if not path.exists():
        return [], ""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return [], ""

    parsed_documents = payload.get("parsed_documents", [])
    session_id = str(payload.get("session_id", "") or "")
    return (parsed_documents if isinstance(parsed_documents, list) else []), session_id


class LocalMCPServer:
    def __init__(self, parsed_documents: List[dict] | None = None, session_id: str = "") -> None:
        self.parsed_documents = parsed_documents or []
        self.session_id = session_id

    def list_resources(self) -> list[str]:
        resolved = self.session_id or "{session_id}"
        return [
            "analysis://sessions/latest",
            f"analysis://sessions/{resolved}/report",
            f"analysis://sessions/{resolved}/documents",
            f"analysis://sessions/{resolved}/qa-history",
            "session://latest",
            f"session://{resolved}/report",
            f"session://{resolved}/documents",
            f"session://{resolved}/qa-history",
        ]

    def list_tools(self) -> list[str]:
        return [
            "search_document_blocks",
            "run_image_ocr",
            "analyze_image",
            "get_report_summary",
            "calculator",
            "web_search",
        ]

    def read_resource(self, uri: str) -> dict | list:
        if uri in {"session://latest", "analysis://sessions/latest"}:
            return read_latest_session_resource()

        if uri.startswith("analysis://sessions/") and uri.endswith("/report"):
            session_id = uri.split("analysis://sessions/", 1)[1].rsplit("/report", 1)[0]
            return read_report_resource(session_id=session_id)

        if uri.startswith("analysis://sessions/") and uri.endswith("/documents"):
            session_id = uri.split("analysis://sessions/", 1)[1].rsplit("/documents", 1)[0]
            return read_documents_resource(session_id=session_id)

        if uri.startswith("analysis://sessions/") and uri.endswith("/qa-history"):
            session_id = uri.split("analysis://sessions/", 1)[1].rsplit("/qa-history", 1)[0]
            return read_qa_history_resource(session_id=session_id)

        if uri.startswith("session://") and uri.endswith("/report"):
            session_id = uri.split("session://", 1)[1].rsplit("/report", 1)[0]
            return read_report_resource(session_id=session_id)

        if uri.startswith("session://") and uri.endswith("/documents"):
            session_id = uri.split("session://", 1)[1].rsplit("/documents", 1)[0]
            return read_documents_resource(session_id=session_id)

        if uri.startswith("session://") and uri.endswith("/qa-history"):
            session_id = uri.split("session://", 1)[1].rsplit("/qa-history", 1)[0]
            return read_qa_history_resource(session_id=session_id)

        return {}

    def call_tool(self, name: str, **kwargs):
        tool_map: dict[str, Callable] = {
            "search_document_blocks": lambda **payload: call_search_document_blocks(
                parsed_documents=self.parsed_documents,
                session_id=self.session_id,
                **payload,
            ),
            "run_image_ocr": lambda **payload: call_run_image_ocr(
                parsed_documents=self.parsed_documents,
                session_id=self.session_id,
                **payload,
            ),
            "analyze_image": lambda **payload: call_analyze_image(
                parsed_documents=self.parsed_documents,
                session_id=self.session_id,
                **payload,
            ),
            "get_report_summary": lambda **payload: call_get_report_summary(
                session_id=payload.get("session_id", self.session_id),
            ),
            "calculator": lambda **payload: call_calculator(
                expression=str(payload.get("expression", "") or ""),
            ),
            "web_search": lambda **payload: call_web_search(
                query=str(payload.get("query", "") or ""),
                top_k=int(payload.get("top_k", 5) or 5),
            ),
        }
        handler = tool_map.get(name)
        if handler is None:
            return {"ok": False, "data": None, "error": f"Unknown MCP tool: {name}", "metadata": {"tool": name}}
        return handler(**kwargs)

    def handle_request(self, method: str, params: dict | None = None):
        payload = params or {}

        if method == "initialize":
            self.session_id = str(payload.get("session_id", "") or "")
            parsed_documents = payload.get("parsed_documents", [])
            self.parsed_documents = parsed_documents if isinstance(parsed_documents, list) else []
            return {
                "session_id": self.session_id,
                "resource_count": len(self.list_resources()),
                "tool_count": len(self.list_tools()),
            }

        if method == "ping":
            return {"pong": True}

        if method == "list_resources":
            return self.list_resources()

        if method == "list_tools":
            return self.list_tools()

        if method == "read_resource":
            uri = str(payload.get("uri", "") or "")
            return self.read_resource(uri)

        if method == "call_tool":
            name = str(payload.get("name", "") or "")
            tool_args = payload.get("arguments", {}) or {}
            if not isinstance(tool_args, dict):
                raise ValueError("tool arguments must be an object")
            return self.call_tool(name, **tool_args)

        if method == "dump_state":
            return {
                "session_id": self.session_id,
                "parsed_documents_count": len(self.parsed_documents),
            }

        raise ValueError(f"Unsupported MCP method: {method}")


def create_fastmcp_server(parsed_documents: List[dict] | None = None, session_id: str = ""):
    try:
        from mcp.server.fastmcp import FastMCP
    except ModuleNotFoundError:
        return None

    backend = LocalMCPServer(parsed_documents=parsed_documents, session_id=session_id)
    mcp = FastMCP(
        "AgentProject Document Analysis MCP",
        instructions=(
            "This MCP server exposes document search, OCR, image analysis, "
            "report summary, calculator tools, web search, and session resources for the AgentProject document assistant."
        ),
    )

    def _to_resource_text(payload) -> str:
        return json.dumps(payload, ensure_ascii=False)

    @mcp.tool()
    def search_document_blocks(query: str, block_type: str = "") -> dict:
        """Search parsed document blocks by semantic/keyword relevance."""
        return backend.call_tool("search_document_blocks", query=query, block_type=block_type)

    @mcp.tool()
    def run_image_ocr(file_name: str = "", image_index: int | None = None, source: str = "") -> dict:
        """Run OCR on a selected image block."""
        return backend.call_tool(
            "run_image_ocr",
            file_name=file_name,
            image_index=image_index,
            source=source,
        )

    @mcp.tool()
    def analyze_image(file_name: str = "", image_index: int | None = None, source: str = "") -> dict:
        """Run multimodal image analysis on a selected image block."""
        return backend.call_tool(
            "analyze_image",
            file_name=file_name,
            image_index=image_index,
            source=source,
        )

    @mcp.tool()
    def get_report_summary(session_id: str = "") -> dict:
        """Read the current or selected session report summary."""
        return backend.call_tool("get_report_summary", session_id=session_id or backend.session_id)

    @mcp.tool()
    def calculator(expression: str) -> dict:
        """Safely evaluate a math expression for precise calculations."""
        return backend.call_tool("calculator", expression=expression)

    @mcp.tool()
    def web_search(query: str, top_k: int = 5) -> dict:
        """Search external web sources when document evidence is insufficient or latest information is required."""
        return backend.call_tool("web_search", query=query, top_k=top_k)

    @mcp.resource("analysis://sessions/latest")
    def latest_analysis_session() -> str:
        """Read the latest analysis session."""
        return _to_resource_text(backend.read_resource("analysis://sessions/latest"))

    @mcp.resource("analysis://sessions/{target_session_id}/report")
    def analysis_report(target_session_id: str) -> str:
        """Read a report resource for a specific analysis session."""
        return _to_resource_text(backend.read_resource(f"analysis://sessions/{target_session_id}/report"))

    @mcp.resource("analysis://sessions/{target_session_id}/documents")
    def analysis_documents(target_session_id: str) -> str:
        """Read parsed documents for a specific analysis session."""
        return _to_resource_text(backend.read_resource(f"analysis://sessions/{target_session_id}/documents"))

    @mcp.resource("analysis://sessions/{target_session_id}/qa-history")
    def analysis_qa_history(target_session_id: str) -> str:
        """Read question-answer history for a specific analysis session."""
        return _to_resource_text(backend.read_resource(f"analysis://sessions/{target_session_id}/qa-history"))

    @mcp.resource("session://latest")
    def latest_session_legacy() -> str:
        """Legacy alias for the latest session resource."""
        return _to_resource_text(backend.read_resource("session://latest"))

    @mcp.resource("session://{target_session_id}/report")
    def session_report_legacy(target_session_id: str) -> str:
        """Legacy alias for a session report resource."""
        return _to_resource_text(backend.read_resource(f"session://{target_session_id}/report"))

    @mcp.resource("session://{target_session_id}/documents")
    def session_documents_legacy(target_session_id: str) -> str:
        """Legacy alias for session documents resource."""
        return _to_resource_text(backend.read_resource(f"session://{target_session_id}/documents"))

    @mcp.resource("session://{target_session_id}/qa-history")
    def session_qa_history_legacy(target_session_id: str) -> str:
        """Legacy alias for session QA history resource."""
        return _to_resource_text(backend.read_resource(f"session://{target_session_id}/qa-history"))

    return mcp
