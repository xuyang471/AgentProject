from __future__ import annotations

import json
from typing import List, Optional

from mcp_server import LocalMCPServer

from .mcp_client import StdioMCPClient


def _create_server_backed_interface(parsed_documents: List[dict], session_id: str):
    try:
        client = StdioMCPClient(parsed_documents=parsed_documents, session_id=session_id)
        return ("client", client)
    except Exception:
        return ("local", LocalMCPServer(parsed_documents=parsed_documents, session_id=session_id))


def create_document_tools(parsed_documents: List[dict], session_id: str = "") -> list:
    try:
        from langchain_core.tools import StructuredTool
    except ModuleNotFoundError:
        return []

    mode, backend = _create_server_backed_interface(parsed_documents, session_id)

    def _read_resource(uri: str):
        if mode == "client":
            return backend.read_resource(uri)
        return backend.read_resource(uri)

    def _call_tool(name: str, arguments: Optional[dict] = None):
        if mode == "client":
            return backend.call_tool(name, arguments or {})
        return backend.call_tool(name, **(arguments or {}))

    def read_latest_session_resource() -> str:
        return str(_read_resource("session://latest"))

    def read_report_resource(session_id: str = "") -> str:
        target_session_id = session_id or getattr(backend, "session_id", "") or ""
        if not target_session_id:
            return "{}"
        return str(_read_resource(f"session://{target_session_id}/report"))

    def search_document_blocks(question: str, block_type: str = "") -> str:
        result = _call_tool("search_document_blocks", {"query": question, "block_type": block_type})
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

    def run_ocr_on_image(file_name: str = "", image_index: Optional[int] = None, source: str = "") -> str:
        result = _call_tool(
            "run_image_ocr",
            {
                "file_name": file_name,
                "image_index": image_index,
                "source": source,
            },
        )
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

    def analyze_image_with_mllm(file_name: str = "", image_index: Optional[int] = None, source: str = "") -> str:
        result = _call_tool(
            "analyze_image",
            {
                "file_name": file_name,
                "image_index": image_index,
                "source": source,
            },
        )
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

    def get_report_summary(session_id: str = "") -> str:
        result = _call_tool("get_report_summary", {"session_id": session_id or getattr(backend, "session_id", "")})
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

    def calculator(expression: str) -> str:
        result = _call_tool("calculator", {"expression": expression})
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

    def web_search(query: str, top_k: int = 5) -> str:
        result = _call_tool("web_search", {"query": query, "top_k": top_k})
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

    return [
        StructuredTool.from_function(
            func=read_latest_session_resource,
            name="read_latest_session_resource",
            description="通过 MCP 资源读取最近一次分析会话的基本信息，包括会话 id、状态和时间。",
        ),
        StructuredTool.from_function(
            func=read_report_resource,
            name="read_report_resource",
            description="通过 MCP 资源读取指定会话的报告内容。默认读取当前会话。",
        ),
        StructuredTool.from_function(
            func=search_document_blocks,
            name="search_document_blocks",
            description="通过 MCP 工具检索当前文档块，可指定 block_type 为 text、image、table 或留空。",
        ),
        StructuredTool.from_function(
            func=run_ocr_on_image,
            name="run_ocr_on_image",
            description="通过 MCP 工具对指定图片重新执行 OCR。优先传 source，也可传 file_name 和 image_index。",
        ),
        StructuredTool.from_function(
            func=analyze_image_with_mllm,
            name="analyze_image_with_mllm",
            description="通过 MCP 工具对指定图片重新执行多模态图片分析。优先传 source，也可传 file_name 和 image_index。",
        ),
        StructuredTool.from_function(
            func=get_report_summary,
            name="get_report_summary",
            description="通过 MCP 工具读取当前会话的报告摘要，适合概括类问题。",
        ),
        StructuredTool.from_function(
            func=calculator,
            name="calculator",
            description="通过 MCP 工具安全计算数学表达式，适合求和、乘除、比例、增长率等精确数值计算问题。",
        ),
        StructuredTool.from_function(
            func=web_search,
            name="web_search",
            description="通过 MCP 工具联网查询外部资料，适合官网、最新信息、外部背景知识等问题。",
        ),
    ]
