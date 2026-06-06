from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import uuid
import weakref
from pathlib import Path
from typing import Any, Optional

from services.file_service import ensure_directory

from mcp_server.server import MCP_CONTEXT_ENV


BASE_DIR = Path(__file__).resolve().parent.parent
MCP_RUNTIME_DIR = BASE_DIR / "output" / "mcp_runtime"


def _decode_json_text(value: str):
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return text


def _write_runtime_context(parsed_documents: list[dict] | None, session_id: str) -> Path:
    runtime_dir = ensure_directory(MCP_RUNTIME_DIR)
    context_path = runtime_dir / f"mcp_context_{uuid.uuid4().hex}.json"
    payload = {
        "session_id": session_id,
        "parsed_documents": parsed_documents or [],
    }
    context_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return context_path


class LegacyStdioMCPClient:
    def __init__(self, parsed_documents: list[dict] | None = None, session_id: str = "") -> None:
        self.parsed_documents = parsed_documents or []
        self.session_id = session_id
        self._process: subprocess.Popen[str] | None = None
        self._finalizer: weakref.finalize | None = None
        self._start()
        self._request(
            "initialize",
            {
                "session_id": self.session_id,
                "parsed_documents": self.parsed_documents,
            },
        )

    def _start(self) -> None:
        self._process = subprocess.Popen(
            [sys.executable, "-m", "mcp_server.main", "--legacy"],
            cwd=str(BASE_DIR),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        self._finalizer = weakref.finalize(self, self._cleanup_process, self._process)
        ready_line = self._process.stdout.readline().strip() if self._process.stdout else ""
        if not ready_line:
            raise RuntimeError("MCP server failed to start")
        ready_payload = json.loads(ready_line)
        if ready_payload.get("status") != "ready":
            raise RuntimeError("MCP server did not report ready state")

    def _request(self, method: str, params: dict | None = None) -> Any:
        if self._process is None or self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("MCP client is not connected")

        payload = {"method": method, "params": params or {}}
        self._process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._process.stdin.flush()

        response_line = self._process.stdout.readline().strip()
        if not response_line:
            error_text = ""
            if self._process.stderr is not None:
                error_text = self._process.stderr.read().strip()
            raise RuntimeError(error_text or "empty MCP response")

        response = json.loads(response_line)
        if not response.get("ok", False):
            raise RuntimeError(str(response.get("error", "unknown MCP error")))
        return response.get("result")

    def list_resources(self) -> list[str]:
        return self._request("list_resources")

    def list_tools(self) -> list[str]:
        return self._request("list_tools")

    def read_resource(self, uri: str) -> Any:
        return self._request("read_resource", {"uri": uri})

    def call_tool(self, name: str, arguments: Optional[dict] = None) -> Any:
        return self._request("call_tool", {"name": name, "arguments": arguments or {}})

    def ping(self) -> dict:
        return self._request("ping")

    def close(self) -> None:
        if self._process is None:
            return
        try:
            if self._process.poll() is None:
                self._request("shutdown")
        except Exception:
            pass
        finally:
            if self._process.stdin:
                self._process.stdin.close()
            if self._process.stdout:
                self._process.stdout.close()
            if self._process.stderr:
                self._process.stderr.close()
            if self._process.poll() is None:
                self._process.terminate()
                self._process.wait(timeout=5)
            if self._finalizer and self._finalizer.alive:
                self._finalizer.detach()
            self._process = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _cleanup_process(process: subprocess.Popen[str]) -> None:
        try:
            if process.stdin:
                process.stdin.close()
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        except Exception:
            pass


class OfficialStdioMCPClient:
    def __init__(self, parsed_documents: list[dict] | None = None, session_id: str = "") -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self.parsed_documents = parsed_documents or []
        self.session_id = session_id
        self._ClientSession = ClientSession
        self._StdioServerParameters = StdioServerParameters
        self._stdio_client = stdio_client
        self._loop = asyncio.new_event_loop()
        self._stdio_cm = None
        self._session_cm = None
        self._session = None
        self._context_path = _write_runtime_context(self.parsed_documents, self.session_id)
        self._start()

    def _start(self) -> None:
        env = dict(os.environ)
        env[MCP_CONTEXT_ENV] = str(self._context_path)
        params = self._StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_server.main", "--official"],
            env=env,
        )
        self._stdio_cm = self._stdio_client(params)
        read_stream, write_stream = self._loop.run_until_complete(self._stdio_cm.__aenter__())
        self._session_cm = self._ClientSession(read_stream, write_stream)
        self._session = self._loop.run_until_complete(self._session_cm.__aenter__())
        self._loop.run_until_complete(self._session.initialize())

    def list_resources(self) -> list[str]:
        resources = []
        response = self._loop.run_until_complete(self._session.list_resources())
        for item in getattr(response, "resources", []) or []:
            uri = getattr(item, "uri", "")
            if uri:
                resources.append(str(uri))
        try:
            templates = self._loop.run_until_complete(self._session.list_resource_templates())
            for item in getattr(templates, "resourceTemplates", []) or []:
                uri_template = getattr(item, "uriTemplate", "")
                if uri_template:
                    resources.append(str(uri_template))
        except Exception:
            pass
        return resources

    def list_tools(self) -> list[str]:
        response = self._loop.run_until_complete(self._session.list_tools())
        return [str(getattr(tool, "name", "")) for tool in getattr(response, "tools", []) if getattr(tool, "name", "")]

    def read_resource(self, uri: str) -> Any:
        response = self._loop.run_until_complete(self._session.read_resource(uri))
        contents = getattr(response, "contents", None) or getattr(response, "content", None) or []
        if not isinstance(contents, list):
            contents = [contents]
        text_parts = []
        for item in contents:
            text = getattr(item, "text", None)
            if text is not None:
                text_parts.append(str(text))
        if not text_parts:
            return {}
        if len(text_parts) == 1:
            return _decode_json_text(text_parts[0])
        return [_decode_json_text(text) for text in text_parts]

    def call_tool(self, name: str, arguments: Optional[dict] = None) -> Any:
        result = self._loop.run_until_complete(self._session.call_tool(name, arguments or {}))
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            return structured

        contents = getattr(result, "content", []) or []
        text_parts = []
        for item in contents:
            text = getattr(item, "text", None)
            if text is not None:
                text_parts.append(str(text))
        if not text_parts:
            return {}
        if len(text_parts) == 1:
            return _decode_json_text(text_parts[0])
        return [_decode_json_text(text) for text in text_parts]

    def ping(self) -> dict:
        ping_method = getattr(self._session, "ping", None)
        if callable(ping_method):
            self._loop.run_until_complete(ping_method())
        return {"pong": True}

    def close(self) -> None:
        try:
            if self._session_cm is not None:
                self._loop.run_until_complete(self._session_cm.__aexit__(None, None, None))
        except Exception:
            pass
        try:
            if self._stdio_cm is not None:
                self._loop.run_until_complete(self._stdio_cm.__aexit__(None, None, None))
        except Exception:
            pass
        try:
            self._loop.close()
        except Exception:
            pass
        if self._context_path.exists():
            try:
                self._context_path.unlink()
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class StdioMCPClient:
    def __init__(self, parsed_documents: list[dict] | None = None, session_id: str = "") -> None:
        self._delegate = None
        self.mode = "legacy"

        try:
            self._delegate = OfficialStdioMCPClient(parsed_documents=parsed_documents, session_id=session_id)
            self.mode = "official"
        except Exception:
            self._delegate = LegacyStdioMCPClient(parsed_documents=parsed_documents, session_id=session_id)
            self.mode = "legacy"

    def list_resources(self) -> list[str]:
        return self._delegate.list_resources()

    def list_tools(self) -> list[str]:
        return self._delegate.list_tools()

    def read_resource(self, uri: str) -> Any:
        return self._delegate.read_resource(uri)

    def call_tool(self, name: str, arguments: Optional[dict] = None) -> Any:
        return self._delegate.call_tool(name, arguments)

    def ping(self) -> dict:
        return self._delegate.ping()

    def close(self) -> None:
        self._delegate.close()
