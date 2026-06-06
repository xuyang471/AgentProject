from __future__ import annotations

import argparse
import json
import sys

from .server import LocalMCPServer, create_fastmcp_server, load_runtime_context_from_env


def _write_message(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _run_legacy_stdio_server() -> None:
    server = LocalMCPServer()
    _write_message({"status": "ready"})

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            _write_message({"ok": False, "error": "invalid_json"})
            continue

        method = request.get("method", "")
        params = request.get("params", {}) or {}

        if method == "shutdown":
            _write_message({"ok": True, "result": "bye"})
            break

        try:
            result = server.handle_request(method, params)
            _write_message({"ok": True, "result": result})
        except Exception as exc:
            _write_message({"ok": False, "error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--official", action="store_true")
    args, _ = parser.parse_known_args()

    if not args.legacy:
        parsed_documents, session_id = load_runtime_context_from_env()
        fastmcp = create_fastmcp_server(parsed_documents=parsed_documents, session_id=session_id)
        if fastmcp is not None:
            fastmcp.run()
            return

    _run_legacy_stdio_server()


if __name__ == "__main__":
    main()
