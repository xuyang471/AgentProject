from __future__ import annotations

import json
import os
from typing import List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_TAVILY_ENDPOINT = "https://api.tavily.com/search"


def _normalize_results(items) -> List[dict]:
    normalized = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "title": str(item.get("title", "") or "").strip(),
                "url": str(item.get("url", "") or "").strip(),
                "snippet": str(item.get("content", "") or item.get("snippet", "") or "").strip(),
            }
        )
    return [item for item in normalized if item["title"] or item["url"] or item["snippet"]]


def search_web(query: str, top_k: int = 5) -> List[dict]:
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    endpoint = os.getenv("TAVILY_SEARCH_URL", DEFAULT_TAVILY_ENDPOINT).strip() or DEFAULT_TAVILY_ENDPOINT
    if not api_key:
        raise RuntimeError("未配置 TAVILY_API_KEY，无法执行联网查询。")

    normalized_query = str(query or "").strip()
    if not normalized_query:
        raise ValueError("联网查询内容不能为空。")

    payload = {
        "api_key": api_key,
        "query": normalized_query,
        "max_results": max(1, min(int(top_k or 5), 10)),
        "search_depth": "basic",
        "include_answer": False,
        "include_images": False,
        "include_raw_content": False,
    }

    request = Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=20) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else str(exc)
        raise RuntimeError(f"联网查询失败，HTTP {exc.code}: {detail[:200]}") from exc
    except URLError as exc:
        raise RuntimeError(f"联网查询失败，网络异常：{exc.reason}") from exc
    except Exception as exc:
        raise RuntimeError(f"联网查询失败：{exc}") from exc

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("联网查询返回结果不是合法 JSON。") from exc

    results = _normalize_results(data.get("results", []))
    return results
