from __future__ import annotations

import re
from typing import Iterable


UNCERTAINTY_PHRASES = [
    "可能",
    "大概",
    "推测",
    "疑似",
    "或许",
    "不确定",
    "无法确认",
    "未明确说明",
]


def _normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip().lower()


def _extract_keywords(text: str) -> set[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return set()

    english_tokens = {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{1,}", normalized)
        if len(token) >= 2
    }
    chinese_tokens = {
        token
        for token in re.findall(r"[\u4e00-\u9fff]{2,6}", normalized)
        if len(token) >= 2
    }
    numeric_tokens = set(re.findall(r"\d+(?:\.\d+)?[%a-zA-Z]*", normalized))
    return english_tokens | chinese_tokens | numeric_tokens


def _collect_evidence_texts(
    sources: list[dict | str] | None = None,
    steps: list[dict] | None = None,
) -> list[str]:
    texts: list[str] = []

    for item in sources or []:
        if isinstance(item, str):
            texts.append(item)
            continue
        texts.append(str(item.get("content", "")))
        texts.append(str(item.get("description", "")))
        texts.append(str(item.get("ocr_text", "")))
        texts.append(str(item.get("source", "")))

    for step in steps or []:
        texts.append(str(step.get("observation", "")))
        texts.append(str(step.get("tool", "")))

    return [text for text in texts if _normalize_text(text)]


def validate_answer(
    answer: str,
    *,
    route_type: str,
    sources: list[dict | str] | None = None,
    steps: list[dict] | None = None,
) -> dict:
    answer_text = _normalize_text(answer)
    evidence_texts = _collect_evidence_texts(sources=sources, steps=steps)
    evidence_keywords = _extract_keywords(" ".join(evidence_texts))
    answer_keywords = _extract_keywords(answer_text)
    overlap = answer_keywords & evidence_keywords

    score = 0
    reasons: list[str] = []
    warnings: list[str] = []

    if evidence_texts:
        score += 2
        reasons.append("存在可追溯的来源或工具观测结果。")
    else:
        warnings.append("当前答案缺少可见证据支撑。")

    if len(evidence_texts) >= 2:
        score += 1
        reasons.append("答案参考了多个证据片段。")

    if overlap:
        score += 2
        reasons.append(f"答案与证据存在关键词重合（{min(len(overlap), 5)} 个以上关键项）。")
    else:
        warnings.append("答案与证据的关键词重合较少，需谨慎核对。")

    if route_type == "rag" and sources:
        score += 1
        reasons.append("答案来自检索增强链路。")
    if route_type == "agent" and steps:
        score += 1
        reasons.append("答案经过 Agent 工具调用验证。")

    uncertainty_hits = [phrase for phrase in UNCERTAINTY_PHRASES if phrase in answer]
    if uncertainty_hits:
        warnings.append("答案包含不确定性表达：" + "、".join(uncertainty_hits[:3]))

    if len(answer_text) > 300 and len(evidence_texts) <= 1:
        warnings.append("答案较长，但证据较少，可能存在扩写风险。")

    if score >= 5 and not warnings:
        confidence = "高"
    elif score >= 3:
        confidence = "中"
    else:
        confidence = "低"

    grounded = bool(evidence_texts) and bool(overlap)
    if not grounded and confidence != "低":
        confidence = "中"

    return {
        "confidence": confidence,
        "grounded": grounded,
        "score": score,
        "reasons": reasons[:3],
        "warnings": warnings[:3],
    }


def format_validation_result(result: dict) -> str:
    confidence = result.get("confidence", "低")
    grounded = "是" if result.get("grounded") else "否"
    lines = [
        "### 答案校验",
        f"- 可信度：{confidence}",
        f"- 是否有证据支撑：{grounded}",
    ]

    reasons = result.get("reasons", [])
    if reasons:
        lines.append(f"- 支撑依据：{'；'.join(reasons)}")

    warnings = result.get("warnings", [])
    if warnings:
        lines.append(f"- 风险提示：{'；'.join(warnings)}")

    return "\n".join(lines)
