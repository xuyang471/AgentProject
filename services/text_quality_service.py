from __future__ import annotations

import re


ARTIFACT_PATTERNS = [
    re.compile(r"(?:/G\d+){3,}", re.IGNORECASE),
    re.compile(r"(?:/c\d+){3,}", re.IGNORECASE),
    re.compile(r"(?:\b\d\b\s*){12,}"),
    re.compile(r"([0-9])\1{8,}"),
    re.compile(r"([A-Za-z])\1{20,}"),
]


def clean_text_artifacts(text: str) -> str:
    cleaned = str(text or "")
    for pattern in ARTIFACT_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def text_quality_score(text: str) -> float:
    cleaned = clean_text_artifacts(text)
    if not cleaned:
        return 0.0

    meaningful_chars = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", cleaned)
    if not meaningful_chars:
        return 0.0

    artifact_hits = sum(len(pattern.findall(str(text or ""))) for pattern in ARTIFACT_PATTERNS)
    slash_count = cleaned.count("/")
    symbol_count = len(re.findall(r"[^0-9A-Za-z\u4e00-\u9fff\s.,;:!?，。；：！？、（）()\-+%]", cleaned))
    total = max(len(cleaned), 1)
    meaningful_ratio = len(meaningful_chars) / total
    penalty = min(0.7, artifact_hits * 0.15 + slash_count / total + symbol_count / total)
    return max(0.0, min(1.0, meaningful_ratio - penalty))


def is_low_quality_text(text: str, min_score: float = 0.18) -> bool:
    cleaned = clean_text_artifacts(text)
    if len(cleaned) < 8:
        return True
    return text_quality_score(text) < min_score


def clean_source_excerpt(text: str, limit: int = 260) -> str:
    cleaned = clean_text_artifacts(text)
    if not cleaned:
        return "该来源文本噪声较多，暂不展示原文摘录。"
    if is_low_quality_text(cleaned):
        return "该来源文本噪声较多，建议优先查看图片、表格或其他文本来源。"
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + "..."
