from __future__ import annotations

from typing import Dict


TEXT_FACT_KEYWORDS = [
    "\u591a\u5c11",
    "\u51e0",
    "\u53c2\u6570",
    "\u529f\u7387",
    "\u7535\u538b",
    "\u7535\u6d41",
    "\u65f6\u95f4",
    "\u7ed3\u8bba",
    "\u6307\u6807",
    "\u6570\u636e",
    "\u5185\u5bb9",
    "\u662f\u4ec0\u4e48",
]

IMAGE_KEYWORDS = [
    "\u56fe",
    "\u56fe\u7247",
    "\u56fe\u8868",
    "\u66f2\u7ebf",
    "\u67f1\u72b6\u56fe",
    "\u997c\u56fe",
    "\u6d41\u7a0b\u56fe",
    "\u793a\u610f\u56fe",
    "\u7167\u7247",
    "\u622a\u56fe",
]

OCR_KEYWORDS = [
    "\u5199\u4e86\u4ec0\u4e48",
    "\u6587\u5b57",
    "\u8bc6\u522b",
    "ocr",
    "\u8bfb\u51fa",
    "\u63d0\u53d6\u6587\u5b57",
]

SUMMARY_KEYWORDS = [
    "\u603b\u7ed3",
    "\u6458\u8981",
    "\u6982\u8ff0",
    "\u6982\u62ec",
    "\u4e3b\u65e8",
    "\u4e3b\u8981\u5185\u5bb9",
]

CALCULATION_KEYWORDS = [
    "\u8ba1\u7b97",
    "\u6c42\u548c",
    "\u5408\u8ba1",
    "\u52a0\u8d77\u6765",
    "\u5e73\u5747",
    "\u6bd4\u4f8b",
    "\u767e\u5206\u6bd4",
    "\u589e\u957f\u7387",
    "\u4e58",
    "\u9664",
    "+",
    "-",
    "*",
    "/",
    "=",
]

WEB_SEARCH_KEYWORDS = [
    "\u6700\u65b0",
    "\u5b98\u7f51",
    "\u5b98\u65b9\u6587\u6863",
    "\u5916\u90e8\u8d44\u6599",
    "\u8054\u7f51",
    "\u641c\u7d22",
    "\u67e5\u4e00\u4e0b",
    "\u7f51\u4e0a",
    "api \u6587\u6863",
    "web search",
]


def route_question(question: str) -> Dict[str, str]:
    normalized = question.strip().lower()

    if any(keyword in normalized for keyword in OCR_KEYWORDS):
        return {
            "route": "agent",
            "intent": "ocr",
            "reason": "Question asks for image text recognition, so Agent OCR tools are preferred.",
        }

    if any(keyword in normalized for keyword in IMAGE_KEYWORDS):
        return {
            "route": "agent",
            "intent": "image",
            "reason": "Question mentions images, charts, or visual content, so Agent tools are preferred.",
        }

    if any(keyword in normalized for keyword in CALCULATION_KEYWORDS):
        return {
            "route": "agent",
            "intent": "calculation",
            "reason": "Question involves explicit calculation, so Agent calculator tools are preferred.",
        }

    if any(keyword in normalized for keyword in WEB_SEARCH_KEYWORDS):
        return {
            "route": "agent",
            "intent": "web_search",
            "reason": "Question asks for latest or external information, so Agent web search tools are preferred.",
        }

    if any(keyword in normalized for keyword in SUMMARY_KEYWORDS):
        return {
            "route": "rag",
            "intent": "summary",
            "reason": "Question asks for a summary, so RAG is preferred first.",
        }

    if any(keyword in normalized for keyword in TEXT_FACT_KEYWORDS):
        return {
            "route": "rag",
            "intent": "text",
            "reason": "Question asks for textual facts, so RAG is preferred first.",
        }

    return {
        "route": "agent",
        "intent": "unknown",
        "reason": "Question type is unclear, so Agent selects tools first.",
    }
