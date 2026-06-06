from __future__ import annotations

from typing import List, Optional

from .agent import answer_with_agent
from .chains import answer_with_langchain
from .graph_workflow import answer_with_langgraph_workflow
from .router import route_question


def _answer_with_function_workflow(question: str, parsed_documents: List[dict], session_id: str = "") -> Optional[dict]:
    route = route_question(question)

    if route["route"] == "rag":
        rag_result = answer_with_langchain(question, parsed_documents)
        if rag_result and rag_result.get("answer"):
            return {
                "mode": "rag",
                "intent": route["intent"],
                "reason": route["reason"],
                "answer": rag_result["answer"],
                "sources": rag_result.get("sources", []),
                "workflow_engine": "function",
            }

        agent_result = answer_with_agent(question, parsed_documents, session_id=session_id)
        if agent_result and agent_result.get("answer"):
            return {
                "mode": "agent",
                "intent": route["intent"],
                "reason": agent_result.get("fallback_reason") or "RAG did not return an answer; Agent tools handled the question.",
                "answer": agent_result["answer"],
                "steps": agent_result.get("steps", []),
                "react_trace": agent_result.get("react_trace", ""),
                "workflow_engine": "function",
            }
    else:
        agent_result = answer_with_agent(question, parsed_documents, session_id=session_id)
        if agent_result and agent_result.get("answer"):
            return {
                "mode": "agent",
                "intent": route["intent"],
                "reason": agent_result.get("fallback_reason") or route["reason"],
                "answer": agent_result["answer"],
                "steps": agent_result.get("steps", []),
                "react_trace": agent_result.get("react_trace", ""),
                "workflow_engine": "function",
            }

        rag_result = answer_with_langchain(question, parsed_documents)
        if rag_result and rag_result.get("answer"):
            return {
                "mode": "rag",
                "intent": route["intent"],
                "reason": "Agent did not return an answer; RAG handled the question.",
                "answer": rag_result["answer"],
                "sources": rag_result.get("sources", []),
                "workflow_engine": "function",
            }

    return None


def answer_with_workflow(question: str, parsed_documents: List[dict], session_id: str = "") -> Optional[dict]:
    try:
        graph_result = answer_with_langgraph_workflow(question, parsed_documents, session_id=session_id)
    except Exception:
        graph_result = None
    return graph_result or _answer_with_function_workflow(question, parsed_documents, session_id=session_id)
