from __future__ import annotations

from typing import List, Optional, TypedDict

from .agent import answer_with_agent
from .chains import answer_with_langchain
from .router import route_question


class WorkflowState(TypedDict, total=False):
    question: str
    parsed_documents: List[dict]
    session_id: str
    route: dict
    result: dict
    tried_rag: bool
    tried_agent: bool


def _route_node(state: WorkflowState) -> WorkflowState:
    return {
        **state,
        "route": route_question(state["question"]),
        "tried_rag": False,
        "tried_agent": False,
    }


def _rag_node(state: WorkflowState) -> WorkflowState:
    rag_result = answer_with_langchain(state["question"], state["parsed_documents"])
    if rag_result and rag_result.get("answer"):
        route = state["route"]
        reason = route["reason"]
        if state.get("tried_agent"):
            reason = "Agent did not return an answer; RAG handled the question."
        return {
            **state,
            "tried_rag": True,
            "result": {
                "mode": "rag",
                "intent": route["intent"],
                "reason": reason,
                "answer": rag_result["answer"],
                "sources": rag_result.get("sources", []),
                "workflow_engine": "langgraph",
            },
        }
    return {**state, "tried_rag": True}


def _agent_node(state: WorkflowState) -> WorkflowState:
    agent_result = answer_with_agent(
        state["question"],
        state["parsed_documents"],
        session_id=state.get("session_id", ""),
    )
    if agent_result and agent_result.get("answer"):
        route = state["route"]
        reason = agent_result.get("fallback_reason") or route["reason"]
        if state.get("tried_rag") and not agent_result.get("fallback_reason"):
            reason = "RAG did not return an answer; Agent tools handled the question."
        return {
            **state,
            "tried_agent": True,
            "result": {
                "mode": "agent",
                "intent": route["intent"],
                "reason": reason,
                "answer": agent_result["answer"],
                "steps": agent_result.get("steps", []),
                "react_trace": agent_result.get("react_trace", ""),
                "workflow_engine": "langgraph",
            },
        }
    return {**state, "tried_agent": True}


def _choose_first_node(state: WorkflowState) -> str:
    return "rag" if state["route"]["route"] == "rag" else "agent"


def _after_rag_node(state: WorkflowState) -> str:
    if state.get("result", {}).get("answer"):
        return "end"
    return "agent" if not state.get("tried_agent") else "end"


def _after_agent_node(state: WorkflowState) -> str:
    if state.get("result", {}).get("answer"):
        return "end"
    return "rag" if not state.get("tried_rag") else "end"


def answer_with_langgraph_workflow(
    question: str,
    parsed_documents: List[dict],
    session_id: str = "",
) -> Optional[dict]:
    if not parsed_documents:
        return None

    try:
        from langgraph.graph import END, StateGraph
    except ModuleNotFoundError:
        return None

    graph = StateGraph(WorkflowState)
    graph.add_node("route", _route_node)
    graph.add_node("rag", _rag_node)
    graph.add_node("agent", _agent_node)
    graph.set_entry_point("route")
    graph.add_conditional_edges("route", _choose_first_node, {"rag": "rag", "agent": "agent"})
    graph.add_conditional_edges("rag", _after_rag_node, {"agent": "agent", "end": END})
    graph.add_conditional_edges("agent", _after_agent_node, {"rag": "rag", "end": END})

    compiled = graph.compile()
    final_state = compiled.invoke(
        {
            "question": question,
            "parsed_documents": parsed_documents,
            "session_id": session_id,
        }
    )
    result = final_state.get("result", {})
    return result if result.get("answer") else None
