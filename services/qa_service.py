from __future__ import annotations

from typing import List

from langchain_app.workflow import answer_with_workflow

from .answer_validation_service import format_validation_result, validate_answer
from .llm_service import answer_from_context
from .retrieval_service import retrieve_relevant_blocks
from .state_service import save_agent_run, save_qa_record
from .text_quality_service import clean_source_excerpt


def _truncate_text(text: str, limit: int = 220) -> str:
    return clean_source_excerpt(text, limit=limit)


def _format_rag_sources(sources: List[dict | str]) -> str:
    sections = []
    for index, item in enumerate(sources, start=1):
        if isinstance(item, str):
            sections.append("\n".join([f"#### 来源{index}", f"- 位置：{item}"]))
            continue

        block_type = item.get("block_type", "")
        source = item.get("source", "未知来源")
        page = item.get("page")
        title = f"#### 来源{index}"
        meta_line = f"- 位置：{source}" + (f"（第 {page} 页）" if page else "")

        if block_type == "image":
            description = item.get("description", "").strip() or _truncate_text(item.get("content", ""))
            image_markdown = item.get("image_markdown", "")
            section = [title, meta_line, "- 类型：图片", f"- 说明：{description}"]
            if image_markdown:
                section.append(image_markdown)
            sections.append("\n".join(section))
        elif block_type == "table":
            content = _truncate_text(item.get("content", ""), limit=300)
            sections.append("\n".join([title, meta_line, "- 类型：表格", f"- 内容摘录：{content}"]))
        else:
            content = _truncate_text(item.get("content", ""), limit=260)
            sections.append("\n".join([title, meta_line, "- 类型：文本", f"- 内容摘录：{content}"]))

    return "\n\n".join(sections)


def _persist_workflow_result(session_id: str | None, question: str, workflow_result: dict) -> None:
    if not session_id:
        return

    mode = workflow_result.get("mode", "unknown")
    answer = workflow_result.get("answer", "")
    intent = workflow_result.get("intent", "")

    if mode == "agent":
        steps = workflow_result.get("steps", [])
        save_agent_run(
            session_id=session_id,
            question=question,
            status="COMPLETED",
            intent=intent,
            route=mode,
            steps=steps,
            final_answer=answer,
        )
        save_qa_record(
            session_id=session_id,
            question=question,
            answer=answer,
            route_type=mode,
            sources=steps,
        )
    else:
        sources = workflow_result.get("sources", [])
        save_qa_record(
            session_id=session_id,
            question=question,
            answer=answer,
            route_type=mode,
            sources=sources,
        )


def answer_question(question: str, parsed_documents: List[dict], session_id: str | None = None) -> str:
    workflow_result = answer_with_workflow(question, parsed_documents, session_id=session_id or "")
    if workflow_result and workflow_result.get("answer"):
        _persist_workflow_result(session_id, question, workflow_result)
        route_reason = workflow_result.get("reason", "")
        if workflow_result.get("mode") == "agent":
            steps = workflow_result.get("steps", [])
            react_trace = str(workflow_result.get("react_trace", "") or "").strip()
            validation_block = format_validation_result(
                validate_answer(
                    workflow_result["answer"],
                    route_type="agent",
                    steps=steps,
                )
            )
            reason_block = f"### 路由说明\n{route_reason}\n\n" if route_reason else ""
            if react_trace:
                return (
                    f"{reason_block}### 回答\n{workflow_result['answer']}\n\n"
                    f"{validation_block}\n\n### ReAct 轨迹\n```text\n{react_trace}\n```"
                )
            if steps:
                step_lines = "\n".join(
                    f"- {step.get('tool', 'unknown_tool')}: {str(step.get('input', {}))}"
                    for step in steps
                )
                return (
                    f"{reason_block}### 回答\n{workflow_result['answer']}\n\n"
                    f"{validation_block}\n\n### Agent 工具轨迹\n{step_lines}"
                )
            return f"{reason_block}### 回答\n{workflow_result['answer']}\n\n{validation_block}"

        sources = workflow_result.get("sources", [])
        source_lines = _format_rag_sources(sources) if sources else ""
        validation_block = format_validation_result(
            validate_answer(
                workflow_result["answer"],
                route_type="rag",
                sources=sources,
            )
        )
        reason_block = f"### 路由说明\n{route_reason}\n\n" if route_reason else ""
        caution_block = (
            "### 结果说明\n以下回答基于当前检索到的文档片段生成，请优先参考下方来源；"
            "若来源不足或不完整，回答可能存在局限。\n\n"
        )
        if source_lines:
            return (
                f"{reason_block}{caution_block}### 回答\n{workflow_result['answer']}\n\n"
                f"{validation_block}\n\n"
                f"### 检索来源\n{source_lines}"
            )
        return (
            f"{reason_block}### 结果说明\n当前未返回明确来源，请谨慎参考以下回答。\n\n"
            f"### 回答\n{workflow_result['answer']}\n\n{validation_block}"
        )

    items = []

    for document in parsed_documents:
        for block in document["blocks"]:
            if not block["content"].strip():
                continue
            items.append(
                {
                    "content": block["content"],
                    "source": block["source"],
                    "type": block["type"],
                }
            )

    top_candidates = retrieve_relevant_blocks(question, items, top_k=3)

    if not top_candidates:
        return "未找到相关内容。"

    context = "\n\n".join(
        f"来源：{item['source']}\n相关度：{item['score']:.3f}\n内容：{item['content']}"
        for item in top_candidates
    )
    answer = answer_from_context(question, context)
    validation_block = format_validation_result(
        validate_answer(
            answer,
            route_type="fallback",
            sources=top_candidates,
        )
    )

    if session_id:
        save_qa_record(
            session_id=session_id,
            question=question,
            answer=answer,
            route_type="fallback",
            sources=top_candidates,
        )

    return f"### 回答\n{answer}\n\n{validation_block}"
