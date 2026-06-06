from __future__ import annotations

import json
from typing import List, Optional

from .models import create_qwen_chat_model
from .router import route_question
from .tools import create_document_tools


def _tool_map(tools: list) -> dict:
    return {getattr(tool, "name", ""): tool for tool in tools}


def _invoke_tool(tools: list, name: str, arguments: Optional[dict] = None):
    tool = _tool_map(tools).get(name)
    if tool is None:
        return ""
    try:
        return tool.invoke(arguments or {})
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


def _loads_json(value) -> object:
    if isinstance(value, (dict, list)):
        if isinstance(value, dict) and "ok" in value:
            return value.get("data") if value.get("ok") else value
        return value
    try:
        parsed = json.loads(str(value))
        if isinstance(parsed, dict) and "ok" in parsed:
            return parsed.get("data") if parsed.get("ok") else parsed
        return parsed
    except Exception:
        return str(value)


def _trace(tool_name: str, tool_input: dict, observation) -> dict:
    return {
        "tool": tool_name,
        "input": tool_input,
        "observation": observation if isinstance(observation, str) else json.dumps(observation, ensure_ascii=False),
    }


def _thought_for_tool(tool_name: str, tool_input: dict, question: str) -> str:
    if tool_name == "search_document_blocks":
        block_type = str(tool_input.get("block_type", "") or "").strip()
        if block_type == "image":
            return "我需要先在当前文档中定位与图片或图表相关的内容块。"
        if block_type == "table":
            return "我需要先检索与表格相关的文档证据。"
        return "我需要先检索当前文档中与问题最相关的证据。"
    if tool_name == "run_ocr_on_image":
        return "我需要读取图片中的文字内容，确认图中具体写了什么。"
    if tool_name == "analyze_image_with_mllm":
        return "我需要进一步理解图片或图表的语义含义。"
    if tool_name == "analyze_image":
        return "我需要进一步理解图片或图表的语义含义。"
    if tool_name == "get_report_summary":
        return "这个问题偏向概括总结，我先读取已有报告摘要。"
    if tool_name == "calculator":
        return "这个问题需要精确数值计算，我先调用计算器避免心算误差。"
    if tool_name == "web_search":
        return "这个问题需要外部或最新资料，我先联网查询补充信息。"
    return f"我需要调用工具 {tool_name} 来补充回答 {question} 所需的信息。"


def _format_observation_for_react(observation) -> str:
    if isinstance(observation, str):
        text = observation.strip()
    else:
        try:
            text = json.dumps(observation, ensure_ascii=False)
        except Exception:
            text = str(observation)
    if len(text) > 600:
        return text[:600] + "..."
    return text


def _build_react_trace(question: str, steps: list[dict], final_answer: str) -> str:
    if not steps:
        if not final_answer:
            return ""
        return "\n".join(
            [
                f"Thought: 我先直接根据当前已有信息回答问题：{question}",
                f"Final Answer: {final_answer}",
            ]
        )

    lines = []
    for step in steps:
        tool_name = str(step.get("tool", "unknown_tool") or "unknown_tool")
        tool_input = step.get("input", {})
        lines.append(f"Thought: {_thought_for_tool(tool_name, tool_input if isinstance(tool_input, dict) else {}, question)}")
        lines.append(f"Action: {tool_name}")
        lines.append(
            "Action Input: "
            + (
                json.dumps(tool_input, ensure_ascii=False)
                if isinstance(tool_input, dict)
                else str(tool_input)
            )
        )
        lines.append(f"Observation: {_format_observation_for_react(step.get('observation', ''))}")
    if final_answer:
        lines.append(f"Final Answer: {final_answer}")
    return "\n".join(lines)


def _first_match_source(search_payload) -> str:
    matches = _loads_json(search_payload)
    if isinstance(matches, list) and matches:
        return str(matches[0].get("source", "") or "")
    return ""


def _format_search_answer(search_payload) -> str:
    matches = _loads_json(search_payload)
    if not isinstance(matches, list) or not matches:
        return ""

    lines = ["Agent 已调用文档检索工具，找到以下依据："]
    for index, item in enumerate(matches[:3], start=1):
        source = item.get("source", "未知来源")
        content = str(item.get("content", "")).replace("\n", " ").strip()
        if len(content) > 260:
            content = content[:260] + "..."
        lines.append(f"{index}. 来源：{source}；内容：{content}")
    return "\n".join(lines)


def _format_image_answer(payload, *, ocr_only: bool = False) -> str:
    data = _loads_json(payload)
    if not isinstance(data, dict) or data.get("error"):
        return ""

    source = data.get("source", "未知图片")
    ocr_text = str(data.get("ocr_text", "") or "").strip()
    description = str(data.get("description", "") or "").strip()

    if ocr_only:
        if not ocr_text:
            return f"Agent 已调用 OCR 工具，但在 {source} 中没有识别到清晰文字。"
        return f"Agent 已调用 OCR 工具。来源：{source}\n识别文字：{ocr_text}"

    parts = [f"Agent 已调用图片分析工具。来源：{source}"]
    if description:
        parts.append(f"图片说明：{description}")
    if ocr_text:
        parts.append(f"OCR 文字：{ocr_text}")
    return "\n".join(parts) if len(parts) > 1 else ""


def _format_calculation_answer(payload) -> str:
    data = _loads_json(payload)
    if not isinstance(data, dict) or data.get("error"):
        return ""
    expression = str(data.get("expression", "")).strip()
    result = data.get("result")
    if expression:
        return f"Agent 已调用计算器工具。\n表达式：{expression}\n结果：{result}"
    return f"Agent 已调用计算器工具。\n结果：{result}"


def _format_web_search_answer(payload) -> str:
    data = _loads_json(payload)
    if not isinstance(data, list) or not data:
        return ""

    lines = ["Agent 已调用联网查询工具，找到以下外部资料："]
    for index, item in enumerate(data[:3], start=1):
        title = str(item.get("title", "") or "").strip() or "未命名结果"
        url = str(item.get("url", "") or "").strip()
        snippet = str(item.get("snippet", "") or "").replace("\n", " ").strip()
        if len(snippet) > 220:
            snippet = snippet[:220] + "..."
        line = f"{index}. 标题：{title}"
        if url:
            line += f"；链接：{url}"
        if snippet:
            line += f"；摘要：{snippet}"
        lines.append(line)
    return "\n".join(lines)


def _answer_with_tool_plan(
    question: str,
    parsed_documents: List[dict],
    session_id: str = "",
    failure_reason: str = "",
    tools: Optional[list] = None,
) -> Optional[dict]:
    if not parsed_documents:
        return None

    tools = tools or create_document_tools(parsed_documents, session_id=session_id)
    if not tools:
        return None

    route = route_question(question)
    intent = route.get("intent", "unknown")
    steps = []

    if intent == "summary":
        tool_input = {"session_id": session_id}
        observation = _invoke_tool(tools, "get_report_summary", tool_input)
        steps.append(_trace("get_report_summary", tool_input, observation))
        summary_payload = _loads_json(observation)
        if isinstance(summary_payload, dict):
            answer = str(summary_payload.get("summary") or summary_payload.get("content_preview") or "").strip()
        else:
            answer = str(summary_payload).strip()
        if answer and answer != "{}":
            prefix = "Agent 工具调用稳定性兜底："
            final_answer = f"{prefix}\n{answer}"
            return {
                "answer": final_answer,
                "steps": steps,
                "fallback_reason": failure_reason,
                "react_trace": _build_react_trace(question, steps, final_answer),
            }

    if intent == "calculation":
        tool_input = {"expression": question}
        observation = _invoke_tool(tools, "calculator", tool_input)
        steps.append(_trace("calculator", tool_input, observation))
        answer = _format_calculation_answer(observation)
        if answer:
            return {
                "answer": answer,
                "steps": steps,
                "fallback_reason": failure_reason,
                "react_trace": _build_react_trace(question, steps, answer),
            }

    if intent == "web_search":
        tool_input = {"query": question, "top_k": 5}
        observation = _invoke_tool(tools, "web_search", tool_input)
        steps.append(_trace("web_search", tool_input, observation))
        answer = _format_web_search_answer(observation)
        if answer:
            return {
                "answer": answer,
                "steps": steps,
                "fallback_reason": failure_reason,
                "react_trace": _build_react_trace(question, steps, answer),
            }

    block_type = "image" if intent in {"image", "ocr"} else ""
    search_input = {"question": question, "block_type": block_type}
    search_observation = _invoke_tool(tools, "search_document_blocks", search_input)
    steps.append(_trace("search_document_blocks", search_input, search_observation))

    if intent in {"image", "ocr"}:
        source = _first_match_source(search_observation)
        if source:
            image_tool = "run_ocr_on_image" if intent == "ocr" else "analyze_image_with_mllm"
            image_input = {"source": source}
            image_observation = _invoke_tool(tools, image_tool, image_input)
            steps.append(_trace(image_tool, image_input, image_observation))
            answer = _format_image_answer(image_observation, ocr_only=intent == "ocr")
            if answer:
                return {
                    "answer": answer,
                    "steps": steps,
                    "fallback_reason": failure_reason,
                    "react_trace": _build_react_trace(question, steps, answer),
                }

    answer = _format_search_answer(search_observation)
    if answer:
        return {
            "answer": answer,
            "steps": steps,
            "fallback_reason": failure_reason,
            "react_trace": _build_react_trace(question, steps, answer),
        }

    return None


def answer_with_agent(question: str, parsed_documents: List[dict], session_id: str = "") -> Optional[dict]:
    tools = create_document_tools(parsed_documents, session_id=session_id)
    if not tools:
        return None

    try:
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    except ModuleNotFoundError:
        return _answer_with_tool_plan(question, parsed_documents, session_id, "LangChain Agent 依赖不可用，已改用确定性工具计划。", tools)

    llm = create_qwen_chat_model()
    if llm is None:
        return _answer_with_tool_plan(question, parsed_documents, session_id, "未检测到可用 Qwen Chat 配置，已改用确定性工具计划。", tools)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个图文文档问答 Agent。"
                "你当前接入了一层本地 MCP 兼容服务，可以通过 MCP 资源读取会话与报告，通过 MCP 工具执行检索、OCR 和图片分析。"
                "规则："
                "1. 回答前先判断是否需要读取 MCP 资源。"
                "2. 文本事实类问题优先使用 search_document_blocks。"
                "3. 总结概括类问题可先读取 read_report_resource 或 get_report_summary。"
                "4. 遇到需要精确数值推导、求和、比例、增长率或数学表达式计算时，优先调用 calculator。"
                "5. 当用户明确要求最新信息、官网资料或外部背景知识，且文档证据不足时，可调用 web_search。"
                "6. 图片文字类问题优先调用 run_ocr_on_image。"
                "7. 图片语义、图表含义类问题优先调用 analyze_image_with_mllm。"
                "8. 只根据 MCP 资源或工具返回的内容作答，不要编造。"
                "9. 最终答案用简洁中文输出，并尽量说明依据来源。"
            ),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    try:
        agent = create_tool_calling_agent(llm, tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            return_intermediate_steps=True,
            max_iterations=6,
            handle_parsing_errors=True,
        )
        result = executor.invoke({"input": question})
    except Exception as exc:
        return _answer_with_tool_plan(question, parsed_documents, session_id, f"LLM 工具调用异常：{exc}", tools)

    output = str(result.get("output", "")).strip()
    intermediate_steps = result.get("intermediate_steps", [])
    tool_traces = []

    for step in intermediate_steps:
        try:
            action, observation = step
            tool_name = getattr(action, "tool", "unknown_tool")
            tool_input = getattr(action, "tool_input", {})
            tool_traces.append(
                {
                    "tool": tool_name,
                    "input": tool_input,
                    "observation": str(observation),
                }
            )
        except Exception:
            continue

    if output:
        return {
            "answer": output,
            "steps": tool_traces,
            "react_trace": _build_react_trace(question, tool_traces, output),
        }

    return _answer_with_tool_plan(question, parsed_documents, session_id, "LLM Agent 未返回有效文本，已改用确定性工具计划。", tools)
