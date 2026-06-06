from __future__ import annotations


def build_rag_prompt():
    try:
        from langchain_core.prompts import ChatPromptTemplate
    except ModuleNotFoundError:
        return None

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个文档问答助手。请仅根据检索到的上下文回答问题。"
                "如果上下文不足以支持结论，请明确说明信息不足。"
                "回答要简洁，并尽量保留来源事实。",
            ),
            (
                "human",
                "问题：{input}\n\n"
                "上下文：\n{context}\n\n"
                "请用中文回答，并在必要时结合来源信息作答。",
            ),
        ]
    )
