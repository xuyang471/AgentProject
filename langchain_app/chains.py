from __future__ import annotations

from pathlib import Path
from typing import List

from .document_builder import build_langchain_documents
from .models import create_qwen_chat_model
from .prompts import build_rag_prompt
from .retrievers import create_hybrid_retriever
from services.text_quality_service import clean_source_excerpt


def answer_with_langchain(question: str, parsed_documents: List[dict]) -> dict | None:
    try:
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
    except ModuleNotFoundError:
        return None

    llm = create_qwen_chat_model()
    if llm is None:
        return None

    documents = build_langchain_documents(parsed_documents)
    if not documents:
        return None

    retriever = create_hybrid_retriever(documents, top_k=3)
    prompt = build_rag_prompt()
    if retriever is None or prompt is None:
        return None

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    result = retrieval_chain.invoke({"input": question})

    contexts = result.get("context", [])
    sources = []
    for doc in contexts:
        metadata = getattr(doc, "metadata", {}) or {}
        image_path = str(metadata.get("image_path", "") or "")
        image_markdown = ""
        if image_path and Path(image_path).exists():
            image_markdown = f"![检索命中的图片]({Path(image_path).resolve()})"

        sources.append(
            {
                "source": metadata.get("source", "未知来源"),
                "file_name": metadata.get("file_name", ""),
                "block_type": metadata.get("block_type", ""),
                "page": metadata.get("page"),
                "content": clean_source_excerpt(getattr(doc, "page_content", "").strip(), limit=800),
                "description": metadata.get("description", ""),
                "ocr_text": metadata.get("ocr_text", ""),
                "image_path": image_path,
                "image_markdown": image_markdown,
            }
        )

    return {
        "answer": str(result.get("answer", "")).strip(),
        "sources": sources,
    }
