from __future__ import annotations

from .embeddings import create_qwen_embeddings


def create_faiss_vectorstore(documents: list):
    try:
        from langchain_community.vectorstores import FAISS
    except ModuleNotFoundError:
        return None

    if not documents:
        return None

    embeddings = create_qwen_embeddings()
    if embeddings is None:
        return None

    try:
        return FAISS.from_documents(documents, embeddings)
    except Exception:
        return None
