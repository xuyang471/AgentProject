from __future__ import annotations

import re

from .vectorstore import create_faiss_vectorstore


def _preprocess_text(text: str) -> list[str]:
    normalized = text.lower().strip()
    ascii_tokens = re.findall(r"[a-z0-9]+", normalized)
    chinese_chars = [char for char in normalized if "\u4e00" <= char <= "\u9fff"]
    chinese_bigrams = [
        "".join(chinese_chars[index : index + 2])
        for index in range(len(chinese_chars) - 1)
    ]
    tokens = ascii_tokens + chinese_chars + chinese_bigrams
    return tokens or normalized.split()


def create_bm25_retriever(documents: list, top_k: int = 3):
    try:
        from langchain_community.retrievers import BM25Retriever
    except ModuleNotFoundError:
        return None

    if not documents:
        return None

    retriever = BM25Retriever.from_documents(
        documents,
        preprocess_func=_preprocess_text,
    )
    retriever.k = top_k
    return retriever


def create_faiss_retriever(documents: list, top_k: int = 3):
    vectorstore = create_faiss_vectorstore(documents)
    if vectorstore is None:
        return None

    try:
        return vectorstore.as_retriever(search_kwargs={"k": top_k})
    except Exception:
        return None


def create_hybrid_retriever(documents: list, top_k: int = 3):
    bm25_retriever = create_bm25_retriever(documents, top_k=top_k)
    faiss_retriever = create_faiss_retriever(documents, top_k=top_k)

    if bm25_retriever and faiss_retriever:
        try:
            from langchain.retrievers import EnsembleRetriever
        except ModuleNotFoundError:
            return bm25_retriever

        return EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.4, 0.6],
        )

    return faiss_retriever or bm25_retriever
