from __future__ import annotations

from typing import List

from .text_quality_service import clean_text_artifacts, is_low_quality_text


def _fallback_rank(question: str, items: List[dict], top_k: int) -> List[dict]:
    question_terms = [term for term in question.lower().split() if term.strip()]
    scored = []

    for item in items:
        content_lower = item["content"].lower()
        score = sum(1 for term in question_terms if term in content_lower)
        scored.append({**item, "score": float(score)})

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def retrieve_relevant_blocks(question: str, items: List[dict], top_k: int = 3) -> List[dict]:
    if not items:
        return []

    cleaned_items = []
    for item in items:
        content = clean_text_artifacts(item.get("content", ""))
        if item.get("type") == "text" and is_low_quality_text(content):
            continue
        if content:
            cleaned_items.append({**item, "content": content})

    if not cleaned_items:
        return []

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ModuleNotFoundError:
        return _fallback_rank(question, cleaned_items, top_k)

    corpus = [item["content"] for item in cleaned_items]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
    matrix = vectorizer.fit_transform(corpus)
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, matrix)[0]

    ranked = []
    for item, score in zip(cleaned_items, similarities):
        ranked.append({**item, "score": float(score)})

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]
