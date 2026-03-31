"""Retrieval quality evaluation helpers."""

from __future__ import annotations

from .models import EvaluationExample
from .retriever import SemanticRetriever


def evaluate_retrieval(
    retriever: SemanticRetriever,
    examples: list[EvaluationExample],
    top_k: int,
) -> dict[str, object]:
    """Evaluates retrieval hit rate, MRR, and keyword recall over labeled examples."""

    if not examples:
        return {"hit_rate": 0.0, "mrr": 0.0, "keyword_recall": 0.0, "details": []}

    hit_count = 0
    reciprocal_rank_sum = 0.0
    keyword_recall_sum = 0.0
    details: list[dict[str, object]] = []

    for example in examples:
        results = retriever.retrieve(query=example.query, top_k=top_k)
        retrieved_sources = [str(result.metadata.get("source", "")) for result in results]

        matched_rank = None
        for index, source in enumerate(retrieved_sources, start=1):
            if any(expected_source in source for expected_source in example.expected_sources):
                matched_rank = index
                break

        if matched_rank is not None:
            hit_count += 1
            reciprocal_rank_sum += 1.0 / matched_rank

        keyword_recall = _compute_keyword_recall(
            retrieved_text=" ".join(result.text.lower() for result in results),
            expected_keywords=example.expected_keywords,
        )
        keyword_recall_sum += keyword_recall

        details.append(
            {
                "query": example.query,
                "retrieved_sources": retrieved_sources,
                "matched_rank": matched_rank,
                "keyword_recall": keyword_recall,
            }
        )

    total_examples = len(examples)
    return {
        "hit_rate": hit_count / total_examples,
        "mrr": reciprocal_rank_sum / total_examples,
        "keyword_recall": keyword_recall_sum / total_examples,
        "details": details,
    }


def _compute_keyword_recall(retrieved_text: str, expected_keywords: list[str]) -> float:
    """Measures how many expected keywords appear in the retrieved context."""

    if not expected_keywords:
        return 1.0

    matches = sum(1 for keyword in expected_keywords if keyword.lower() in retrieved_text)
    return matches / len(expected_keywords)
