"""Adversarial evaluation: does the system answer when it should, and decline when it shouldn't?

Positive queries should clear the retrieval-confidence gate (answerable); negative queries
(features the corpus doesn't have) should fall below it, triggering the hallucination fallback.
This catches a system that confidently answers questions it has no evidence for.
"""
from __future__ import annotations

from typing import List

from semantic_index.rag import has_sufficient_evidence


def evaluate_answerability(searcher, dataset: List[dict], min_score: float, k: int = 5) -> dict:
    pos_answered = pos_total = neg_declined = neg_total = 0
    for ex in dataset:
        hits = searcher.search(ex["query"], limit=k)
        answered = has_sufficient_evidence(hits, min_score)
        if ex.get("negative"):
            neg_total += 1
            neg_declined += 0 if answered else 1
        else:
            pos_total += 1
            pos_answered += 1 if answered else 0
    return {
        "positives_answered_rate": pos_answered / pos_total if pos_total else 0.0,
        "negatives_declined_rate": neg_declined / neg_total if neg_total else 0.0,
        "false_answers": neg_total - neg_declined,  # negatives it wrongly tried to answer
    }
