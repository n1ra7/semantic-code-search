"""Standard information-retrieval metrics, as pure functions (easy to unit-test).

All operate on `retrieved` — an ordered list of item ids (here, file paths), best
first — and `relevant` — the set of ids that count as correct for that query.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Set


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r in relevant) / len(top)


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = retrieved[:k]
    return sum(1 for r in relevant if r in top) / len(relevant)


def hit_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """1.0 if any relevant item appears in the top-k, else 0.0."""
    return 1.0 if any(r in relevant for r in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """1 / rank of the first relevant hit (0 if none)."""
    for i, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Normalized discounted cumulative gain with binary relevance."""
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(retrieved[:k], start=1) if r in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def dedupe_preserving_order(items: Iterable[str]) -> List[str]:
    seen: List[str] = []
    for it in items:
        if it not in seen:
            seen.append(it)
    return seen
