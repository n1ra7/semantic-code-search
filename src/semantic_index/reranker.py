"""Cross-encoder reranking (second stage of two-stage retrieval).

First-stage retrieval (dense or hybrid) is fast but approximate. A cross-encoder scores
each (query, chunk) pair jointly for much higher precision, at the cost of latency — so we
only rerank a small candidate set (top-N) and keep the best `limit`.
"""
from __future__ import annotations

from typing import List

from .config import settings


class Reranker:
    def __init__(self, model_name: str | None = None) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name or settings.rerank_model
        self.model = TextCrossEncoder(model_name=self.model_name)

    def rerank(self, query: str, documents: List[str]) -> List[float]:
        """Return a relevance score per document (higher = more relevant)."""
        return [float(s) for s in self.model.rerank(query, documents)]


def apply_rerank(hits: List[dict], scores: List[float], limit: int) -> List[dict]:
    """Reorder hits by rerank score (desc), tag each with rerank_score, keep top `limit`.

    Pure function (no model/Qdrant) so the ranking logic is unit-testable.
    """
    ranked = sorted(zip(hits, scores), key=lambda pair: pair[1], reverse=True)
    out: List[dict] = []
    for hit, score in ranked[:limit]:
        enriched = dict(hit)
        enriched["rerank_score"] = float(score)
        out.append(enriched)
    return out
