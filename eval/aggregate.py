"""File-level score aggregation over chunk hits.

The default eval collapses chunk hits to files best-first — i.e. a file is ranked by its
single best chunk (max pooling). With fine-grained (AST) chunks a relevant file often has
MANY moderately-scoring chunks instead of one big one; summing the top-N chunk scores per
file rewards that accumulated evidence and makes chunking strategies comparable.
"""
from __future__ import annotations

from typing import List


def aggregate_hits_by_file(hits: List[dict], top_n: int = 3) -> List[dict]:
    """Collapse chunk hits into one hit per file, scored by the sum of its top-N chunk
    scores (rerank_score preferred over score when present). Sorted best-first."""

    def score_of(hit: dict) -> float:
        return float(hit.get("rerank_score", hit.get("score", 0.0)))

    by_file: dict[str, List[float]] = {}
    for hit in hits:
        by_file.setdefault(hit["path"], []).append(score_of(hit))
    ranked = sorted(
        ((path, sum(sorted(scores, reverse=True)[:top_n])) for path, scores in by_file.items()),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return [{"path": path, "score": score} for path, score in ranked]


class FileAggSearcher:
    """Wrap a Searcher: fetch a deeper chunk pool, return file-level aggregated hits."""

    def __init__(self, inner, top_n: int = 3, fetch: int = 40) -> None:
        self.inner = inner
        self.top_n = top_n
        self.fetch = fetch

    def search(self, query: str, limit: int = 8, language=None) -> List[dict]:
        hits = self.inner.search(query, limit=self.fetch, language=language)
        return aggregate_hits_by_file(hits, top_n=self.top_n)[:limit]
