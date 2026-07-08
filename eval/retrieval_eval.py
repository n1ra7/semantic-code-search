"""Measure retrieval quality against a labeled dataset.

For each query we retrieve chunks, reduce them to an ordered list of unique files, and
score them against the ground-truth relevant files. Reports per-query results and the
mean of each metric. Run against a live index:

    python -m eval.retrieval_eval            # uses the real Searcher (needs Qdrant + model)

`run_eval` takes an injected searcher so the metrics can also be exercised offline in tests.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import List

from eval.metrics import (
    dedupe_preserving_order,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

DATASET = Path(__file__).parent / "dataset.jsonl"
METRIC_KEYS = ["precision@k", "recall@k", "hit@k", "mrr", "ndcg@k"]


def load_dataset(path: Path = DATASET) -> List[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_eval(searcher, dataset: List[dict], k: int = 5):
    rows = []
    for ex in dataset:
        # Retrieve generously, then collapse to unique files, best-first.
        hits = searcher.search(ex["query"], limit=k * 4)
        retrieved = dedupe_preserving_order(h["path"] for h in hits)
        relevant = set(ex["relevant"])
        rows.append(
            {
                "query": ex["query"],
                "precision@k": precision_at_k(retrieved, relevant, k),
                "recall@k": recall_at_k(retrieved, relevant, k),
                "hit@k": hit_at_k(retrieved, relevant, k),
                "mrr": reciprocal_rank(retrieved, relevant),
                "ndcg@k": ndcg_at_k(retrieved, relevant, k),
                "top": retrieved[:k],
            }
        )
    agg = {m: mean(r[m] for r in rows) for m in METRIC_KEYS} if rows else {}
    return rows, agg


def main() -> None:
    from semantic_index.search import Searcher

    dataset = load_dataset()
    rows, agg = run_eval(Searcher(), dataset)
    for r in rows:
        flag = "✓" if r["hit@k"] else "✗"
        print(f"{flag} mrr={r['mrr']:.2f} ndcg={r['ndcg@k']:.2f}  {r['query']}")
        print(f"    top: {r['top']}")
    print("\n== means over dataset ==")
    for m in METRIC_KEYS:
        print(f"  {m:<12} {agg.get(m, 0.0):.3f}")


if __name__ == "__main__":
    main()
