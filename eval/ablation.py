"""Retrieval ablation: measure the contribution of each upgrade.

Runs the eval across a sequence of configurations — baseline (line chunks, dense) →
+ AST chunking → + hybrid → + reranking — indexing each into its own collection and
reporting the metrics side by side, so each step's gain is visible.

    python -m eval.ablation /path/to/gitea --dataset eval/dataset_gitea.jsonl

`run_ablation` takes injected components so the orchestration can be exercised offline
in tests; `main` builds the real FastEmbed / cross-encoder models.
"""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from eval.retrieval_eval import METRIC_KEYS, load_dataset, run_eval
from semantic_index.indexer import Indexer
from semantic_index.search import Searcher

# (label, {chunk, hybrid, rerank}) — cumulative: each row adds one upgrade.
CONFIGS = [
    ("baseline (line, dense)", {"chunk": "line", "hybrid": False, "rerank": False}),
    ("+ AST chunking", {"chunk": "ast", "hybrid": False, "rerank": False}),
    ("+ hybrid (BM25+dense)", {"chunk": "ast", "hybrid": True, "rerank": False}),
    ("+ reranking", {"chunk": "ast", "hybrid": True, "rerank": True}),
]


def run_ablation(
    corpus: str,
    dataset: List[dict],
    embedder,
    sparse_embedder=None,
    reranker=None,
    k: int = 5,
    configs=CONFIGS,
):
    rows = []
    for i, (label, cfg) in enumerate(configs):
        se = sparse_embedder if cfg["hybrid"] else None
        rr = reranker if cfg["rerank"] else None
        collection = f"ablation_{i}"
        state = os.path.join(tempfile.gettempdir(), f"ablation_{i}.sqlite")
        if os.path.exists(state):
            os.remove(state)  # fresh state so each config fully re-indexes

        Indexer(
            embedder=embedder,
            sparse_embedder=se,
            chunk_strategy=cfg["chunk"],
            collection=collection,
            state_db=state,
        ).index(corpus)

        searcher = Searcher(embedder=embedder, sparse_embedder=se, reranker=rr, collection=collection)
        _, agg = run_eval(searcher, dataset, k=k)
        rows.append((label, agg))
    return rows


def format_table(rows, keys=METRIC_KEYS) -> str:
    header = "| Config | " + " | ".join(keys) + " |"
    sep = "|" + "---|" * (len(keys) + 1)
    lines = [header, sep]
    for label, agg in rows:
        cells = " | ".join(f"{agg.get(k, 0.0):.3f}" for k in keys)
        lines.append(f"| {label} | {cells} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the retrieval ablation and print a results table.")
    parser.add_argument("corpus", help="Path to the corpus to index (e.g. a cloned Gitea).")
    parser.add_argument("--dataset", default="eval/dataset_gitea.jsonl")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    from semantic_index.embedder import Embedder, SparseEmbedder
    from semantic_index.reranker import Reranker

    dataset = load_dataset(Path(args.dataset))
    positives = [ex for ex in dataset if not ex.get("negative")]  # retrieval metrics need a ground truth
    rows = run_ablation(args.corpus, positives, Embedder(), SparseEmbedder(), Reranker(), k=args.k)
    print(format_table(rows))


if __name__ == "__main__":
    main()
