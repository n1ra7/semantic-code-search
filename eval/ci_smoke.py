"""Deterministic end-to-end eval smoke test for CI (no model download).

Uses a hashing stand-in embedder instead of a downloaded model, so CI gates the whole
retrieval pipeline (chunk -> embed -> Qdrant -> search -> eval) reproducibly, without
depending on Hugging Face downloads. Exits non-zero if metrics fall below a floor.

Run locally or in CI (with a Qdrant service on :6333):
    python -m eval.ci_smoke
"""
from __future__ import annotations

import hashlib
import math
import re
import sys
import time
import urllib.request
from pathlib import Path

from eval.retrieval_eval import load_dataset, run_eval
from semantic_index.config import settings
from semantic_index.indexer import Indexer
from semantic_index.search import Searcher

DIM = 256
_TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")
HIT_FLOOR = 0.35  # hashing embedder on our own repo; a real model scores much higher


class HashEmbedder:
    """Deterministic bag-of-tokens embedder — no model, no network."""

    dim = DIM

    def embed(self, texts):
        out = []
        for text in texts:
            vec = [0.0] * DIM
            for word in _TOK.findall(text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


def _wait_for_qdrant(timeout: int = 60) -> None:
    for _ in range(timeout):
        try:
            urllib.request.urlopen(settings.qdrant_url + "/readyz", timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise SystemExit(f"Qdrant not reachable at {settings.qdrant_url}")


def main() -> int:
    _wait_for_qdrant()
    emb = HashEmbedder()
    Indexer(embedder=emb, collection="ci_smoke", state_db="/tmp/ci_smoke.sqlite").index(".")
    dataset = [ex for ex in load_dataset(Path("eval/dataset.jsonl")) if not ex.get("negative")]
    _, agg = run_eval(Searcher(embedder=emb, collection="ci_smoke"), dataset, k=5)
    print("eval means:", {k: round(v, 3) for k, v in agg.items()})
    if agg["hit@k"] < HIT_FLOOR:
        print(f"FAIL: hit@k {agg['hit@k']:.3f} < floor {HIT_FLOOR}")
        return 1
    print(f"OK: hit@k {agg['hit@k']:.3f} >= floor {HIT_FLOOR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
