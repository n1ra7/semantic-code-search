"""Query side: embed the query, retrieve (dense or hybrid), optionally cross-encoder rerank."""
from __future__ import annotations

from typing import List, Optional

from .config import settings
from .embedder import Embedder, SparseEmbedder
from .reranker import Reranker, apply_rerank
from .store import VectorStore


class Searcher:
    def __init__(
        self,
        embedder: Embedder | None = None,
        sparse_embedder: SparseEmbedder | None = None,
        reranker: Reranker | None = None,
        collection: str | None = None,
    ) -> None:
        self.embedder = embedder or Embedder()
        self.hybrid = sparse_embedder is not None or settings.retrieval_mode == "hybrid"
        self.sparse = sparse_embedder or (SparseEmbedder() if settings.retrieval_mode == "hybrid" else None)
        self.store = VectorStore(dim=self.embedder.dim, hybrid=self.hybrid, collection=collection)
        self.reranker = reranker or (Reranker() if settings.rerank_enabled else None)

    def search(self, query: str, limit: int = 8, language: Optional[str] = None) -> List[dict]:
        vector = self.embedder.embed([query])[0]
        sparse_vector = self.sparse.embed([query])[0] if self.sparse else None

        # When reranking, retrieve a larger candidate pool first, then re-order it.
        fetch = max(limit, settings.rerank_candidates) if self.reranker else limit
        hits = self.store.search(vector, limit=fetch, language=language, sparse_vector=sparse_vector)

        if self.reranker and hits:
            scores = self.reranker.rerank(query, [h["text"] for h in hits])
            hits = apply_rerank(hits, scores, limit)
        return hits
