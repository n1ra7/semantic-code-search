"""Query side: embed the query, then nearest-neighbour (dense) or RRF-fused hybrid search."""
from __future__ import annotations

from typing import List, Optional

from .config import settings
from .embedder import Embedder, SparseEmbedder
from .store import VectorStore


class Searcher:
    def __init__(self, embedder: Embedder | None = None, sparse_embedder: SparseEmbedder | None = None) -> None:
        self.embedder = embedder or Embedder()
        self.hybrid = sparse_embedder is not None or settings.retrieval_mode == "hybrid"
        self.sparse = sparse_embedder or (SparseEmbedder() if settings.retrieval_mode == "hybrid" else None)
        self.store = VectorStore(dim=self.embedder.dim, hybrid=self.hybrid)

    def search(self, query: str, limit: int = 8, language: Optional[str] = None) -> List[dict]:
        vector = self.embedder.embed([query])[0]
        sparse_vector = self.sparse.embed([query])[0] if self.sparse else None
        return self.store.search(vector, limit=limit, language=language, sparse_vector=sparse_vector)
