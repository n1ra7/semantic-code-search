"""Query side: embed the query with the same model, then nearest-neighbour in Qdrant."""
from __future__ import annotations

from typing import List, Optional

from .embedder import Embedder
from .store import VectorStore


class Searcher:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or Embedder()
        self.store = VectorStore(dim=self.embedder.dim)

    def search(self, query: str, limit: int = 8, language: Optional[str] = None) -> List[dict]:
        vector = self.embedder.embed([query])[0]
        return self.store.search(vector, limit=limit, language=language)
