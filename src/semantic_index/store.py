"""Qdrant vector store wrapper.

Qdrant is open-source and run locally via Docker (see docker-compose.yml), so this
is the drop-in replacement for managed Qdrant Cloud at zero cost. The collection is
created lazily with the embedder's detected dimension and cosine distance.
"""
from __future__ import annotations

import uuid
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from .chunker import Chunk
from .config import settings


class VectorStore:
    def __init__(self, dim: int, url: str | None = None, collection: str | None = None) -> None:
        self.client = QdrantClient(url=url or settings.qdrant_url)
        self.collection = collection or settings.collection
        self.dim = dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        names = {c.name for c in self.client.get_collections().collections}
        if self.collection not in names:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )
            # Indexed payload field so per-file deletes (re-index) are fast.
            self.client.create_payload_index(
                collection_name=self.collection, field_name="path", field_schema="keyword"
            )

    def delete_by_path(self, path: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(must=[FieldCondition(key="path", match=MatchValue(value=path))]),
        )

    def upsert(self, vectors: List[List[float]], chunks: List[Chunk]) -> None:
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "path": ch.path,
                    "language": ch.language,
                    "start_line": ch.start_line,
                    "end_line": ch.end_line,
                    "text": ch.text,
                },
            )
            for vec, ch in zip(vectors, chunks)
        ]
        if points:
            self.client.upsert(collection_name=self.collection, points=points)

    def search(self, vector: List[float], limit: int = 8, language: Optional[str] = None) -> List[dict]:
        query_filter = None
        if language:
            query_filter = Filter(must=[FieldCondition(key="language", match=MatchValue(value=language))])
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        return [{"score": p.score, **p.payload} for p in response.points]
