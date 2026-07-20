"""Qdrant vector store wrapper (dense, or dense + sparse hybrid).

Qdrant is open-source and run locally via Docker, so this is the drop-in replacement
for managed Qdrant Cloud at zero cost. Vectors are stored under a named "dense" vector
and, in hybrid mode, an additional "sparse" (BM25) vector. Hybrid search prefetches both
and fuses them server-side with Reciprocal Rank Fusion (RRF).
"""
from __future__ import annotations

import uuid
from typing import List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Modifier,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from .chunker import Chunk
from .config import settings

DENSE = "dense"
SPARSE = "sparse"


class VectorStore:
    def __init__(
        self,
        dim: int,
        url: str | None = None,
        collection: str | None = None,
        hybrid: bool = False,
    ) -> None:
        self.client = QdrantClient(url=url or settings.qdrant_url)
        self.collection = collection or settings.collection
        self.dim = dim
        self.hybrid = hybrid
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        names = {c.name for c in self.client.get_collections().collections}
        if self.collection in names:
            return
        sparse_config = (
            {SPARSE: SparseVectorParams(modifier=Modifier.IDF)} if self.hybrid else None
        )
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={DENSE: VectorParams(size=self.dim, distance=Distance.COSINE)},
            sparse_vectors_config=sparse_config,
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

    def upsert(
        self,
        vectors: List[List[float]],
        chunks: List[Chunk],
        sparse_vectors: Optional[List[Tuple[List[int], List[float]]]] = None,
    ) -> None:
        points = []
        for i, (vec, ch) in enumerate(zip(vectors, chunks)):
            named = {DENSE: vec}
            if self.hybrid and sparse_vectors is not None:
                indices, values = sparse_vectors[i]
                named[SPARSE] = SparseVector(indices=indices, values=values)
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=named,
                    payload={
                        "path": ch.path,
                        "language": ch.language,
                        "start_line": ch.start_line,
                        "end_line": ch.end_line,
                        "text": ch.text,
                    },
                )
            )
        if points:
            self.client.upsert(collection_name=self.collection, points=points)

    def search(
        self,
        vector: List[float],
        limit: int = 8,
        language: Optional[str] = None,
        sparse_vector: Optional[Tuple[List[int], List[float]]] = None,
    ) -> List[dict]:
        query_filter = None
        if language:
            query_filter = Filter(must=[FieldCondition(key="language", match=MatchValue(value=language))])

        if self.hybrid and sparse_vector is not None:
            indices, values = sparse_vector
            response = self.client.query_points(
                collection_name=self.collection,
                prefetch=[
                    Prefetch(query=vector, using=DENSE, limit=limit * 4, filter=query_filter),
                    Prefetch(
                        query=SparseVector(indices=indices, values=values),
                        using=SPARSE,
                        limit=limit * 4,
                        filter=query_filter,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        else:
            response = self.client.query_points(
                collection_name=self.collection,
                query=vector,
                using=DENSE,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
            )
        return [{"score": p.score, **p.payload} for p in response.points]
