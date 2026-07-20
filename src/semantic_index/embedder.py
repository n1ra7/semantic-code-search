"""Local embedding via FastEmbed.

FastEmbed downloads a quantized ONNX model on first use and runs entirely on CPU,
so there is no per-token cost and no external service call (the Bedrock replacement).
The vector dimension is detected at runtime so any FastEmbed model works unchanged.
"""
from __future__ import annotations

from typing import Iterable, List

from fastembed import TextEmbedding

from .config import settings


class Embedder:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embed_model
        self.model = TextEmbedding(model_name=self.model_name)
        # Probe once to learn the embedding size instead of hard-coding it.
        self._dim = len(next(iter(self.model.embed(["dimension probe"]))))

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        return [vec.tolist() for vec in self.model.embed(list(texts))]


class SparseEmbedder:
    """Sparse (lexical) embeddings for hybrid retrieval — BM25 by default.

    Complements the dense embedder: BM25 captures exact symbol / keyword matches
    (e.g. an error string or a function name) that dense semantic search can miss.
    Returns each vector as (indices, values), the shape Qdrant's SparseVector expects.
    """

    def __init__(self, model_name: str | None = None) -> None:
        from fastembed import SparseTextEmbedding

        self.model_name = model_name or settings.sparse_model
        self.model = SparseTextEmbedding(model_name=self.model_name)

    def embed(self, texts: Iterable[str]) -> List[tuple]:
        out: List[tuple] = []
        for emb in self.model.embed(list(texts)):
            out.append((emb.indices.tolist(), emb.values.tolist()))
        return out
