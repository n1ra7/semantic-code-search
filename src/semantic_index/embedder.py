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
