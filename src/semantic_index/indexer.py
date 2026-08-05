"""Indexing orchestration: walk -> chunk -> embed -> upsert, incrementally.

Only files whose content hash changed are re-embedded; deleted files are pruned.
Chunks are embedded and upserted in batches to keep memory flat on large repos.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

from .chunker import Chunk, chunk_file
from .chunker_ast import chunk_file_ast
from .config import settings
from .embedder import Embedder, SparseEmbedder
from .state import StateStore
from .store import VectorStore

INCLUDE_EXTS = {
    ".py", ".go", ".java", ".js", ".jsx", ".ts", ".tsx",
    ".c", ".h", ".cpp", ".cc", ".hpp", ".cs", ".rs", ".rb",
}
SKIP_DIRS = {
    ".git", "node_modules", ".venv", "venv", "__pycache__",
    "dist", "build", ".idea", ".mypy_cache", ".pytest_cache",
}


def contextualize(chunks: List[Chunk], texts: List[str], mode: str) -> List[str]:
    """The texts to EMBED for each chunk (stored payload text stays unchanged).

    mode="path" prepends the file path as a context header ("contextual chunk headers"),
    so a per-function chunk still carries its module/feature context — e.g. a chunk from
    services/auth/oauth2.go embeds with "auth" and "oauth2" present even if the code
    text itself never mentions them.
    """
    if mode != "path":
        return texts
    return [f"{chunk.path}\n{text}" for chunk, text in zip(chunks, texts)]


class Indexer:
    def __init__(
        self,
        embedder: Embedder | None = None,
        sparse_embedder: SparseEmbedder | None = None,
        chunk_strategy: str | None = None,
        collection: str | None = None,
        state_db: str | None = None,
        embed_context: str | None = None,
    ) -> None:
        self.embedder = embedder or Embedder()
        # Hybrid when a sparse embedder is injected or RETRIEVAL=hybrid is configured.
        self.hybrid = sparse_embedder is not None or settings.retrieval_mode == "hybrid"
        self.sparse = sparse_embedder or (SparseEmbedder() if settings.retrieval_mode == "hybrid" else None)
        self.store = VectorStore(dim=self.embedder.dim, hybrid=self.hybrid, collection=collection)
        self.state = StateStore(state_db)
        # Pick the chunker by strategy: "ast" (tree-sitter) or "line" (sliding window).
        strategy = chunk_strategy or settings.chunk_strategy
        self._chunk = chunk_file_ast if strategy == "ast" else chunk_file
        self.embed_context = embed_context or settings.embed_context

    def _iter_files(self, root: Path) -> Iterator[Path]:
        for path in root.rglob("*"):
            if path.is_dir():
                continue
            if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
                continue
            if path.suffix.lower() not in INCLUDE_EXTS:
                continue
            yield path

    def index(self, root: str) -> dict:
        root_path = Path(root).resolve()
        seen: set[str] = set()
        indexed = skipped = 0
        batch_chunks: List[Chunk] = []
        batch_texts: List[str] = []

        def flush() -> None:
            if not batch_chunks:
                return
            embed_texts = contextualize(batch_chunks, batch_texts, self.embed_context)
            if settings.embed_max_chars > 0:
                embed_texts = [t[: settings.embed_max_chars] for t in embed_texts]
            vectors = self.embedder.embed(embed_texts)
            sparse = self.sparse.embed(embed_texts) if self.sparse else None
            self.store.upsert(vectors, batch_chunks, sparse_vectors=sparse)
            batch_chunks.clear()
            batch_texts.clear()

        for path in self._iter_files(root_path):
            rel = str(path.relative_to(root_path))
            seen.add(rel)
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            sha = StateStore.content_sha(text)
            if not self.state.needs_reindex(rel, sha):
                skipped += 1
                continue

            # Changed file: drop its old chunks, then re-add.
            self.store.delete_by_path(rel)
            chunks = self._chunk(rel, text)
            for ch in chunks:
                batch_chunks.append(ch)
                batch_texts.append(ch.text)
                if len(batch_chunks) >= settings.batch_size:
                    flush()
            self.state.record(rel, sha, len(chunks))
            indexed += 1

        flush()

        removed = 0
        for gone in self.state.all_paths() - seen:
            self.store.delete_by_path(gone)
            self.state.remove(gone)
            removed += 1

        return {"indexed": indexed, "skipped": skipped, "removed": removed}
