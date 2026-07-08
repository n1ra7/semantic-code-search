"""Indexing orchestration: walk -> chunk -> embed -> upsert, incrementally.

Only files whose content hash changed are re-embedded; deleted files are pruned.
Chunks are embedded and upserted in batches to keep memory flat on large repos.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

from .chunker import Chunk, chunk_file
from .config import settings
from .embedder import Embedder
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


class Indexer:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or Embedder()
        self.store = VectorStore(dim=self.embedder.dim)
        self.state = StateStore()

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
            vectors = self.embedder.embed(batch_texts)
            self.store.upsert(vectors, batch_chunks)
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
            chunks = chunk_file(rel, text)
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
