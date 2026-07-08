"""Language-aware code chunking.

A sliding window with overlap keeps related lines together while ensuring a chunk
never exceeds the embedding model's useful context. The window steps by
(max - overlap) lines so adjacent chunks share context and boundary matches are not
lost. Language is inferred from the file extension and stored on every chunk so
searches can be filtered by language.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .config import settings

EXT_LANG = {
    ".py": "python",
    ".go": "go",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rs": "rust",
    ".rb": "ruby",
}


@dataclass
class Chunk:
    path: str
    language: str
    start_line: int
    end_line: int
    text: str


def language_for(path: str) -> str:
    return EXT_LANG.get(Path(path).suffix.lower(), "text")


def chunk_file(path: str, text: str) -> List[Chunk]:
    language = language_for(path)
    lines = text.splitlines()
    if not lines:
        return []

    window = max(1, settings.max_chunk_lines)
    step = window - settings.chunk_overlap_lines
    if step <= 0:
        step = window

    chunks: List[Chunk] = []
    i = 0
    while i < len(lines):
        block = lines[i : i + window]
        if any(line.strip() for line in block):  # skip all-blank windows
            chunks.append(
                Chunk(
                    path=path,
                    language=language,
                    start_line=i + 1,
                    end_line=i + len(block),
                    text="\n".join(block),
                )
            )
        i += step
    return chunks
