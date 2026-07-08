"""Incremental-indexing state in SQLite (the DynamoDB replacement).

For each file we store a content hash. On the next run, files whose hash is unchanged
are skipped, changed files are re-embedded, and files that disappeared are pruned from
the vector store. This is what keeps re-indexing cheap after the first full pass.
"""
from __future__ import annotations

import hashlib
import sqlite3
import time
from typing import Optional, Set

from .config import settings


class StateStore:
    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or settings.state_db
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                path        TEXT PRIMARY KEY,
                content_sha TEXT NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                indexed_at  REAL NOT NULL
            )
            """
        )
        self.conn.commit()

    @staticmethod
    def content_sha(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()

    def needs_reindex(self, path: str, sha: str) -> bool:
        row = self.conn.execute("SELECT content_sha FROM files WHERE path = ?", (path,)).fetchone()
        return row is None or row[0] != sha

    def record(self, path: str, sha: str, chunk_count: int) -> None:
        self.conn.execute(
            """
            INSERT INTO files(path, content_sha, chunk_count, indexed_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                content_sha = excluded.content_sha,
                chunk_count = excluded.chunk_count,
                indexed_at  = excluded.indexed_at
            """,
            (path, sha, chunk_count, time.time()),
        )
        self.conn.commit()

    def all_paths(self) -> Set[str]:
        return {row[0] for row in self.conn.execute("SELECT path FROM files")}

    def remove(self, path: str) -> None:
        self.conn.execute("DELETE FROM files WHERE path = ?", (path,))
        self.conn.commit()
