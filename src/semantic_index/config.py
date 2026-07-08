"""Runtime configuration, all overridable via environment variables.

Nothing here is secret: local Qdrant + a local embedding model means no API keys
and no cloud credentials are required to run the whole pipeline.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Vector store (self-hosted Qdrant, replaces managed Qdrant Cloud).
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection: str = os.getenv("COLLECTION", "code_chunks")

    # Embeddings (FastEmbed, runs locally on CPU, replaces AWS Bedrock).
    # A code-specialized default; swap via EMBED_MODEL. See README for options.
    embed_model: str = os.getenv("EMBED_MODEL", "jinaai/jina-embeddings-v2-base-code")

    # Incremental-indexing state (SQLite, replaces DynamoDB).
    state_db: str = os.getenv("STATE_DB", "./index_state.sqlite")

    # Chunking + batching.
    max_chunk_lines: int = int(os.getenv("MAX_CHUNK_LINES", "60"))
    chunk_overlap_lines: int = int(os.getenv("CHUNK_OVERLAP_LINES", "10"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "64"))


settings = Settings()
