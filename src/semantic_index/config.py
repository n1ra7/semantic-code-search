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
    # Default chosen by measurement, not size: on the Gitea benchmark bge-small (0.067 GB)
    # beat the code-specialized jina model (0.64 GB) once path-context embedding was on
    # (recall@5 0.804 vs 0.679) — and it indexes ~5x faster with no long-context memory
    # spikes. See the ablation in the README. Swap via EMBED_MODEL.
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

    # Retrieval mode: "dense" (vector only) or "hybrid" (dense + sparse BM25, RRF-fused).
    retrieval_mode: str = os.getenv("RETRIEVAL", "dense")
    sparse_model: str = os.getenv("SPARSE_MODEL", "Qdrant/bm25")

    # Cross-encoder reranking of the top candidates (RERANK=on|off).
    rerank_enabled: bool = os.getenv("RERANK", "off").lower() in ("on", "true", "1")
    rerank_model: str = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
    rerank_candidates: int = int(os.getenv("RERANK_CANDIDATES", "40"))

    # Incremental-indexing state (SQLite, replaces DynamoDB).
    state_db: str = os.getenv("STATE_DB", "./index_state.sqlite")

    # RAG chat: a local Ollama server generates answers from retrieved code.
    # Free and offline; if Ollama isn't running, chat degrades to showing retrieved context.
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    chat_model: str = os.getenv("CHAT_MODEL", "qwen2.5-coder:7b")

    # Hallucination fallback: if the top retrieval score is below this, decline to answer
    # instead of generating. Interpreted in the retrieval score's scale (tune per mode).
    fallback_min_score: float = float(os.getenv("FALLBACK_MIN_SCORE", "0.2"))

    # Observability: expose Prometheus metrics from the MCP server when enabled.
    metrics_enabled: bool = os.getenv("METRICS_ENABLED", "off").lower() in ("on", "true", "1")
    metrics_port: int = int(os.getenv("METRICS_PORT", "9464"))

    # Chunking strategy: "line" (sliding window) or "ast" (function/class boundaries via tree-sitter).
    chunk_strategy: str = os.getenv("CHUNK_STRATEGY", "line")

    # Embedding context: "path" (default) prepends the file path to each chunk's text when
    # EMBEDDING it (the stored payload text is unchanged), so a chunk from
    # services/auth/oauth2.go carries "auth"/"oauth2" signal even if the code never says it.
    # This was the single largest measured improvement (recall@5 0.464 -> 0.804 on the Gitea
    # benchmark). Set to "none" to disable.
    embed_context: str = os.getenv("EMBED_CONTEXT", "path")

    # Cap on characters passed to the EMBEDDING model per chunk (stored text unchanged).
    # Long-context models (e.g. jina-code, 8192 tokens) allocate attention memory ~seq_len^2;
    # a chunk of minified/generated long-line content can spike tens of GB even at small batch
    # sizes. ~4k chars ≈ ~1k tokens keeps memory bounded with negligible retrieval impact.
    embed_max_chars: int = int(os.getenv("EMBED_MAX_CHARS", "4096"))

    # Chunking + batching.
    max_chunk_lines: int = int(os.getenv("MAX_CHUNK_LINES", "60"))
    chunk_overlap_lines: int = int(os.getenv("CHUNK_OVERLAP_LINES", "10"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "64"))


settings = Settings()
