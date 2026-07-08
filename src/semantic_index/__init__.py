"""Semantic code search — a self-hosted, zero-cost RAG index over source repositories.

Pipeline: walk repo -> chunk code -> embed (FastEmbed, local CPU) -> upsert to Qdrant,
with incremental re-indexing tracked in SQLite, and an MCP server exposing search to AI agents.
"""

__version__ = "0.1.0"
