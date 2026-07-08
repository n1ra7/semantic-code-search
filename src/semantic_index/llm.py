"""Minimal client for a local Ollama server (the free, offline LLM backend).

Uses only the standard library, so no extra dependency. If Ollama isn't running,
callers can check `available()` and degrade gracefully instead of crashing.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from .config import settings


class OllamaClient:
    def __init__(self, url: str | None = None, model: str | None = None) -> None:
        self.url = (url or settings.ollama_url).rstrip("/")
        self.model = model or settings.chat_model

    def available(self) -> bool:
        """True if an Ollama server responds (used to fall back to retrieval-only)."""
        try:
            urllib.request.urlopen(self.url + "/api/tags", timeout=2)
            return True
        except (urllib.error.URLError, OSError):
            return False

    def generate(self, prompt: str, system: str | None = None) -> str:
        body: dict = {"model": self.model, "prompt": prompt, "stream": False}
        if system:
            body["system"] = system
        req = urllib.request.Request(
            self.url + "/api/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read())["response"].strip()
