"""Retrieval-augmented generation: answer a question grounded in retrieved code.

Retrieve top-K chunks from the index, assemble them as cited context, and ask a
local LLM to answer using only that context. The answer carries its source
chunks (path:line-range) so claims are traceable back to real code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .llm import OllamaClient
from .search import Searcher

SYSTEM_PROMPT = (
    "You are a precise coding assistant. Answer the question using ONLY the provided "
    "code context. Cite the files you use as `path:start-end`. If the context is not "
    "sufficient to answer, say so plainly instead of guessing."
)


@dataclass
class Answer:
    question: str
    answer: str
    sources: List[dict]
    context: str


class RagChat:
    def __init__(self, searcher: Searcher | None = None, llm: OllamaClient | None = None) -> None:
        self.searcher = searcher or Searcher()
        self.llm = llm or OllamaClient()

    @staticmethod
    def build_context(hits: List[dict]) -> str:
        return "\n\n".join(
            f"# {h['path']}:{h['start_line']}-{h['end_line']}\n{h['text']}" for h in hits
        )

    def ask(self, question: str, k: int = 6, language: Optional[str] = None) -> Answer:
        hits = self.searcher.search(question, limit=k, language=language)
        context = self.build_context(hits)
        prompt = (
            f"Code context:\n\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer using only the context above, and cite the files you used:"
        )
        answer = self.llm.generate(prompt, system=SYSTEM_PROMPT)
        sources = [
            {
                "path": h["path"],
                "start_line": h["start_line"],
                "end_line": h["end_line"],
                "score": h["score"],
            }
            for h in hits
        ]
        return Answer(question=question, answer=answer, sources=sources, context=context)
