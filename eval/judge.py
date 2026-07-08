"""RAGAS-style answer evaluation via an LLM-as-judge (uses the local Ollama model).

Two lightweight, widely-used RAG metrics implemented directly so the harness stays
dependency-free and offline:

- faithfulness:      does every claim in the answer follow from the retrieved context?
- context_relevance: was the retrieved context actually relevant to the question?

For the full metric suite you can swap these for the `ragas` library; the interface
(context, question, answer -> score) is the same.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from semantic_index.llm import OllamaClient

_FAITHFULNESS = """Evaluate whether the ANSWER is faithful to the CONTEXT.
An answer is faithful only if every claim it makes is supported by the context.

CONTEXT:
{context}

ANSWER:
{answer}

Reply with ONLY a JSON object: {{"score": 0.0-1.0, "reason": "<short>"}}"""

_CONTEXT_RELEVANCE = """Evaluate whether the CONTEXT is relevant to the QUESTION
(i.e. does it contain information needed to answer it).

QUESTION:
{question}

CONTEXT:
{context}

Reply with ONLY a JSON object: {{"score": 0.0-1.0, "reason": "<short>"}}"""


def _score(prompt: str, llm: OllamaClient) -> dict:
    raw = llm.generate(prompt)
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return {"score": None, "reason": f"unparseable judge output: {raw[:120]}"}
    try:
        obj = json.loads(match.group(0))
        return {"score": float(obj.get("score")), "reason": obj.get("reason", "")}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"score": None, "reason": f"unparseable judge output: {raw[:120]}"}


def judge_faithfulness(context: str, answer: str, llm: Optional[OllamaClient] = None) -> dict:
    llm = llm or OllamaClient()
    return _score(_FAITHFULNESS.format(context=context, answer=answer), llm)


def judge_context_relevance(question: str, context: str, llm: Optional[OllamaClient] = None) -> dict:
    llm = llm or OllamaClient()
    return _score(_CONTEXT_RELEVANCE.format(question=question, context=context), llm)
