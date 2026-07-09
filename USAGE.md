# Usage guide

A step-by-step walkthrough for running **semantic-code-search** locally. Best run on a personal machine with normal internet access (the embedding model downloads on first use).

---

## Prerequisites (both free)

- **Docker** — runs the Qdrant vector database locally.
- **Ollama** — only needed for `chat` (not for `index` / `search`). Install from [ollama.com](https://ollama.com), then pull a model once:
  ```bash
  ollama pull qwen2.5-coder:7b
  ```

---

## One-time setup

```bash
git clone https://github.com/n1ra7/semantic-code-search
cd semantic-code-search

python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

---

## Everyday use

### 1. Start the vector database (leave it running)
```bash
docker run -d -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant
```

### 2. Index a codebase
Point it at any repository — this project, or a work/personal one:
```bash
semantic-index index .
# or:
semantic-index index /path/to/some/project
```
> The **first run** downloads the embedding model once (~a minute). Re-running only re-embeds files that changed and prunes files that were deleted.

### 3. Search by meaning
No need for exact keywords — search by intent:
```bash
semantic-index search "where do we handle retry and backoff?"
semantic-index search "database connection setup" --language python
```
Each result shows a relevance score, the file, and the line range.

### 4. Ask questions (RAG chat)
Needs Ollama running (step in Prerequisites):
```bash
semantic-index chat "how does incremental re-indexing decide what to skip?"
```
Prints an answer **plus the source files** it used (`path:line-range`). If no Ollama server is reachable, `chat` falls back to showing the retrieved code instead of failing.

### 5. Measure retrieval quality
```bash
python -m eval.retrieval_eval
```
Reports **precision@k, recall@k, hit@k, MRR, nDCG** over the labeled dataset in `eval/dataset.jsonl`. This is where you get real, quotable numbers for how well retrieval performs.

---

## Use it inside Claude Code (MCP)

Register the MCP server so Claude can search your indexed code:
```bash
claude mcp add semantic-code-search -- semantic-index-mcp
```
Then ask Claude things like *"search my codebase for where auth tokens are refreshed"* — it calls the `search_code` and `answer_question` tools this project exposes.

---

## 60-second smoke test (no Ollama needed)

```bash
docker run -d -p 6333:6333 qdrant/qdrant
semantic-index index .
semantic-index search "how is the vector collection created?"
```
If you see ranked results pointing at `src/semantic_index/store.py`, the whole pipeline works.

---

## Configuration

All settings have working defaults and are overridable via environment variables — see [`.env.example`](.env.example). Common ones:

| Variable | Purpose | Default |
|---|---|---|
| `QDRANT_URL` | vector DB endpoint | `http://localhost:6333` |
| `EMBED_MODEL` | embedding model (FastEmbed) | `jinaai/jina-embeddings-v2-base-code` |
| `CHAT_MODEL` | Ollama model for `chat` | `qwen2.5-coder:7b` |
| `OLLAMA_URL` | Ollama server | `http://localhost:11434` |
| `STATE_DB` | incremental-index state file | `./index_state.sqlite` |

---

## Troubleshooting

- **`No results` on search** — make sure you ran `semantic-index index <path>` first, and that Qdrant is running (`curl http://localhost:6333/readyz`).
- **`chat` says "Ollama not reachable"** — start Ollama and pull the model (`ollama pull qwen2.5-coder:7b`). Search/index still work without it.
- **First index is slow** — that's the one-time model download; later runs are fast and offline.
- **Behind a corporate proxy?** The embedding model download may be blocked; run on a network without TLS interception, or point `EMBED_MODEL` at a model your environment can reach.
