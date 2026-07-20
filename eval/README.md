# Evaluation datasets

Retrieval quality is measured against labeled datasets: natural-language queries mapped
to the file(s) that *should* be retrieved.

| Dataset | Corpus | Purpose |
|---|---|---|
| `dataset.jsonl` | this repo | quick smoke test |
| `dataset_gitea.jsonl` | [go-gitea/gitea](https://github.com/go-gitea/gitea) | the real benchmark (feature-level queries + negatives) |

## `dataset_gitea.jsonl`

Feature-level queries whose meaning is obvious to any reviewer ("where is login handled?",
"how are pull requests merged?"), with **file paths relative to the Gitea repo root**. It also
includes **negative queries** (`"negative": true`, empty `relevant`) — features Gitea does not
have — used to check that the system declines to answer instead of fabricating one.

Each line:
```json
{"query": "How are pull requests merged?", "relevant": ["services/pull/merge.go"], "negative": false}
```

### Run it against Gitea

```bash
git clone --depth 1 https://github.com/go-gitea/gitea /path/to/gitea
docker run -d -p 6333:6333 qdrant/qdrant
semantic-index index /path/to/gitea
python -m eval.retrieval_eval --dataset eval/dataset_gitea.jsonl   # runner flag added in the ablation PR
```

> Labels were derived from the Gitea source and should be re-verified whenever Gitea's
> layout changes. They are a starting point for the ablation benchmark, not a frozen truth set.
