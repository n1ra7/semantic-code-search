# Ablation results — full grid

Benchmark: [go-gitea/gitea](https://github.com/go-gitea/gitea) (~11.7k line chunks / ~23.9k AST
chunks), 28 labeled feature-level queries ([`dataset_gitea.jsonl`](dataset_gitea.jsonl)).
Cells are **recall@k / MRR / nDCG@k**. Pooling: `max` ranks a file by its best chunk;
`sumN` by the sum of its top-N chunk scores ([`aggregate.py`](aggregate.py)).

Generated from the raw run data; see the README for the narrative and headline findings.


## bge-small · line chunks · no context

| retrieval · pooling | k=5 | k=10 |
|---|---|---|
| dense · max | 0.464 / 0.385 / 0.373 | 0.661 / 0.385 / 0.437 |
| dense · sum3 | 0.536 / 0.443 / 0.431 | 0.679 / 0.443 / 0.482 |
| hybrid · max | 0.536 / 0.393 / 0.410 | 0.571 / 0.423 / 0.436 |
| hybrid · sum3 | 0.536 / 0.489 / 0.471 | 0.643 / 0.492 / 0.506 |
| rerank@100 · max | 0.482 / 0.365 / 0.360 | 0.589 / 0.368 / 0.394 |
| rerank@100 · sum3 | 0.268 / 0.190 / 0.180 | 0.375 / 0.197 / 0.213 |
| rerank@40 · max | 0.464 / 0.357 / 0.350 | — |
| rerank@40 · sum3 | 0.286 / 0.226 / 0.207 | — |

## bge-small · line chunks · path context

| retrieval · pooling | k=5 | k=10 |
|---|---|---|
| dense · max | 0.804 / 0.588 / 0.629 | 0.804 / 0.590 / 0.629 |
| dense · sum3 | 0.732 / 0.613 / 0.613 | 0.804 / 0.613 / 0.637 |
| hybrid · max | 0.679 / 0.589 / 0.576 | 0.857 / 0.570 / 0.622 |
| hybrid · sum3 | 0.714 / 0.635 / 0.628 | 0.857 / 0.635 / 0.675 |
| rerank@100 · max | 0.482 / 0.365 / 0.360 | 0.589 / 0.372 / 0.393 |
| rerank@100 · sum3 | 0.304 / 0.216 / 0.204 | 0.411 / 0.221 / 0.238 |

## bge-small · AST chunks · path context

| retrieval · pooling | k=5 | k=10 |
|---|---|---|
| dense · max | 0.643 / 0.548 / 0.537 | 0.857 / 0.550 / 0.609 |
| dense · sum3 | 0.786 / 0.617 / 0.638 | 0.857 / 0.617 / 0.662 |
| hybrid · max | 0.607 / 0.491 / 0.492 | 0.750 / 0.497 / 0.539 |
| hybrid · sum3 | 0.643 / 0.526 / 0.520 | 0.786 / 0.525 / 0.567 |
| rerank@100 · max | 0.411 / 0.320 / 0.308 | 0.554 / 0.325 / 0.357 |
| rerank@100 · sum3 | 0.250 / 0.241 / 0.207 | 0.286 / 0.249 / 0.219 |

## jina-code · line chunks · no context

| retrieval · pooling | k=5 | k=10 |
|---|---|---|
| dense · max | 0.571 / 0.441 / 0.447 | 0.696 / 0.446 / 0.491 |
| dense · sum2 | 0.607 / 0.471 / 0.473 | 0.768 / 0.471 / 0.528 |
| dense · sum3 | 0.589 / 0.475 / 0.474 | 0.768 / 0.475 / 0.535 |
| dense · sum5 | 0.589 / 0.392 / 0.413 | 0.768 / 0.392 / 0.475 |
| hybrid · max | 0.571 / 0.521 / 0.499 | 0.714 / 0.526 / 0.547 |
| hybrid · sum2 | 0.607 / 0.496 / 0.494 | 0.714 / 0.496 / 0.530 |
| hybrid · sum3 | 0.607 / 0.499 / 0.496 | 0.714 / 0.499 / 0.532 |
| hybrid · sum5 | 0.607 / 0.499 / 0.496 | 0.714 / 0.499 / 0.532 |

## jina-code · line chunks · path context

| retrieval · pooling | k=5 | k=10 |
|---|---|---|
| dense · max | 0.679 / 0.480 / 0.506 | 0.804 / 0.487 / 0.548 |
| dense · sum2 | 0.714 / 0.554 / 0.558 | 0.786 / 0.554 / 0.584 |
| dense · sum3 | 0.661 / 0.549 / 0.535 | 0.786 / 0.549 / 0.579 |
| dense · sum5 | 0.661 / 0.504 / 0.507 | 0.786 / 0.504 / 0.550 |
| hybrid · max | 0.714 / 0.500 / 0.521 | 0.821 / 0.498 / 0.557 |
| hybrid · sum2 | 0.696 / 0.527 / 0.540 | 0.821 / 0.527 / 0.583 |
| hybrid · sum3 | 0.696 / 0.568 / 0.570 | 0.857 / 0.568 / 0.623 |
| hybrid · sum5 | 0.696 / 0.571 / 0.570 | 0.857 / 0.571 / 0.623 |

## Previous-run reference (stacked ablation, bge-small, max-pool, k=5)

| Config | recall / MRR / nDCG |
|---|---|
| baseline (line, dense) | 0.464 / 0.385 / 0.373 |
| + AST chunking | 0.268 / 0.179 / 0.173 |
| + hybrid | 0.393 / 0.248 / 0.262 |
| + reranking | 0.411 / 0.294 / 0.288 |

## Operational findings (long-context embedding models)

Two distinct OOM mechanisms were hit and fixed while producing these numbers:
1. **Batch spike** — jina-code (8192-token context) at `BATCH_SIZE=64` allocated ~19 GB in one
   batch (ONNX pads to the longest sequence; memory ~ batch x seq_len^2). Fix: small batches.
2. **Sequence spike** — even at `BATCH_SIZE=4`, one long-line generated file filled the full
   8192-token context (~17 GB). Fix: `EMBED_MAX_CHARS` caps the text sent to the embedder
   (stored chunk text unchanged).
