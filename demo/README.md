# Demo GIF

Assets for recording the demo GIF used in the README and for social posts.

## Record it (3 steps)

```bash
# 1. install VHS once  (https://github.com/charmbracelet/vhs)
#    macOS:  brew install vhs
#    Linux:  see the VHS releases page (also needs ttyd + ffmpeg)

# 2. prep the environment (starts Qdrant, indexes this repo)
bash demo/setup_demo.sh
#    and, for the chat beat, have Ollama running:
ollama pull qwen2.5-coder:7b

# 3. render
vhs demo/demo.tape        # -> demo/demo.gif
```

## What it shows
1. **Semantic search** — find code by meaning, not keywords
2. **RAG chat** — a question answered from the code, with source citations
3. **Evaluation** — retrieval-quality metrics (precision@k, MRR, nDCG)

## Tuning
- Output cut off? Increase the relevant `Sleep` in `demo.tape` (the chat step depends on your machine's generation speed).
- Change size/theme/typing speed via the `Set` directives at the top of `demo.tape`.

## Use it
Embed at the top of the main README:

```markdown
![demo](demo/demo.gif)
```

Or upload the GIF/MP4 directly to a LinkedIn post.
