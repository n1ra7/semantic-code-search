# VHS tape — renders a demo GIF of semantic-code-search.
# Prereqs: run `bash demo/setup_demo.sh` first, and (for the chat beat) have Ollama running.
# Record with:  vhs demo/demo.tape   ->   produces demo/demo.gif
#
# Tuning: if a command's output gets cut off, increase its `Sleep`. Generation speed
# for the chat step depends on your machine — bump that Sleep if the answer is truncated.

Output demo/demo.gif

Set Shell "bash"
Set FontSize 20
Set Width 1200
Set Height 720
Set Padding 24
Set TypingSpeed 55ms
Set Theme "Catppuccin Mocha"

# --- setup, not recorded ---
Hide
Type "source .venv/bin/activate" Enter
Type "clear" Enter
Show

# Beat 1 — search by meaning
Type "#  Search your code by MEANING, not keywords" Enter
Sleep 1200ms
Type 'semantic-index search "where do we handle retry and backoff?"' Enter
Sleep 4s

# Beat 2 — RAG chat, grounded with citations
Type "#  Ask questions - answered from your code, with sources" Enter
Sleep 1200ms
Type 'semantic-index chat "how does incremental re-indexing decide what to skip?"' Enter
Sleep 9s

# Beat 3 — it's measured
Type "#  ...and retrieval quality is evaluated (precision@k, MRR, nDCG)" Enter
Sleep 1200ms
Type "python -m eval.retrieval_eval" Enter
Sleep 6s

# Outro
Type "#  github.com/n1ra7/semantic-code-search" Enter
Sleep 3s
