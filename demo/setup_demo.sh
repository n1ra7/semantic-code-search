#!/usr/bin/env bash
# One-time prep before recording the demo GIF with VHS.
# Starts Qdrant and indexes THIS repo so the recorded commands are fast and populated.
#
#   bash demo/setup_demo.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

# 1. Qdrant (skip if one is already listening)
if ! curl -sf http://localhost:6333/readyz >/dev/null 2>&1; then
  echo "Starting Qdrant…"
  docker run -d --name scs-demo-qdrant -p 6333:6333 qdrant/qdrant >/dev/null
  until curl -sf http://localhost:6333/readyz >/dev/null 2>&1; do sleep 1; done
fi

# 2. venv + install
[ -d .venv ] || python -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -e ".[dev]"

# 3. index this repo (the FIRST run downloads the embedding model once, then it's cached)
semantic-index index .

echo
echo "✅ Setup done."
echo "   For the chat beat, make sure Ollama is running:  ollama pull qwen2.5-coder:7b"
echo "   Then record the GIF:                              vhs demo/demo.tape"
echo "   Output:                                           demo/demo.gif"
