#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
  echo "Create a venv first: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

source .venv/bin/activate

if curl -s -o /dev/null -w '' http://127.0.0.1:7860/ 2>/dev/null; then
  echo "UI already running at http://127.0.0.1:7860"
  exit 0
fi

echo "Starting Movie Agent UI at http://127.0.0.1:7860"
exec python -u app.py
