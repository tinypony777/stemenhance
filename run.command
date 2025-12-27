#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Stable temp dirs (avoid Gradio tempfile issues)
TMP_BASE="$HOME/Library/Caches/bakuage-stem-enhancer/tmp"
mkdir -p "$TMP_BASE"
export TMPDIR="$TMP_BASE"
export GRADIO_TEMP_DIR="$TMP_BASE"

# Prefer python3.12 if available (more compatible wheels), otherwise fallback.
PY_BIN=""
if command -v python3.12 >/dev/null 2>&1; then
  PY_BIN="python3.12"
elif command -v python3 >/dev/null 2>&1; then
  PY_BIN="python3"
else
  echo "python3 not found. Install Python from python.org or Homebrew." >&2
  exit 1
fi

if [ ! -d ".venv" ]; then
  "$PY_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt

python app.py
