#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Please install from https://docs.astral.sh/uv/getting-started/" >&2
  exit 1
fi

PY_SPEC=""
if [[ -f .python-version ]]; then
  PY_SPEC=$(cat .python-version)
fi

echo "[sigil] Creating venv with uv..."
if [[ -n "$PY_SPEC" ]]; then
  uv venv --python "$PY_SPEC" .venv
else
  uv venv .venv
fi

PY=".venv/bin/python"

if [[ -f uv.lock ]]; then
  echo "[sigil] Using uv.lock to sync dependencies..."
  uv pip sync uv.lock --python "$PY"
else
  echo "[sigil] Installing package and dev extras..."
  uv pip install -e . --python "$PY"
  uv pip install -e '.[dev]' --python "$PY"
fi

echo "[sigil] Done. Activate with: source .venv/bin/activate"
echo "[sigil] Run tests with: .venv/bin/pytest -q"

