#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 [correctness|latency]" >&2
  exit 2
fi
MODE="$1"
if [[ "$MODE" != "correctness" && "$MODE" != "latency" ]]; then
  echo "Unknown mode: $MODE" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d node_modules ]]; then
  npm install
fi

npx ts-node src/eval.ts "$MODE"
