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
BUILD_DIR="$ROOT_DIR/build"
mkdir -p "$BUILD_DIR"

GXX=${CXX:-g++}
if ! command -v "$GXX" >/dev/null 2>&1; then
  echo "C++ compiler not found (checked $GXX)." >&2
  exit 1
fi

"$GXX" -std=c++17 -O2 -I"$ROOT_DIR/include" \
  "$ROOT_DIR/src/matrix.cpp" "$ROOT_DIR/src/benchmark.cpp" \
  -o "$BUILD_DIR/benchmark"

"$BUILD_DIR/benchmark" "$MODE"
