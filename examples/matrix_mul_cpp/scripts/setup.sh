#!/usr/bin/env bash
set -euo pipefail

if command -v g++ >/dev/null 2>&1; then
  echo "g++ located: $(command -v g++)"
else
  echo "g++ not found. Install Xcode command line tools or GNU toolchain." >&2
  exit 1
fi

mkdir -p build
