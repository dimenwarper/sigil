from __future__ import annotations

import random
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from md_formatter import format_markdown  # type: ignore

INPUT = Path(__file__).with_name("sample_input.md").read_text()
EXPECTED = Path(__file__).with_name("expected_output.md").read_text()


def check_correctness() -> None:
    if format_markdown(INPUT) == EXPECTED:
        print("correctness=true")
    else:
        print("correctness=false")
        sys.exit(1)


def measure_latency() -> None:
    scrambled = "\n\n".join(random.sample(INPUT.splitlines(), len(INPUT.splitlines())))
    start = time.perf_counter()
    for _ in range(200):
        format_markdown(scrambled)
    duration_ms = (time.perf_counter() - start) * 1000.0
    print(f"latency_ms={duration_ms:.3f}")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"correctness", "latency"}:
        print("Usage: python3 run_eval.py [correctness|latency]", file=sys.stderr)
        sys.exit(2)
    if sys.argv[1] == "correctness":
        check_correctness()
    else:
        measure_latency()


if __name__ == "__main__":
    main()
