from __future__ import annotations

import random
import sys
import time

from adder_opt.addition import add_all


DATASET = [float(i) for i in range(512)] * 8


def check_correctness() -> None:
    cases = [
        ([], 0.0),
        ([1, 2, 3], 6.0),
        ([-5, 5, 10], 10.0),
    ]
    for values, expected in cases:
        result = add_all(values)
        if abs(result - expected) > 1e-9:
            print("correctness=false")
            sys.exit(1)
    print("correctness=true")


def measure_latency() -> None:
    # Shuffle to avoid constant folding tricks during optimization.
    shuffled = DATASET[:]
    random.shuffle(shuffled)
    start = time.perf_counter()
    for _ in range(200):
        add_all(shuffled)
    duration_ms = (time.perf_counter() - start) * 1000.0
    print(f"latency_ms={duration_ms:.3f}")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"correctness", "latency"}:
        print("Usage: python3 benchmark.py [correctness|latency]", file=sys.stderr)
        sys.exit(2)
    mode = sys.argv[1]
    if mode == "correctness":
        check_correctness()
    else:
        measure_latency()


if __name__ == "__main__":
    main()
