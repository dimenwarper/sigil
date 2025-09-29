from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from validator import validate_payload  # type: ignore

VALID = json.loads(Path(__file__).with_name("fixtures").joinpath("valid.json").read_text())
INVALID = json.loads(Path(__file__).with_name("fixtures").joinpath("invalid.json").read_text())


def check_correctness() -> None:
    ok_valid, errs_valid = validate_payload(VALID)
    ok_invalid, errs_invalid = validate_payload(INVALID)
    if ok_valid and not ok_invalid and errs_invalid:
        print("correctness=true")
    else:
        print("correctness=false")
        sys.exit(1)


def measure_latency() -> None:
    payloads = [VALID, INVALID] * 128
    random.shuffle(payloads)
    start = time.perf_counter()
    for payload in payloads:
        validate_payload(payload)
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
