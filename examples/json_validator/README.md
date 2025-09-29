# JSON Validator Example

A tiny validation helper that enforces required fields and type checks on payloads. The baseline performs redundant work; Sigil proposals can streamline the validator while the eval guards correctness and throughput.

## Contents

- `validator/rules.py` – baseline implementation of `validate_payload`.
- `fixtures/` – small collection of JSON blobs used by the eval.
- `run_eval.py` – correctness and latency harness.
- `.sigil/` – spec and eval definitions.
- `scripts/setup.sh` – helper for environment setup.

## Manual Eval

```bash
python3 run_eval.py correctness
python3 run_eval.py latency
```

## Sigil

```bash
sigil run --spec json_validator --workspace demo --num 2
```
