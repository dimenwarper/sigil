# Adder Optimization Example

This minimal Python project sums numbers using a deliberately inefficient loop. The Sigil spec pins the `add_all` function so candidate patches can optimize the hot path.

## Layout

- `adder_opt/addition.py` – baseline implementation of `add_all`.
- `benchmark.py` – emits latency and correctness metrics for the evaluator.
- `.sigil/` – spec and eval definitions.
- `scripts/setup.sh` – optional helper to create a virtual environment.

## Running the Eval Manually

```bash
python3 benchmark.py correctness
python3 benchmark.py latency
```

## Using Sigil

```bash
sigil run --spec adder_opt --workspace demo --num 2
```
