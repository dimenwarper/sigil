# Markdown Formatter Example

This example exposes a `format_markdown` helper with intentionally clumsy spacing rules. Sigil proposals can clean up the formatter while the eval keeps output identical to a golden file and tracks runtime.

## Files

- `formatter/formatting.py` – baseline formatter.
- `sample_input.md` / `expected_output.md` – fixture pair used during evaluation.
- `run_eval.py` – emits correctness and latency metrics.
- `.sigil/` – spec and eval definitions.
- `scripts/setup.sh` – helper to bootstrap a virtual environment.

## Manual Eval

```bash
python3 run_eval.py correctness
python3 run_eval.py latency
```

## Sigil

```bash
sigil run --spec markdown_formatter --workspace demo --num 2
```
