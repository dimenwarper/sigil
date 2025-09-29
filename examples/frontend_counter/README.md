# Frontend Counter Example

A miniature React + TypeScript project whose reducer stores a counter with history. The baseline reducer is intentionally inefficient; Sigil can focus on `updateCounter` while the eval safeguards behaviour and measures how quickly actions apply.

## Layout

- `src/state.ts` – reducer and state helpers (optimization target).
- `src/App.tsx` – tiny UI for context.
- `.sigil/` – spec and eval files.
- `scripts/setup.sh` – installs dependencies via npm.
- `scripts/run_eval.sh` – runs correctness and latency checks through `ts-node`.

## Manual Eval

```bash
./scripts/run_eval.sh correctness
./scripts/run_eval.sh latency
```

## Sigil

```bash
sigil run --spec frontend_counter --workspace demo --num 2
```
