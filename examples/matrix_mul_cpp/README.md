# Matrix Multiplication C++ Example

A compact C++ project showcasing an easily optimizable `multiply` kernel. Sigil can focus on the hot loops while the eval checks correctness and reports runtime.

## Structure

- `include/matrix.hpp` – function declarations.
- `src/matrix.cpp` – baseline implementation.
- `src/benchmark.cpp` – tiny harness used by the eval.
- `.sigil/` – spec and eval definitions.
- `scripts/run_eval.sh` – builds and executes the benchmark.
- `scripts/setup.sh` – bootstrap helper (installs build dir and checks for `g++`).

## Manual Eval

```bash
./scripts/run_eval.sh correctness
./scripts/run_eval.sh latency
```

## Sigil

```bash
sigil run --spec matrix_mul_cpp --workspace demo --backend modal --num 2
```
