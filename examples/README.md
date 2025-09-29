# Sigil Example Repositories

Self-contained mini projects for experimenting with Sigil outside the unit test harness.

## Included Examples

- **adder_opt** – Python numeric kernel with latency + correctness eval.
- **markdown_formatter** – Python formatter guarded by golden output checks.
- **json_validator** – Python JSON validation helper tracking throughput.
- **matrix_mul_cpp** – C++ matrix multiplication benchmark compiled per run.
- **frontend_counter** – React + TypeScript reducer evaluated via ts-node.

Each example contains:

- `.sigil/` spec and eval definitions ready for `sigil run`.
- Source code with clearly marked optimization targets.
- Minimal setup and evaluation scripts to keep the project standalone.

Use these as blueprints for wiring new codebases into Sigil or as sandboxes for backend experiments.
