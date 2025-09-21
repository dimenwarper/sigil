
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/c2a4fe9c-74a5-46c2-bb58-5a20439e6efb">
    <img alt="sigil" src="https://github.com/dimenwarper/sigil/blob/400500a852b604cadb522dbc6ae5057d2daae908/logo.png" width=15%>
  </picture>
</p>


# Sigil: an open specification for LLM‑guided code optimization (v0.1)

Sigil is a minimal, language‑agnostic framework that turns codebases into searchable design spaces and LLMs into proposal policies. It separates *what* should be optimized (specs), *how* candidates are judged (evals), and *how* search proceeds (optimizers), while recording a tamper‑evident provenance of every change as a tree of unified diffs in a workspace. The defaults are brutally simple; everything advanced is a plugin.
This repo includes a complete implementation with multiple LLM providers, backends for running evals, optimization algorithms, interactive setup, and a CLI interface.

---

## 🔧 Installation & Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/dimenwarper/sigil.git
cd sigil

# Install with pip (Python 3.9+)
pip install -e .
```

### Quick Start

1. **Interactive Setup** - Configure your LLM providers and backends:
   ```bash
   sigil setup
   ```

2. **Try the Example** - Run optimization on the included symbolic regression example:
   ```bash
   cd tests/symbolic_regression
   sigil run --spec symbolic_regression --workspace demo --num 3
   ```

3. **Generate Your Own Spec** - Create a specification for your code:
   ```bash
   sigil generate-spec myopt "optimize the sorting function in utils.py"
   sigil generate-eval --spec myopt performance "measure execution time"
   sigil run --spec myopt --workspace experiment1
   ```

### Key Features

- **🤖 Multiple LLM Providers**: OpenAI, Anthropic, or stub providers for testing
- **⚡ Advanced Optimizers**: Simple proposals or AlphaEvolve (island-based evolutionary algorithm)  
- **🖥️ Flexible Backends**: Local execution or distributed Ray backend
- **🎨 Rich Interactive CLI**: Live progress display, colored output, and guided setup
- **📊 Structured Evaluation**: YAML-based specs and evals with automatic validation

---

## 1. Scope and guarantees

A spec declares optimization targets but is silent about algorithms. An eval declares measurements but is silent about edits. An optimizer implements propose‑build‑evaluate‑select at scale but is silent about domain semantics. A workspace is the authoritative record: every candidate is a content‑addressed patch plus its metrics and build logs. Sigil guarantees that any published result can be *replayed* from the recorded patch, environment, seed, and eval definition to reproduce metrics within stated tolerances.

---

## 2. Core objects and their contracts

**Spec.** A small declarative file naming the code elements eligible for modification and how to locate them. It carries no optimization hints. Targets to be optimized are located via *pins* that are language‑agnostic and ideally defined as comments in the source code. A pin has an identity (stable across revisions), a locator, and immutable interface constraints that proposed edits must respect.

**Eval.** A declarative measurement program referencing one or more specs. It provides datasets or tracers, metric definitions, aggregation rules, and acceptance criteria. It defines budgets (timeouts, max cost) and hermetic build/execution instructions for the harness, but never prescribes optimization steps.

**Workspace.** A subdirectory of a spec at `<spec>/workspaces/<workspace>` that stores the baseline, every candidate patch, its lineage, metrics, logs, seeds, and environment manifests. All runs write into a specific workspace. The workspace is the single source of truth for inspection, selection, serving, and publication.

**Run.** A finite search over candidates under an optimizer. A run emits a DAG of patches rooted at the baseline. Nodes store diffs; edges encode parentage. The run terminates by wall clock, evaluation budget, or convergence tests. Runs contribute to workspaces.

**Optimizer.** An orchestration method that treats LLMs as a proposal policies. It implements `propose(context) → {patches}` and `select(population, metrics) → survivors`, and it orchestrates parallel evaluation via a backend. Example optimizers include AlphaEvolve (population‑based, Pareto with novelty), tree search, simulated annealing (single‑trajectory), etc.

**Backend.** An execution substrate providing isolation and scale (local multiprocess, Ray, Kubernetes, or a tertiary provider). Backends must enforce resource limits, network policies, and provide artifact caching.

**Tracer.** A pluggable collector that captures real inputs to targeted functions in production or staging to construct in‑distribution evaluation corpora with sampling, anonymization, and bounded retention.

**Registry.** An optional publish target that accepts signed result bundles (patch + eval manifest + metrics + attestations) and exposes usage/telemetry for collaborative discovery and credit.

---

## 3. File formats (dead‑simple, language‑agnostic)

### 3.1 Spec (YAML)

The spec answers only “what is mutable and how to find it.” Pins can be: file globs; symbol names; regex delimeters; or tree‑sitter queries if available. Interface constraints pin signatures so proposal edits cannot break call sites.

```yaml
# myspec.sigil.yaml
version: 0.1
name: myspec
description: improve the sorting methods in methods.py 
repo_root: .
# Link this spec to one or more evals by name; optional explicit paths for clarity
evals: ["walltime"]
pins:
  - id: my_method
    language: python
    files: ["src/methods.py"]
    symbol: "my_method"
    ast_query: null
base_commit: [commit_id]
```

### 3.2 Eval (YAML)

The eval defines how to build, run, and measure candidates. It may compose several metrics and specify an aggregator.

```yaml
# walltime.eval.yaml
version: 0.1
name: walltime
inputs:
  generator: "./bench/gen_inputs.py --sizes 1024,1024,1024 --reps 10"
metrics:
  - id: latency_ms
    kind: timer
    command: "./bench/bench_my_kernel --reps 100"
    parse: "regex:(?<=p50_ms=)([0-9.]+)"
  - id: correctness
    kind: checker
    command: "./bench/check_correctness --tolerance 1e-5"
    parse: "exit_code==0"
aggregate:
  objective: "min(latency_ms) subject_to correctness==true"
  tie_breakers: ["mean(latency_ms)"]
accept:
  rule: "latency_ms <= 0.95 * baseline.latency_ms and correctness==true"
budgets:
  candidate_timeout_s: 120
  total_wall_clock_h: 2
replay:
  seed: 17
```

### 3.3 Workspace layout

The home directory where all of these yamls should live is a `.sigil` directory within a repository. All other directories live there. All state for a spec `myspec` and workspace `myworkspace` lives under `.sigil/myspec/workspaces/myworkspace/`.

```
myspec/
  workspaces/
    myworkspace/
      runs/
        2025-09-13T17-03-12Z_alphaevolve_/        # run_id
          baseline/
            patch.diff           # empty or normalization diff
            metrics.json
            env.lock
          candidates/
            7f/b1/7fb1…/
              patch.diff
              parent: BASELINE
              metrics.json
              logs.txt
              seed: 123
              prompt.json
          index.json             # lineage DAG, statuses, Pareto front
          run.json               # optimizer, backend, seeds, versions
      selected/
        my_kernel@v3 -> ./runs/<run_id>/candidates/7fb1…
```

### 3.4 Global config

```yaml
# sigil.yaml
version: 0.1
backend_profiles:
  mybackend:
    kind: k8s
    namespace: sigil
    cpu: 4
    mem_gb: 16
    gpu: 1
optimizer_params:
  alphaevolve:
    models:
      default_llm:
        provider: openai-compatible
        model: gpt-4.1-mini
        rate_limit_qps: 2
registry: https://registry.sigil
```

---

## 4. CLI contract

Sigil’s CLI is declarative and composable. Generation helpers produce skeletons but are optional.

### Core Commands

```bash
# Interactive configuration setup
$ sigil setup

# Generate specifications (with optional LLM assistance)
$ sigil generate-spec myspec "improve the kernel in my_kernel to do matmuls"
$ sigil generate-spec --llm "optimize sorting algorithms" --files "utils.py,algorithms.py"

# Generate evaluations (with optional LLM assistance)
$ sigil generate-eval --spec myspec performance "measure wall time of my_kernel"
$ sigil generate-eval --spec myspec --via-description correctness "verify output matches expected"

# Run optimization with flexible options
$ sigil run --spec myspec --workspace myworkspace --optimizer alphaevolve --num 5
$ sigil run --spec myspec --workspace myworkspace --provider openai --backend ray

# Inspect and validate results
$ sigil inspect --spec myspec --workspace myworkspace
$ sigil validate-patch --spec myspec --patch-file changes.diff
$ sigil add-candidate --spec myspec --workspace myworkspace --patch-file optimized.diff
```

### Provider and Backend Options

**LLM Providers:**
- `openai` - OpenAI GPT models (requires `OPENAI_API_KEY`)
- `anthropic` - Anthropic Claude models (requires `ANTHROPIC_API_KEY`)  
- `stub` - Test provider (basic no-op diffs)

**Execution Backends:**
- `local` - Local execution with threading (default)
- `ray` - Distributed execution with Ray (requires `ray` package)

**Optimizers:**
- `simple` - Basic LLM proposal generation
- `alphaevolve` - Island-based evolutionary algorithm with migration

`run` creates a population loop with N workers. Each worker receives the baseline or a survivor, a narrowed code context around the pin, the eval manifest, and a proposal temperature. The worker asks the LLM for a modification expressed strictly as a unified diff relative to the current parent; the system applies the patch in a scratch checkout, builds, runs the eval, records metrics, and returns the result. The optimizer maintains the Pareto set and novelty archive, schedules exploration versus exploitation, and halts on budget. Everything else—SA, RL, MADS—is a different `--optimizer` with the same outer contract.

`inspect` renders the lineage DAG, deltas versus baseline, and the eval front. `select` promotes a candidate into `selected/` for serving and downstream integration. `publish` bundles patch, metrics, eval manifest, and attestations, signs them, and sets them ready for a PR. Additionally, the codeopt run itself is registered in the registry.

---

## 5. Example: Symbolic Regression

Sigil includes a complete working example that demonstrates optimization from `f(x) = x` to `f(x) = x**2` using symbolic regression. This example showcases the full workflow:

### Files Structure
```
tests/symbolic_regression/
├── target_function.py          # Function to optimize: f(x) = x → f(x) = x**2
├── test_correctness.py         # Correctness checker with accuracy scoring
├── .sigil/
│   ├── symbolic_regression.sigil.yaml    # Spec with function pin
│   └── quadratic_correctness.eval.yaml   # Evaluation definition
```

### Running the Example
```bash
cd tests/symbolic_regression

# Configure Sigil (interactive setup)
sigil setup

# Run optimization with 3 candidate proposals
sigil run --spec symbolic_regression --workspace demo --num 3

# Check results
sigil inspect --spec symbolic_regression --workspace demo
```

### What Happens
1. **Target Function**: Starts with `return x` (linear)
2. **LLM Optimization**: Proposes patches to transform the function
3. **Evaluation**: Tests candidates against `f(x) = x**2` behavior
4. **Scoring**: Reports accuracy percentage for each candidate
5. **Selection**: Best performing patches are stored for inspection

This example demonstrates Sigil's ability to perform semantic code transformations guided by LLM reasoning and validated through automated testing.

---

## 6. Collaboration model

A published candidate is a portable artifact that others can fetch, replay, and extend as a new run parent. Credit is attached at the patch level; downstream improvements keep lineage so cumulative contributions are visible. Opening a project for contributions is a single command: publish the spec and evals; contributors run `sigil run --spec yourspec …` and push their results. The registry exposes download counts, replay confirmations, and deployment telemetry (latency deltas and error rates) when consumers opt‑in to share usage statistics.

---

## 7. Implementation Status & Testing

Sigil includes an implementation of the above specification with:

- **✅ Core Framework**: Specs, evals, workspaces, and patch management
- **✅ LLM Integration**: OpenAI, Anthropic, and stub providers with fallback support  
- **✅ Optimization Algorithms**: Simple proposals and AlphaEvolve evolutionary optimizer
- **✅ Execution Backends**: Local threading and distributed Ray backend
- **✅ Rich CLI Interface**: Interactive setup, live progress display, colored output
- **✅ Comprehensive Testing**: 489 lines of tests covering optimization algorithms
- **✅ Working Example**: Symbolic regression demonstration with full workflow
- **✅ Configuration Management**: YAML-based config with validation and environment detection

---

## 8. Opinionated defaults

YAML everywhere for human inspection. Unified diffs as the only edit format, with AST validation to ensure syntactic integrity. Tree‑sitter for pins when available; file and symbol-based targeting as the primary mechanism. Population‑based search first; annealing and RL later. Hermetic containers for every eval. Redacted but preserved LLM prompts for audit. No optimizer has permission to mutate outside declared pins. Correctness is non‑negotiable; performance wins are otherwise meaningless.
