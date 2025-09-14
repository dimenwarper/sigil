from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from . import __version__
from .config import load_global_config
from .evals import EvalDef, load_eval, run_eval_commands, scaffold_eval
from .spec import load_spec, scaffold_spec
from .patches import validate_patch_against_pins, store_candidate, canonicalize_diff
from .workspace import ws_path, runs_path
from .workspace import (
    baseline_dir,
    capture_env_lock,
    candidates_root,
    run_dir,
    run_id,
    runs_path,
    write_json,
)


def cmd_generate_spec(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    p = scaffold_spec(repo_root, args.name, args.description, args.repo_root)
    print(f"Created spec: {p}")
    return 0


def cmd_generate_eval(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    p = scaffold_eval(repo_root, args.spec, args.name, args.description)
    print(f"Created eval: {p}")
    return 0


def _baseline_only_run(repo_root: Path, spec_name: str, workspace: str, optimizer: str, eval_name: str) -> int:
    rd = run_dir(repo_root, spec_name, workspace, run_id(optimizer))
    bd = baseline_dir(rd)
    bd.mkdir(parents=True, exist_ok=True)
    # Snapshot empty patch
    (bd / "patch.diff").write_text("")
    # Load eval and run baseline
    eval_def = load_eval(repo_root, eval_name)
    budgets = eval_def.budgets or {}
    timeout = budgets.get("candidate_timeout_s")
    results = run_eval_commands(eval_def, repo_root=repo_root, workdir=bd, timeout_s=timeout)
    write_json(bd / "metrics.json", results)
    write_json(bd / "env.lock", capture_env_lock(repo_root))
    # Write run.json and index.json
    run_info: Dict[str, Any] = {
        "optimizer": optimizer,
        "backend": "local",
        "spec": spec_name,
        "workspace": workspace,
        "eval": eval_name,
        "version": __version__,
    }
    write_json(rd / "run.json", run_info)
    index: Dict[str, Any] = {
        "nodes": [
            {
                "id": "BASELINE",
                "parent": None,
                "status": "evaluated",
                "metrics": results.get("metrics", {}),
            }
        ],
        "pareto_front": ["BASELINE"],
    }
    write_json(rd / "index.json", index)
    print(f"Baseline run completed at {rd}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    # Load spec for validation
    spec = load_spec(repo_root, args.spec)
    if not spec.evals:
        raise SystemExit("Spec has no evals linked. Use generate-eval or add one.")
    eval_name = args.eval or spec.evals[0]
    return _baseline_only_run(repo_root, spec_name=spec.name, workspace=args.workspace, optimizer=args.optimizer, eval_name=eval_name)


def cmd_inspect(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    runs = runs_path(repo_root, args.spec, args.workspace)
    if not runs.exists():
        raise SystemExit(f"No runs found under {runs}")
    latest = sorted([p for p in runs.iterdir() if p.is_dir()])[-1]
    run_info = json.loads((latest / "run.json").read_text())
    index = json.loads((latest / "index.json").read_text())
    print(f"Run: {latest.name} (optimizer={run_info.get('optimizer')}, eval={run_info.get('eval')})")
    print("Nodes:")
    for n in index.get("nodes", []):
        nid = n.get("id")
        metrics = n.get("metrics", {})
        print(f"  - {nid}: {metrics}")
    return 0


def _resolve_run_dir(repo_root: Path, spec: str, workspace: str, run_id_opt: str | None) -> Path:
    if run_id_opt:
        rd = runs_path(repo_root, spec, workspace) / run_id_opt
        if not rd.exists():
            raise SystemExit(f"Run not found: {rd}")
        return rd
    runs = runs_path(repo_root, spec, workspace)
    if not runs.exists():
        raise SystemExit(f"No runs found under {runs}")
    latest = sorted([p for p in runs.iterdir() if p.is_dir()])[-1]
    return latest


def cmd_validate_patch(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    spec = load_spec(repo_root, args.spec)
    patch_text = Path(args.patch_file).read_text()
    ok, msg = validate_patch_against_pins(patch_text, spec, parent_root=repo_root)
    if not ok:
        print(f"INVALID: {msg}")
        return 2
    print("VALID: patch conforms to pin regions")
    return 0


def cmd_add_candidate(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    spec = load_spec(repo_root, args.spec)
    rd = _resolve_run_dir(repo_root, args.spec, args.workspace, args.run)
    patch_text = Path(args.patch_file).read_text()
    ok, msg = validate_patch_against_pins(patch_text, spec, parent_root=repo_root)
    if not ok:
        raise SystemExit(f"Patch validation failed: {msg}")
    digest, cdir = store_candidate(rd, patch_text, parent_id=args.parent, seed=args.seed)
    print(f"Stored candidate {digest} at {cdir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sigil", description="Sigil CLI")
    p.add_argument("--repo-root", default=".", help="Repository root (default: .)")
    p.add_argument("--version", action="version", version=f"sigil {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    gspec = sub.add_parser("generate-spec", help="Create a spec scaffold")
    gspec.add_argument("name")
    gspec.add_argument("description")
    gspec.set_defaults(func=cmd_generate_spec)

    geval = sub.add_parser("generate-eval", help="Create an eval scaffold and link to spec")
    geval.add_argument("--spec", required=True)
    geval.add_argument("name")
    geval.add_argument("description")
    geval.set_defaults(func=cmd_generate_eval)

    run = sub.add_parser("run", help="Run optimization (baseline-only in v0.1 MVP)")
    run.add_argument("--spec", required=True)
    run.add_argument("--workspace", required=True)
    run.add_argument("--optimizer", default="alphaevolve")
    run.add_argument("--eval", required=False)
    run.set_defaults(func=cmd_run)

    insp = sub.add_parser("inspect", help="Inspect latest run in workspace")
    insp.add_argument("--spec", required=True)
    insp.add_argument("--workspace", required=True)
    insp.set_defaults(func=cmd_inspect)

    vpatch = sub.add_parser("validate-patch", help="Validate a unified diff against pin regions")
    vpatch.add_argument("--spec", required=True)
    vpatch.add_argument("--patch-file", required=True)
    vpatch.set_defaults(func=cmd_validate_patch)

    addc = sub.add_parser("add-candidate", help="Store a validated patch as a content-addressed candidate")
    addc.add_argument("--spec", required=True)
    addc.add_argument("--workspace", required=True)
    addc.add_argument("--patch-file", required=True)
    addc.add_argument("--parent", default="BASELINE")
    addc.add_argument("--run", required=False, help="Run ID (defaults to latest)")
    addc.add_argument("--seed", type=int, required=False)
    addc.set_defaults(func=cmd_add_candidate)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
