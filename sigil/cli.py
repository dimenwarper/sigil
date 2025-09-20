from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

from . import __version__
from .evals import load_eval, scaffold_eval, generate_eval_with_llm
from .spec import load_spec, scaffold_spec, generate_spec_with_llm
from .patches import (
    validate_patch_against_pins,
    store_candidate,
    Candidate,
)
from .optimization import SimpleOptimizer
from .backend import EvalResult
from .workspace import (
    run_dir,
    run_id,
    runs_path,
    write_json,
)
from .config import (
    Config,
    get_available_llm_providers,
    get_available_backends,
    DEFAULT_CONFIG_DICT,
)


def cmd_generate_spec(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    
    # Check if this is an LLM-powered generation request
    if hasattr(args, 'via_description') and args.via_description:
        # LLM-powered generation
        target_files = None
        if hasattr(args, 'files') and args.files:
            target_files = args.files.split(',')
        
        try:
            spec_path, eval_path = generate_spec_with_llm(
                prompt=args.description,  # Use description as the prompt
                repo_root=repo_root,
                target_files=target_files
            )
            print(f"Generated spec: {spec_path}")
            print(f"Generated eval: {eval_path}")
            return 0
        except Exception as e:
            print(f"Error generating spec: {e}")
            return 1
    else:
        # Traditional scaffold generation - name is required
        if not args.name:
            print("Error: 'name' argument is required when not using --llm")
            return 1
        p = scaffold_spec(repo_root, args.name, args.description, args.repo_root)
        print(f"Created spec: {p}")
        return 0


def cmd_generate_eval(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    
    # Check if this is an LLM-powered generation request
    if hasattr(args, 'via_description') and args.via_description:
        # LLM-powered generation
        try:
            p = generate_eval_with_llm(
                description=args.description,
                spec_name=args.spec,
                eval_name=args.name,
                repo_root=repo_root
            )
            print(f"Generated eval: {p}")
            return 0
        except Exception as e:
            print(f"Error generating eval: {e}")
            return 1
    else:
        # Traditional scaffold generation
        p = scaffold_eval(repo_root, args.spec, args.name, args.description)
        print(f"Created eval: {p}")
        return 0


def _run(
    repo_root: Path,
    spec_name: str,
    workspace: str,
    optimizer: str,
    eval_name: str,
    provider_kind: str,
    backend_kind: str,
    num: int,
) -> int:
    rd = run_dir(repo_root, spec_name, workspace, run_id(optimizer))
    eval_def = load_eval(repo_root, eval_name)

    # Load config for provider/backend creation
    config = Config.load(repo_root)
    
    spec = load_spec(repo_root, spec_name)
    
    provider = config.get_llm_provider(provider_kind)
    backend = config.get_backend(backend_kind)
    
    optimizer = SimpleOptimizer()
    responses = optimizer.propose(spec, provider, num=max(1, num))
    patches: list[str] = [r.patch for r in responses]

    # Validate and store candidates
    candidates = set()
    nodes: list[Dict[str, Any]] = []
    for patch_text in patches:
        ok, msg = validate_patch_against_pins(patch_text, spec, parent_root=repo_root)
        if not ok:
            raise SystemExit(f"Proposed patch invalid: {msg}")
        digest, cdir = store_candidate(rd, patch_text)
        candidates.add(Candidate(digest, patch_text))
    candidates = list(candidates)

    # Evaluate in selected backend
    results: list[EvalResult] = backend.evaluate(eval_def, repo_root, candidates)
    # Write candidate results and build index
    for res, candidate in zip(results, candidates):
        cdir = rd / "candidates" / res.id[:2] / res.id[2:4] / res.id
        write_json(cdir / "metrics.json", {"metrics": res.metrics, "logs": res.logs, "error": res.error})
        (cdir / "logs.txt").write_text("candidate evaluated")
        nodes.append({
            "id": res.id, 
            "status": "evaluated" if not res.error else "error", 
            "metrics": res.metrics,
            "candidate": candidate.id,
            }
        )
    write_json(rd / "index.json", {"nodes": nodes})
    write_json(rd / "run.json", {"optimizer": optimizer.__class__.__name__, "backend": backend_kind, "spec": spec_name, "workspace": workspace, "eval": eval_name})
    print(f"Simple LLM run completed at {rd}; candidates: {[it.id for it in candidates]}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    
    # Load configuration
    config = Config.load(repo_root)
    
    # Use config defaults if not specified via command line
    provider_kind = args.provider if hasattr(args, 'provider') and args.provider else config.get_preferred_llm_provider()
    backend_kind = args.backend if hasattr(args, 'backend') and args.backend else config.get_preferred_backend()
    
    # Only validate configuration if we're relying on config defaults
    # If providers/backends are explicitly specified, we can skip validation
    using_config_provider = not (hasattr(args, 'provider') and args.provider)
    using_config_backend = not (hasattr(args, 'backend') and args.backend)
    
    if using_config_provider or using_config_backend:
        issues = config.validate()
        # Filter issues to only show relevant ones
        relevant_issues = []
        for issue in issues:
            if using_config_provider and "LLM provider" in issue:
                relevant_issues.append(issue)
            elif using_config_backend and "backend" in issue:
                relevant_issues.append(issue)
            elif "version" in issue:  # Always show version issues
                relevant_issues.append(issue)
        
        if relevant_issues:
            print("Configuration issues found:")
            for issue in relevant_issues:
                print(f"  - {issue}")
            print("Run 'sigil setup' to fix configuration issues.")
            return 1
    
    # Load spec for validation
    spec = load_spec(repo_root, args.spec)
    if not spec.evals:
        raise SystemExit("Spec has no evals linked. Use generate-eval or add one.")
    eval_name = args.eval or spec.evals[0]
    
    return _run(
        repo_root, 
        spec_name=spec.name, 
        workspace=args.workspace, 
        optimizer=args.optimizer, 
        eval_name=eval_name,
        provider_kind=provider_kind,
        backend_kind=backend_kind,
        num=args.num
        )


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


def cmd_setup(args: argparse.Namespace) -> int:
    """Interactive setup command to configure Sigil."""
    repo_root = Path(args.repo_root).resolve()
    
    print("🔧 Sigil Configuration Setup")
    print("=" * 40)
    
    # Load existing config or start with defaults
    config_file = repo_root / "sigil.yaml"
    if config_file.exists():
        print(f"Found existing configuration at {config_file}")
        config = Config.load(repo_root)
        print("Current configuration will be updated.")
    else:
        print("Creating new configuration...")
        config = Config(DEFAULT_CONFIG_DICT.copy())
    
    print()
    
    # Configure LLM providers
    print("📡 LLM Provider Configuration")
    print("-" * 30)
    
    available_providers = get_available_llm_providers()
    
    # Show available providers
    print("Available LLM providers:")
    for name, provider_info in available_providers.items():
        description = provider_info.get("description", "")
        status = "✓" if provider_info["available"] else "✗"
        print(f"  {name}: {status} {description}")
        if not provider_info["available"] and provider_info.get("requires_env"):
            missing = [env for env in provider_info["requires_env"] if not os.getenv(env)]
            if missing:
                print(f"    Missing: {', '.join(missing)}")
    
    print()
    llm_config = config.data.get("llm", {})
    current_primary = llm_config.get("primary_provider", "openai")
    primary = input(f"Primary LLM provider [{current_primary}]: ").strip() or current_primary
    
    current_fallback = llm_config.get("fallback_provider", "stub")
    fallback = input(f"Fallback LLM provider [{current_fallback}]: ").strip() or current_fallback
    
    # Update LLM config
    config.data.setdefault("llm", {})
    config.data["llm"]["primary_provider"] = primary
    config.data["llm"]["fallback_provider"] = fallback
    
    print()
    
    # Configure backend
    print("🖥️  Backend Configuration")
    print("-" * 25)
    
    available_backends = get_available_backends()
    
    print("Available backends:")
    for name, backend_info in available_backends.items():
        description = backend_info.get("description", "")
        status = "✓" if backend_info["available"] else "✗"
        print(f"  {name}: {status} {description}")
    
    print()
    backend_config = config.data.get("backend", {})
    current_backend = backend_config.get("default", "local")
    backend = input(f"Default backend [{current_backend}]: ").strip() or current_backend
    
    # Update backend config
    config.data.setdefault("backend", {})
    config.data["backend"]["default"] = backend
    
    print()
    
    # Show environment variable requirements
    print("🔐 Environment Variables")
    print("-" * 25)
    
    required_envs = []
    if primary in available_providers:
        provider_info = available_providers[primary]
        for env_var in provider_info.get("requires_env", []):
            required_envs.append((primary, env_var))
    
    if fallback in available_providers and fallback != primary:
        provider_info = available_providers[fallback]
        for env_var in provider_info.get("requires_env", []):
            required_envs.append((fallback, env_var))
    
    if required_envs:
        print("Required environment variables:")
        for provider_name, env_var in required_envs:
            status = "✓ Set" if os.getenv(env_var) else "✗ Not set"
            print(f"  {env_var} (for {provider_name}): {status}")
        
        if any(not os.getenv(env_var) for _, env_var in required_envs):
            print("\nSome required environment variables are missing.")
            print("Set them in your shell profile or .env file.")
    else:
        print("No API keys required for current configuration.")
    
    print()
    
    # Validate and save
    issues = config.validate()
    if issues:
        print("⚠️  Configuration issues:")
        for issue in issues:
            print(f"  - {issue}")
        print()
        
        if not input("Save configuration anyway? [y/N]: ").strip().lower().startswith('y'):
            print("Configuration not saved.")
            return 1
    
    # Save configuration
    config.save(repo_root)
    print(f"✅ Configuration saved to {config_file}")
    
    # Show next steps
    print()
    print("🚀 Next Steps")
    print("-" * 15)
    print("1. Set any missing environment variables")
    print("2. Create a spec: sigil generate-spec <name> '<description>'")
    print("3. Run optimization: sigil run --spec <name> --workspace <workspace>")
    
    return 0




def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sigil", description="Sigil CLI")
    p.add_argument("--repo-root", default=".", help="Repository root (default: .)")
    p.add_argument("--version", action="version", version=f"sigil {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    gspec = sub.add_parser("generate-spec", help="Create a spec scaffold (optionally using LLM)")
    gspec.add_argument("name", nargs='?', help="Spec name (optional when using --llm)")
    gspec.add_argument("description", help="Description or natural language prompt for what to optimize")
    gspec.add_argument("--llm", action="store_true", help="Use LLM to generate spec and eval from description")
    gspec.add_argument("--files", help="Comma-separated list of files to analyze (only with --llm)")
    gspec.set_defaults(func=cmd_generate_spec)

    geval = sub.add_parser("generate-eval", help="Create an eval scaffold (optionally using LLM)")
    geval.add_argument("--spec", required=True, help="Spec name to link this eval to")
    geval.add_argument("name", help="Evaluation name")
    geval.add_argument("description", help="Description or natural language prompt for evaluation")
    geval.add_argument("--via-description", action="store_true", help="Use LLM to generate eval from description")
    geval.set_defaults(func=cmd_generate_eval)

    run = sub.add_parser("run", help="Run optimization")
    run.add_argument("--spec", required=True)
    run.add_argument("--workspace", required=True)
    run.add_argument("--optimizer", default="alphaevolve")
    run.add_argument("--eval", required=False)
    run.add_argument("--llm", choices=["baseline", "simple-llm"], default="baseline")
    run.add_argument("--provider", choices=["openai", "anthropic", "stub"], required=False, 
                     help="LLM provider (defaults to config preference)")
    run.add_argument("--backend", choices=["local", "ray"], required=False,
                     help="Execution backend (defaults to config preference)")
    run.add_argument("--num", type=int, default=1, help="Number of candidates to propose")
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

    setup = sub.add_parser("setup", help="Interactive configuration setup")
    setup.set_defaults(func=cmd_setup)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
