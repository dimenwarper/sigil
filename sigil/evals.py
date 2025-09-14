from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .utils import CmdResult, parse_regex_value, run_cmd, safe_env


@dataclass
class MetricDef:
    id: str
    kind: str  # "timer" | "checker"
    command: str
    parse: Optional[str] = None


@dataclass
class EvalDef:
    version: str
    name: str
    inputs_generator: Optional[str]
    metrics: List[MetricDef]
    aggregate: Dict[str, Any]
    accept: Dict[str, Any]
    budgets: Dict[str, Any]
    replay: Dict[str, Any]


def eval_path(repo_root: Path, name: str) -> Path:
    return repo_root / ".sigil" / f"{name}.eval.yaml"


def load_eval(repo_root: Path, name: str) -> EvalDef:
    p = eval_path(repo_root, name)
    if not p.exists():
        raise FileNotFoundError(f"Eval not found: {p}")
    data = yaml.safe_load(p.read_text()) or {}
    metrics = [MetricDef(**m) for m in (data.get("metrics") or [])]
    inputs = None
    if isinstance(data.get("inputs"), dict):
        inputs = data["inputs"].get("generator")
    return EvalDef(
        version=str(data.get("version", "0.1")),
        name=str(data.get("name", name)),
        inputs_generator=inputs,
        metrics=metrics,
        aggregate=data.get("aggregate") or {},
        accept=data.get("accept") or {},
        budgets=data.get("budgets") or {},
        replay=data.get("replay") or {},
    )


def scaffold_eval(repo_root: Path, spec_name: str, eval_name: str, description: str) -> Path:
    p = eval_path(repo_root, eval_name)
    if p.exists():
        raise FileExistsError(f"Eval already exists: {p}")
    data: Dict[str, Any] = {
        "version": "0.1",
        "name": eval_name,
        "description": description,
        "inputs": {
            "generator": None,
        },
        "metrics": [
            {
                "id": "correctness",
                "kind": "checker",
                "command": "python -c 'import sys; sys.exit(0)'",
                "parse": "exit_code==0",
            }
        ],
        "aggregate": {
            "objective": "min(latency_ms) subject_to correctness==true",
            "tie_breakers": ["mean(latency_ms)"],
        },
        "accept": {
            "rule": "correctness==true",
        },
        "budgets": {
            "candidate_timeout_s": 60,
            "total_wall_clock_h": 1,
        },
        "replay": {"seed": 17},
    }
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    # Also update spec to link this eval name if present
    spec_file = Path(repo_root) / ".sigil" / f"{spec_name}.sigil.yaml"
    if spec_file.exists():
        sdata = yaml.safe_load(spec_file.read_text()) or {}
        evals = list(sdata.get("evals") or [])
        if eval_name not in evals:
            evals.append(eval_name)
        sdata["evals"] = evals
        spec_file.write_text(yaml.safe_dump(sdata, sort_keys=False))
    return p


def _parse_metric_value(metric: MetricDef, res: CmdResult) -> Any:
    if metric.kind == "checker":
        # default checker: success if exit code 0
        if metric.parse:
            if metric.parse.strip() == "exit_code==0":
                return res.returncode == 0
            # allow regex on stdout
            val = parse_regex_value(metric.parse, res.stdout)
            return bool(val)
        return res.returncode == 0
    elif metric.kind == "timer":
        # prefer regex parse from stdout
        if metric.parse:
            val = parse_regex_value(metric.parse, res.stdout)
            try:
                return float(val) if val is not None else None
            except Exception:
                return None
        # fallback: wall time in ms from CmdResult.duration_s
        return res.duration_s * 1000.0
    else:
        return None


def run_eval_commands(
    eval_def: EvalDef,
    repo_root: Path,
    workdir: Path,
    timeout_s: Optional[int] = None,
) -> Dict[str, Any]:
    env = safe_env()
    results: Dict[str, Any] = {"metrics": {}, "logs": {}}
    # Inputs generator (optional)
    if eval_def.inputs_generator:
        res = run_cmd(eval_def.inputs_generator, cwd=repo_root, env=env, timeout_s=timeout_s)
        results["logs"]["generator"] = {"stdout": res.stdout, "stderr": res.stderr, "returncode": res.returncode}
    # Metrics
    for m in eval_def.metrics:
        res = run_cmd(m.command, cwd=repo_root, env=env, timeout_s=timeout_s)
        results["logs"][m.id] = {"stdout": res.stdout, "stderr": res.stderr, "returncode": res.returncode}
        results["metrics"][m.id] = _parse_metric_value(m, res)
    return results

