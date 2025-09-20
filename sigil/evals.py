from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import re

import yaml
from pydantic import BaseModel, Field

from .utils import CmdResult, parse_regex_value, run_cmd, safe_env


class MetricKind(Enum):
    CHECKER = "checker"
    TIMER = "timer"
    NUMERIC = "numeric"


@dataclass
class MetricDef:
    id: str
    kind: MetricKind
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
    metrics = []
    for m in (data.get("metrics") or []):
        metric_data = dict(m)
        # Convert string kind to enum
        if "kind" in metric_data:
            metric_data["kind"] = MetricKind(metric_data["kind"])
        metrics.append(MetricDef(**metric_data))
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
                "kind": MetricKind.CHECKER.value,
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
    if metric.kind == MetricKind.CHECKER:
        # default checker: success if exit code 0
        if metric.parse:
            if metric.parse.strip() == "exit_code==0":
                return res.returncode == 0
            # allow regex on stdout
            val = parse_regex_value(metric.parse, res.stdout)
            return bool(val)
        return res.returncode == 0
    elif metric.kind == MetricKind.TIMER:
        # prefer regex parse from stdout
        if metric.parse:
            val = parse_regex_value(metric.parse, res.stdout)
            try:
                return float(val) if val is not None else None
            except Exception:
                return None
        # fallback: wall time in ms from CmdResult.duration_s
        return res.duration_s * 1000.0
    elif metric.kind == MetricKind.NUMERIC:
        # numeric value: parse as float from stdout using regex
        if metric.parse:
            val = parse_regex_value(metric.parse, res.stdout)
            try:
                return float(val) if val is not None else None
            except Exception:
                return None
        # fallback: try to parse entire stdout as float
        try:
            return float(res.stdout.strip())
        except Exception:
            return None
    else:
        return None


def run_eval_commands(
    eval_def: EvalDef,
    repo_root: Path,
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


class LLMEvalRequest(BaseModel):
    """Request model for LLM-powered evaluation generation."""
    description: str = Field(description="Natural language description of what to evaluate")
    spec_name: str = Field(description="Name of the spec this eval is for")
    repo_root: Path = Field(description="Repository root path")
    target_files: Optional[List[str]] = Field(default=None, description="Files being optimized")
    pins_info: Optional[Dict[str, Any]] = Field(default=None, description="Information about pins being optimized")


class LLMEvalResponse(BaseModel):
    """Response model from LLM evaluation generation."""
    name: str = Field(description="Generated evaluation name")
    description: str = Field(description="Generated evaluation description")
    inputs_generator: Optional[str] = Field(default=None, description="Command to generate test inputs")
    metrics: List[Dict[str, Any]] = Field(description="Generated evaluation metrics")
    aggregate: Dict[str, Any] = Field(description="Aggregation rules")
    accept: Dict[str, Any] = Field(description="Acceptance criteria")
    budgets: Dict[str, Any] = Field(description="Resource budgets")


def _analyze_spec_context(repo_root: Path, spec_name: str) -> Dict[str, Any]:
    """Analyze spec file to understand what's being optimized."""
    from .spec import load_spec, spec_path
    
    context = {"pins": [], "files": [], "symbols": []}
    
    try:
        spec_file = spec_path(repo_root, spec_name)
        if spec_file.exists():
            spec = load_spec(repo_root, spec_name)
            context["description"] = spec.description
            
            for pin in spec.pins:
                pin_info = {
                    "id": pin.id,
                    "files": pin.files or [],
                    "symbol": pin.symbol,
                    "language": pin.language
                }
                context["pins"].append(pin_info)
                if pin.files:
                    context["files"].extend(pin.files)
                if pin.symbol:
                    context["symbols"].append(pin.symbol)
    except Exception:
        # If we can't load spec, provide minimal context
        pass
    
    return context


def _infer_optimization_type(
    description: str, 
    context: Dict[str, Any], 
    provider = None
) -> str:
    """Infer what type of optimization is being performed using LLM."""
    if provider is None:
        # Load provider from global config
        from .config import Config
        config = Config.load(context.get('repo_root', Path.cwd()))
        preferred_provider = config.get_preferred_llm_provider()
        provider = config.get_llm_provider(preferred_provider)
    
    # Fallback keyword search if LLM fails
    def fallback_inference() -> str:
        desc_lower = description.lower()
        if any(word in desc_lower for word in ["performance", "speed", "fast", "optimize", "efficient", "time", "latency"]):
            return "performance"
        elif any(word in desc_lower for word in ["memory", "ram", "allocation", "heap", "stack"]):
            return "memory"
        elif any(word in desc_lower for word in ["accuracy", "correct", "precision", "quality", "error"]):
            return "accuracy"
        elif any(word in desc_lower for word in ["sort", "search", "algorithm", "complexity"]):
            return "algorithmic"
        elif any(word in desc_lower for word in ["size", "space", "storage", "compression"]):
            return "space"
        elif any(word in desc_lower for word in ["throughput", "bandwidth", "concurrent", "parallel"]):
            return "throughput"
        else:
            return "general"
    
    try:
        system_prompt = """Given a description and context, classify the primary optimization type.

Return ONLY one of these optimization types:
- performance: Speed, latency, execution time improvements
- memory: Memory usage, allocation, heap/stack optimization
- accuracy: Correctness, precision, quality improvements  
- algorithmic: Algorithm complexity, data structure improvements
- space: Code size, storage, compression optimizations
- throughput: Concurrent processing, bandwidth, parallel execution
- energy: Power consumption, battery life optimizations
- general: Mixed or unclear optimization goals

Consider both the explicit description and implicit context from the code being optimized."""

        context_info = ""
        if context.get("symbols"):
            context_info += f"Functions/methods: {', '.join(context['symbols'])}\n"
        if context.get("files"):
            context_info += f"Files: {', '.join(context['files'])}\n"
        if context.get("description"):
            context_info += f"Spec description: {context['description']}\n"

        user_content = f"""Description: {description}

Context:
{context_info}

Classify the primary optimization type:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        response = provider.generate(messages).strip().lower()
        
        # Validate response is one of our expected types
        valid_types = {
            "performance", "memory", "accuracy", "algorithmic", 
            "space", "throughput", "energy", "general"
        }
        
        if response in valid_types:
            return response
        else:
            # Try to extract valid type from response
            for valid_type in valid_types:
                if valid_type in response:
                    return valid_type
            
            # Fallback if no valid type found
            return fallback_inference()
            
    except Exception:
        # If LLM call fails, use keyword fallback
        return fallback_inference()


def generate_eval_with_llm(
    description: str,
    spec_name: str,
    eval_name: str,
    repo_root: Path,
    provider = None
) -> Path:
    """
    Generate an evaluation using LLM based on a natural language description.
    
    Args:
        description: Natural language description of what to evaluate
        spec_name: Name of the spec this eval is for
        eval_name: Name for the generated evaluation
        repo_root: Repository root path
        provider: Optional LLM provider (defaults to config-based selection)
    
    Returns:
        Path to the generated evaluation file
    """
    if provider is None:
        # Load provider from global config
        from .config import Config
        config = Config.load(repo_root)
        preferred_provider = config.get_preferred_llm_provider()
        provider = config.get_llm_provider(preferred_provider)
    
    # Analyze spec context
    spec_context = _analyze_spec_context(repo_root, spec_name)
    spec_context['repo_root'] = repo_root  # Add repo_root for provider loading
    optimization_type = _infer_optimization_type(description, spec_context, provider)
    
    # Create LLM prompt
    system_prompt = """Given an evaluation request and code context, generate a JSON response with the following schema:

{
  "name": "evaluation_name",
  "description": "Clear description of what this evaluation measures",
  "inputs_generator": "command to generate test inputs (optional)",
  "metrics": [
    {
      "id": "metric_name",
      "kind": "timer|checker|numeric",
      "command": "executable command to measure this metric",
      "parse": "how to parse the result (e.g., 'exit_code==0' or 'regex:pattern')"
    }
  ],
  "aggregate": {
    "objective": "optimization objective (e.g., 'min(latency_ms) subject_to correctness==true')",
    "tie_breakers": ["list of tie-breaking criteria"]
  },
  "accept": {
    "rule": "acceptance criteria (e.g., 'correctness==true and latency_ms <= 100')"
  },
  "budgets": {
    "candidate_timeout_s": 120,
    "total_wall_clock_h": 2
  }
}

Guidelines:
- Create realistic, executable commands for metrics
- Always include a correctness metric to ensure optimizations don't break functionality
- For performance optimization: include timing metrics (latency, throughput)
- For memory optimization: include memory usage, allocation tracking
- For accuracy optimization: include precision, error rate, quality metrics
- For algorithmic optimization: include complexity analysis, operation counts
- For space optimization: include size measurements, compression ratios
- For throughput optimization: include concurrent processing metrics
- For energy optimization: include power consumption, battery usage
- Use appropriate metric kinds: timer for performance, checker for correctness, numeric for other measurements
- Make commands that can actually be executed in the repository context
- Parse patterns should extract meaningful values from command output"""

    user_content = f"""Evaluation request: {description}

Spec context:
- Spec name: {spec_name}
- Optimization type: {optimization_type}
- Files being optimized: {spec_context.get('files', [])}
- Functions/symbols: {spec_context.get('symbols', [])}
- Pins: {json.dumps(spec_context.get('pins', []), indent=2)}

Generate a comprehensive evaluation that measures both correctness and the optimization target."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    # Get LLM response
    response_text = provider.generate(messages)
    
    # Parse JSON response
    try:
        response_text = response_text.strip()
        if response_text.startswith("```"):
            # Remove code fences
            lines = response_text.split('\n')
            response_text = '\n'.join(lines[1:-1])
        
        response_data = json.loads(response_text)
        llm_response = LLMEvalResponse.model_validate(response_data)
    except Exception as e:
        raise RuntimeError(f"Failed to parse LLM response: {e}\nResponse: {response_text[:500]}...")
    
    # Create evaluation file
    eval_data = {
        "version": "0.1",
        "name": llm_response.name,
        "description": llm_response.description,
        "inputs": {"generator": llm_response.inputs_generator} if llm_response.inputs_generator else {"generator": None},
        "metrics": llm_response.metrics,
        "aggregate": llm_response.aggregate,
        "accept": llm_response.accept,
        "budgets": llm_response.budgets,
        "replay": {"seed": 17}
    }
    
    # Write eval file
    p = eval_path(repo_root, eval_name)
    if p.exists():
        raise FileExistsError(f"Eval already exists: {p}")
    
    p.write_text(yaml.safe_dump(eval_data, sort_keys=False))
    
    # Link eval to spec
    spec_file = repo_root / ".sigil" / f"{spec_name}.sigil.yaml"
    if spec_file.exists():
        sdata = yaml.safe_load(spec_file.read_text()) or {}
        evals = list(sdata.get("evals") or [])
        if eval_name not in evals:
            evals.append(eval_name)
        sdata["evals"] = evals
        spec_file.write_text(yaml.safe_dump(sdata, sort_keys=False))
    
    return p

