from __future__ import annotations

from dataclasses import dataclass
import uuid as _uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import re

import yaml
from pydantic import BaseModel, Field


@dataclass
class Pin:
    id: str
    language: Optional[str] = None
    files: Optional[List[str]] = None
    symbol: Optional[str] = None
    ast_query: Optional[str] = None
    uuid: Optional[str] = None


@dataclass
class Spec:
    version: str
    name: str
    description: str
    repo_root: Path
    evals: List[str]
    pins: List[Pin]
    base_commit: Optional[str] = None
    id: Optional[str] = None


def spec_dir(repo_root: Path) -> Path:
    return repo_root / ".sigil"


def spec_path(repo_root: Path, name: str) -> Path:
    # Stored as .sigil/<name>.sigil.yaml
    return spec_dir(repo_root) / f"{name}.sigil.yaml"


def load_spec(repo_root: Path, name: str) -> Spec:
    p = spec_path(repo_root, name)
    if not p.exists():
        raise FileNotFoundError(f"Spec not found: {p}")
    data = yaml.safe_load(p.read_text()) or {}
    pins = [Pin(**pin) for pin in (data.get("pins") or [])]
    return Spec(
        version=str(data.get("version", "0.1")),
        name=str(data.get("name", name)),
        description=str(data.get("description", "")),
        repo_root=repo_root,
        evals=list(data.get("evals") or []),
        pins=pins,
        base_commit=data.get("base_commit"),
        id=data.get("id"),
    )


def scaffold_spec(repo_root: Path, name: str, description: str, repo_rel_root: str = ".") -> Path:
    sd = spec_dir(repo_root)
    sd.mkdir(parents=True, exist_ok=True)
    p = spec_path(repo_root, name)
    if p.exists():
        raise FileExistsError(f"Spec already exists: {p}")
    data: Dict[str, Any] = {
        "version": "0.1",
        "name": name,
        "description": description,
        "repo_root": repo_rel_root,
        "evals": [],
        "pins": [
            {
                "id": "example_region",
                "language": "python",
                "files": ["src/example.py"],
                "symbol": None,
                "ast_query": None,
                "uuid": str(_uuid.uuid4()),
            }
        ],
        "base_commit": None,
        "id": str(_uuid.uuid4()),
    }
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    # Ensure workspace home
    (sd / name / "workspaces").mkdir(parents=True, exist_ok=True)
    return p


def spec_uri(spec: Spec) -> str:
    sid = spec.id or spec.name
    return f"sigil:spec/{sid}"


def pin_uri(spec: Spec, pin: Pin) -> str:
    sid = spec.id or spec.name
    pid = pin.uuid or pin.id
    return f"sigil:pin/{sid}/{pid}"


class LLMSpecRequest(BaseModel):
    """Request model for LLM-powered spec generation."""
    prompt: str = Field(description="Natural language description of what to optimize")
    repo_root: Path = Field(description="Repository root path")
    target_files: Optional[List[str]] = Field(default=None, description="Specific files to analyze (optional)")
    

class LLMSpecResponse(BaseModel):
    """Response model from LLM spec generation."""
    name: str = Field(description="Generated spec name")
    description: str = Field(description="Generated spec description")
    pins: List[Dict[str, Any]] = Field(description="Generated pins configuration")
    eval_name: str = Field(description="Generated evaluation name")
    eval_description: str = Field(description="Generated evaluation description")
    eval_metrics: List[Dict[str, Any]] = Field(description="Generated evaluation metrics")


def _analyze_target_files(repo_root: Path, files: List[str]) -> Dict[str, str]:
    """Analyze target files to provide context to the LLM."""
    file_contents = {}
    for file_path in files:
        full_path = repo_root / file_path
        if full_path.exists() and full_path.is_file():
            try:
                content = full_path.read_text(encoding='utf-8')
                # Truncate very large files to avoid token limits
                if len(content) > 8000:
                    content = content[:8000] + "\n... (truncated)"
                file_contents[file_path] = content
            except Exception:
                file_contents[file_path] = "# Could not read file"
    return file_contents


def _extract_functions_and_classes(content: str) -> List[str]:
    """Extract function and class names from Python code."""
    symbols = []
    # Simple regex patterns for Python functions and classes
    func_pattern = r'^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    class_pattern = r'^class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:(]'
    
    for line in content.split('\n'):
        line = line.strip()
        func_match = re.match(func_pattern, line)
        if func_match:
            symbols.append(func_match.group(1))
        class_match = re.match(class_pattern, line)
        if class_match:
            symbols.append(class_match.group(1))
    
    return symbols


def generate_spec_with_llm(
    prompt: str,
    repo_root: Path,
    target_files: Optional[List[str]] = None,
    provider = None
) -> tuple[Path, Path]:
    """
    Generate a spec and evaluation using LLM based on a natural language prompt.
    
    Args:
        prompt: Natural language description of what to optimize
        repo_root: Repository root path
        target_files: Optional list of specific files to analyze
        provider: Optional LLM provider (defaults to config-based selection)
    
    Returns:
        Tuple of (spec_path, eval_path) for the generated files
    """
    if provider is None:
        # Load provider from global config
        from .config import Config
        config = Config.load(repo_root)
        preferred_provider = config.get_preferred_llm_provider()
        provider = config.get_llm_provider(preferred_provider)
    
    # Auto-discover target files if not specified
    if target_files is None:
        # Look for common patterns mentioned in the prompt
        discovered_files = []
        for pattern in [r'(\w+\.py)', r'([\w/]+\.py)', r'(test[\w/]*\.py)']:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            discovered_files.extend(matches)
        
        # Validate discovered files exist
        target_files = []
        for file_path in discovered_files:
            full_path = repo_root / file_path
            if full_path.exists():
                target_files.append(file_path)
        
        # Fallback: look for .py files in common locations
        if not target_files:
            common_locations = ['src', 'lib', '.']
            for loc in common_locations:
                loc_path = repo_root / loc
                if loc_path.exists() and loc_path.is_dir():
                    py_files = list(loc_path.glob('*.py'))[:3]  # Limit to first 3
                    target_files.extend([str(f.relative_to(repo_root)) for f in py_files])
                    if target_files:
                        break
    
    # Analyze target files
    file_contents = _analyze_target_files(repo_root, target_files or [])
    
    # Build context for LLM
    file_analysis = []
    for file_path, content in file_contents.items():
        symbols = _extract_functions_and_classes(content)
        file_analysis.append({
            "path": file_path,
            "symbols": symbols[:10],  # Limit symbols to avoid token overflow
            "content_preview": content[:500] + "..." if len(content) > 500 else content
        })
    
    # Create LLM prompt
    system_prompt = """Given a user's request and code context, generate a JSON response with the following schema:

{
  "name": "spec_name",
  "description": "Clear description of what will be optimized",
  "pins": [
    {
      "id": "unique_pin_id",
      "language": "python",
      "files": ["path/to/file.py"],
      "symbol": "function_or_class_name",
      "ast_query": null,
      "uuid": "generated_uuid"
    }
  ],
  "eval_name": "evaluation_name",
  "eval_description": "Description of how optimization will be measured",
  "eval_metrics": [
    {
      "id": "metric_name",
      "kind": "timer|checker|numeric",
      "command": "command to run for measurement",
      "parse": "how to parse the result"
    }
  ]
}

Guidelines:
- Create specific, actionable pins targeting functions/classes mentioned in the request
- Generate realistic evaluation metrics (performance, correctness, etc.)
- Use appropriate metric kinds: timer for performance, checker for correctness, numeric for other measurements
- Commands should be realistic and executable
- Pin IDs should be descriptive and unique"""

    user_content = f"""Request: {prompt}

Available files and their contents:
{json.dumps(file_analysis, indent=2)}

Generate a complete optimization specification with appropriate pins and evaluation metrics."""

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
        llm_response = LLMSpecResponse.model_validate(response_data)
    except Exception as e:
        raise RuntimeError(f"Failed to parse LLM response: {e}\nResponse: {response_text[:500]}...")
    
    # Generate UUIDs for pins
    for pin_data in llm_response.pins:
        if 'uuid' not in pin_data or not pin_data['uuid']:
            pin_data['uuid'] = str(_uuid.uuid4())
    
    # Create spec file
    spec_data = {
        "version": "0.1",
        "name": llm_response.name,
        "description": llm_response.description,
        "repo_root": ".",
        "evals": [llm_response.eval_name],
        "pins": llm_response.pins,
        "base_commit": None,
        "id": str(_uuid.uuid4()),
    }
    
    # Ensure .sigil directory exists
    sigil_dir = repo_root / ".sigil"
    sigil_dir.mkdir(parents=True, exist_ok=True)
    
    # Write spec file
    spec_path = sigil_dir / f"{llm_response.name}.sigil.yaml"
    if spec_path.exists():
        raise FileExistsError(f"Spec already exists: {spec_path}")
    
    spec_path.write_text(yaml.safe_dump(spec_data, sort_keys=False))
    
    # Create evaluation file
    eval_data = {
        "version": "0.1",
        "name": llm_response.eval_name,
        "description": llm_response.eval_description,
        "inputs": {"generator": None},
        "metrics": llm_response.eval_metrics,
        "aggregate": {
            "objective": "min(latency_ms) subject_to correctness==true",
            "tie_breakers": ["mean(latency_ms)"]
        },
        "accept": {"rule": "correctness==true"},
        "budgets": {
            "candidate_timeout_s": 120,
            "total_wall_clock_h": 2
        },
        "replay": {"seed": 17}
    }
    
    eval_path = sigil_dir / f"{llm_response.eval_name}.eval.yaml"
    if eval_path.exists():
        raise FileExistsError(f"Eval already exists: {eval_path}")
    
    eval_path.write_text(yaml.safe_dump(eval_data, sort_keys=False))
    
    # Create workspace directory
    workspace_dir = sigil_dir / llm_response.name / "workspaces"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    
    return spec_path, eval_path
