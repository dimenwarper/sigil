from __future__ import annotations

from dataclasses import dataclass
import uuid as _uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


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
