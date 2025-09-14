from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .utils import ensure_dir, now_utc_compact, write_text


def ws_root_for_spec(repo_root: Path, spec_name: str) -> Path:
    return repo_root / ".sigil" / spec_name / "workspaces"


def ws_path(repo_root: Path, spec_name: str, workspace_name: str) -> Path:
    return ws_root_for_spec(repo_root, spec_name) / workspace_name


def runs_path(repo_root: Path, spec_name: str, workspace_name: str) -> Path:
    return ws_path(repo_root, spec_name, workspace_name) / "runs"


def run_id(optimizer: str) -> str:
    return f"{now_utc_compact()}_{optimizer}_"


def run_dir(repo_root: Path, spec_name: str, workspace_name: str, run_id_val: str) -> Path:
    return runs_path(repo_root, spec_name, workspace_name) / run_id_val


def baseline_dir(run_dir_path: Path) -> Path:
    return run_dir_path / "baseline"


def candidates_root(run_dir_path: Path) -> Path:
    return run_dir_path / "candidates"


def capture_env_lock(repo_root: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": None,
        "pip_freeze": [],
    }
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root)).decode().strip()
        info["git_commit"] = commit
    except Exception:
        pass
    try:
        freeze = subprocess.check_output(["python", "-m", "pip", "freeze"], cwd=str(repo_root)).decode().splitlines()
        info["pip_freeze"] = freeze
    except Exception:
        pass
    return info


def write_json(p: Path, data: Dict[str, Any]) -> None:
    ensure_dir(p.parent)
    p.write_text(json.dumps(data, indent=2, sort_keys=False))

