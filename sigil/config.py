from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "version": "0.1",
    "backend_profiles": {
        "local": {"kind": "local", "cpu": 0, "mem_gb": 0},
    },
    "optimizer_params": {
        "alphaevolve": {
            "population": 4,
        }
    },
}


def load_global_config(repo_root: Path) -> Dict[str, Any]:
    cfg_file = repo_root / "sigil.yaml"
    if not cfg_file.exists():
        return DEFAULT_CONFIG.copy()
    data = yaml.safe_load(cfg_file.read_text()) or {}
    # Shallow merge defaults
    merged = DEFAULT_CONFIG.copy()
    for k, v in data.items():
        merged[k] = v
    return merged

