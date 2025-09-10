"""
Configuration management for Sigil.
"""

import os
import toml
from pathlib import Path
from typing import Optional

from .models import SigilConfig


_global_config: Optional[SigilConfig] = None


def load_config(config_path: Optional[Path] = None) -> SigilConfig:
    """Load Sigil configuration from file or defaults."""
    if config_path is None:
        # Look for config in standard locations
        candidates = [
            Path.cwd() / ".sigil" / "config.toml",
            Path.cwd() / "sigil.toml", 
            Path.home() / ".sigil" / "config.toml",
        ]
        
        for candidate in candidates:
            if candidate.exists():
                config_path = candidate
                break
    
    if config_path and config_path.exists():
        with open(config_path) as f:
            config_data = toml.load(f)
        return SigilConfig(**config_data)
    else:
        return SigilConfig()


def get_config() -> SigilConfig:
    """Get the global Sigil configuration."""
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


def set_config(config: SigilConfig):
    """Set the global Sigil configuration."""
    global _global_config
    _global_config = config


def save_config(config: SigilConfig, config_path: Optional[Path] = None):
    """Save configuration to file."""
    if config_path is None:
        config_path = Path.cwd() / ".sigil" / "config.toml"
    
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict and save
    config_dict = config.model_dump()
    # Convert known Path objects to strings for TOML serialization
    config_dict["workspace_dir"] = str(config_dict["workspace_dir"])
    if config_dict.get("pyodide_node_dir") is not None:
        config_dict["pyodide_node_dir"] = str(config_dict["pyodide_node_dir"])
    # Convert enum to string
    config_dict["resolver_mode"] = config_dict["resolver_mode"].value
    
    with open(config_path, "w") as f:
        toml.dump(config_dict, f)
