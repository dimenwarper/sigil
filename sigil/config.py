from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, List

import yaml


DEFAULT_CONFIG_DICT: Dict[str, Any] = {
    "version": "0.1",
    "llm": {
        "primary_provider": "openai",
        "fallback_provider": "stub",
    },
    "backend": {
        "default": "local",
    },
    "backend_profiles": {
        "local": {"kind": "local", "cpu": 0, "mem_gb": 0},
    },
    "optimizer_params": {
        "alphaevolve": {
            "population": 4,
        }
    },
}


class Config:
    """Sigil configuration management."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.data = config_dict
    
    @classmethod
    def load(cls, repo_root: Path) -> 'Config':
        """Load configuration from sigil.yaml file."""
        cfg_file = repo_root / "sigil.yaml"
        if not cfg_file.exists():
            return cls(DEFAULT_CONFIG_DICT.copy())
        data = yaml.safe_load(cfg_file.read_text()) or {}
        # Deep merge defaults with user config
        merged = _deep_merge(DEFAULT_CONFIG_DICT.copy(), data)
        return cls(merged)
    
    def save(self, repo_root: Path) -> None:
        """Save configuration to sigil.yaml file."""
        cfg_file = repo_root / "sigil.yaml"
        with open(cfg_file, 'w') as f:
            yaml.safe_dump(self.data, f, default_flow_style=False, sort_keys=False)
    
    def get_llm_provider(self, provider_name: str):
        """Factory method to create LLM provider instances from config."""
        from .llm import OpenAICompatibleProvider, StubProvider
        
        providers = get_available_llm_providers()
        if provider_name not in providers:
            raise ValueError(f"Unknown LLM provider: {provider_name}")
        
        provider_info = providers[provider_name]
        
        if provider_info["class"] == "StubProvider":
            return StubProvider()
        elif provider_info["class"] == "OpenAICompatibleProvider":
            # Handle special env mapping for anthropic
            if provider_name == "anthropic":
                env_mapping = provider_info.get("env_mapping", {})
                api_key = os.getenv(env_mapping.get("api_key", "ANTHROPIC_API_KEY"))
                base_url = os.getenv(env_mapping.get("base_url", "ANTHROPIC_BASE_URL"))
                model = os.getenv(env_mapping.get("model", "ANTHROPIC_MODEL"))
                return OpenAICompatibleProvider(api_key=api_key, base_url=base_url, model=model)
            else:
                # Default OpenAI behavior
                return OpenAICompatibleProvider()
        else:
            raise ValueError(f"Unknown provider class: {provider_info['class']}")
    
    def get_backend(self, backend_name: str):
        """Factory method to create backend instances from config."""
        from .backend import LocalBackend, RayBackend
        
        backends = get_available_backends()
        if backend_name not in backends:
            raise ValueError(f"Unknown backend: {backend_name}")
        
        backend_info = backends[backend_name]
        
        if backend_info["class"] == "LocalBackend":
            return LocalBackend()
        elif backend_info["class"] == "RayBackend":
            return RayBackend()
        else:
            raise ValueError(f"Unknown backend class: {backend_info['class']}")
    
    def get_preferred_llm_provider(self) -> str:
        """Get the preferred LLM provider, falling back if primary is unavailable."""
        llm_config = self.data.get("llm", {})
        primary = llm_config.get("primary_provider", "openai")
        fallback = llm_config.get("fallback_provider", "stub")
        
        providers = get_available_llm_providers()
        
        # Check if primary provider is available
        if primary in providers and providers[primary]["available"]:
            return primary
        
        # Fall back to fallback provider
        if fallback in providers and providers[fallback]["available"]:
            return fallback
        
        # If both fail, return stub as last resort
        return "stub"
    
    def get_preferred_backend(self) -> str:
        """Get the preferred backend from configuration."""
        backend_config = self.data.get("backend", {})
        preferred = backend_config.get("default", "local")
        
        backends = get_available_backends()
        
        # Check if preferred backend is available
        if preferred in backends and backends[preferred]["available"]:
            return preferred
        
        # Fall back to local if preferred is not available
        return "local"
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Check version
        if self.data.get("version") != "0.1":
            issues.append("Unsupported config version")
        
        # Validate LLM config
        llm_config = self.data.get("llm", {})
        primary = llm_config.get("primary_provider")
        fallback = llm_config.get("fallback_provider")
        
        available_providers = get_available_llm_providers()
        
        if primary and primary not in available_providers:
            issues.append(f"Primary LLM provider '{primary}' is not available")
        elif primary and not available_providers[primary]["available"]:
            missing_env = [env for env in available_providers[primary]["requires_env"] if not os.getenv(env)]
            issues.append(f"Primary LLM provider '{primary}' missing environment variables: {missing_env}")
        
        if fallback and fallback not in available_providers:
            issues.append(f"Fallback LLM provider '{fallback}' is not available")
        elif fallback and not available_providers[fallback]["available"]:
            missing_env = [env for env in available_providers[fallback]["requires_env"] if not os.getenv(env)]
            if missing_env:  # Only warn if fallback needs env vars (stub doesn't)
                issues.append(f"Fallback LLM provider '{fallback}' missing environment variables: {missing_env}")
        
        # Validate backend config
        backend_config = self.data.get("backend", {})
        default_backend = backend_config.get("default")
        
        available_backends = get_available_backends()
        
        if default_backend and default_backend not in available_backends:
            issues.append(f"Default backend '{default_backend}' is not available")
        elif default_backend and not available_backends[default_backend]["available"]:
            issues.append(f"Default backend '{default_backend}' is not available (missing dependencies)")
        
        return issues


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_available_llm_providers() -> Dict[str, Dict[str, Any]]:
    """Get all available LLM providers with their availability status."""
    providers = {
        "openai": {
            "class": "OpenAICompatibleProvider",
            "description": "OpenAI GPT models via API",
            "requires_env": ["OPENAI_API_KEY"],
            "optional_env": ["OPENAI_BASE_URL", "OPENAI_MODEL"],
        },
        "anthropic": {
            "class": "OpenAICompatibleProvider",  # Can use same class with different env vars
            "description": "Anthropic Claude models via API",
            "requires_env": ["ANTHROPIC_API_KEY"],
            "optional_env": ["ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL"],
            "env_mapping": {
                "api_key": "ANTHROPIC_API_KEY",
                "base_url": "ANTHROPIC_BASE_URL", 
                "model": "ANTHROPIC_MODEL"
            }
        },
        "stub": {
            "class": "StubProvider",
            "description": "Deterministic test provider (increments constants)",
            "requires_env": [],
        }
    }
    
    # Check availability
    for name, info in providers.items():
        required_env = info.get("requires_env", [])
        info["available"] = all(os.getenv(env) for env in required_env)
    
    return providers


def get_available_backends() -> Dict[str, Dict[str, Any]]:
    """Get all available backends with their availability status."""
    backends = {
        "local": {
            "class": "LocalBackend",
            "description": "Local execution with threading",
            "available": True,
        },
        "ray": {
            "class": "RayBackend", 
            "description": "Distributed execution with Ray",
            "available": _check_ray_available(),
        }
    }
    return backends


def _check_ray_available() -> bool:
    """Check if Ray is available for import."""
    try:
        import ray
        return True
    except ImportError:
        return False