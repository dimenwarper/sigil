"""Sigil: A framework for auto-improving code through LLM-guided optimization."""

__version__ = "0.1.0"

from .core import Spec, improve, track

__all__ = ["Spec", "improve", "track", "__version__"]
