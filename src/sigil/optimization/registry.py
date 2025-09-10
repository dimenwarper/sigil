"""
Optimizer registry mapping CLI names to classes.
"""

from __future__ import annotations

from typing import Dict, Type

from .base import BaseOptimizer
from .simple import SimpleOptimizer, RandomSearchOptimizer, GreedyOptimizer


_REGISTRY: Dict[str, Type[BaseOptimizer]] = {
    "simple": SimpleOptimizer,
    "random": RandomSearchOptimizer,
    "greedy": GreedyOptimizer,
}


def get_optimizer_class(name: str) -> Type[BaseOptimizer] | None:
    return _REGISTRY.get(name)


def list_optimizers() -> Dict[str, Type[BaseOptimizer]]:
    return dict(_REGISTRY)

