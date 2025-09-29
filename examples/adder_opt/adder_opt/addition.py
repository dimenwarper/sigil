"""Baseline addition routine with obvious optimization opportunities."""

from __future__ import annotations

from typing import Iterable


def add_all(values: Iterable[float]) -> float:
    total = 0.0
    for value in values:
        # Use explicit float conversion to emphasize work in the hot loop.
        total = float(total + float(value))
    return total
