"""
Sigil: A framework for LLM-guided code optimization.

This package provides tools for automatically improving code through 
controlled optimization runs guided by LLMs.
"""

from .spec import Spec, improve
from .core.models import FunctionID, CandidateID

__version__ = "0.1.0"
__all__ = ["Spec", "improve", "FunctionID", "CandidateID"]