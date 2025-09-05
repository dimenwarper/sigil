"""Core components of Sigil framework."""

import functools
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from pydantic import BaseModel, Field

F = TypeVar("F", bound=Callable[..., Any])


class Spec(BaseModel):
    """Specification that defines a namespace for code optimization runs."""
    
    name: str = Field(..., description="Unique name for the specification")
    description: Optional[str] = Field(None, description="Description of the spec")
    workspace_root: Path = Field(default_factory=lambda: Path(".sigil"), description="Root path for workspaces")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        arbitrary_types_allowed = True

    def __post_init__(self) -> None:
        """Initialize spec workspace directory."""
        self.workspace_path.mkdir(parents=True, exist_ok=True)
    
    @property
    def workspace_path(self) -> Path:
        """Get the workspace path for this spec."""
        return self.workspace_root / "ws" / self.name


class FunctionTracker:
    """Tracks function calls and evaluations for optimization."""
    
    def __init__(self):
        self.specs: Dict[str, Spec] = {}
        self.tracked_functions: Dict[str, Dict[str, Any]] = {}
        self.samples: List[Dict[str, Any]] = []
    
    def register_spec(self, spec: Spec) -> None:
        """Register a spec for tracking."""
        self.specs[spec.name] = spec
        spec.__post_init__()
    
    def track_call(self, func_name: str, spec_name: str, args: tuple, kwargs: dict, result: Any, eval_score: Optional[float] = None) -> None:
        """Track a function call with its inputs, outputs, and evaluation."""
        sample = {
            "function": func_name,
            "spec": spec_name,
            "args": args,
            "kwargs": kwargs,
            "result": result,
            "eval_score": eval_score,
        }
        self.samples.append(sample)


# Global tracker instance
_tracker = FunctionTracker()


def track(spec: Spec) -> None:
    """Register a spec for tracking."""
    _tracker.register_spec(spec)


def improve(
    with_eval: Optional[Callable[[Any], float]] = None,
    serve_workspace: str = "default",
    spec: Optional[Union[str, Spec]] = None,
) -> Callable[[F], F]:
    """Decorator to mark functions for optimization.
    
    Args:
        with_eval: Evaluation function that scores the output
        serve_workspace: Name of the workspace to serve solutions from
        spec: Spec name or Spec object to use for tracking
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Execute the original function
            result = func(*args, **kwargs)
            
            # Evaluate the result if evaluation function is provided
            eval_score = None
            if with_eval is not None:
                try:
                    eval_score = with_eval(result)
                except Exception as e:
                    print(f"Warning: Evaluation failed for {func.__name__}: {e}")
            
            # Track the call if we have a spec
            if spec is not None:
                spec_name = spec.name if isinstance(spec, Spec) else spec
                _tracker.track_call(
                    func_name=func.__name__,
                    spec_name=spec_name,
                    args=args,
                    kwargs=kwargs,
                    result=result,
                    eval_score=eval_score,
                )
            
            return result
        
        # Store metadata on the wrapper
        wrapper._sigil_eval = with_eval  # type: ignore
        wrapper._sigil_workspace = serve_workspace  # type: ignore
        wrapper._sigil_spec = spec  # type: ignore
        wrapper._sigil_original = func  # type: ignore
        
        return wrapper  # type: ignore
    
    return decorator


def get_tracker() -> FunctionTracker:
    """Get the global function tracker."""
    return _tracker