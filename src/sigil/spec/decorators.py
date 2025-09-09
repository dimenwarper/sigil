"""
Decorators for marking functions for improvement.
"""

import functools
import inspect
from typing import Callable, Optional, Any

from ..core.config import get_config
from ..core.models import ResolverMode
from .spec import get_active_spec
from ..resolver.resolver import resolve_function


def improve(
    with_eval: Optional[Callable] = None,
    serve_workspace: Optional[str] = None,
    spec_name: Optional[str] = None
):
    """
    Decorator to mark a function for improvement.
    
    Args:
        with_eval: Evaluation function for this pin
        serve_workspace: Workspace to serve optimized versions from  
        spec_name: Spec name (uses default if not specified)
    """
    def decorator(func: Callable) -> Callable:
        config = get_config()
        
        # Determine spec name
        actual_spec_name = spec_name or config.default_spec
        if not actual_spec_name:
            raise ValueError("No spec specified and no default spec configured")
        
        # Get or create spec
        spec = get_active_spec(actual_spec_name)
        if not spec:
            raise ValueError(f"Spec '{actual_spec_name}' is not active. Call sigil.track() first.")
        
        # Add pin to spec
        pin = spec.add_pin(func, 
                          eval_function=with_eval.__name__ if with_eval else None,
                          serve_workspace=serve_workspace)
        
        # Add evaluator if provided
        if with_eval:
            spec.add_evaluator(with_eval.__name__, with_eval)
        
        # Save the spec after adding the pin
        spec.save()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            config = get_config()
            
            # In OFF mode or if tracking disabled, always use original
            if config.resolver_mode == ResolverMode.OFF or not config.tracker_enabled:
                return func(*args, **kwargs)
            
            # Use resolver to determine which implementation to call
            resolved_func = resolve_function(func, pin, *args, **kwargs)
            return resolved_func(*args, **kwargs)
        
        # Store metadata on wrapper
        wrapper._sigil_pin = pin
        wrapper._sigil_original = func
        wrapper._sigil_spec = spec
        
        return wrapper
    
    return decorator