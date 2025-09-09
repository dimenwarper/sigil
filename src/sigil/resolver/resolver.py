"""
Function resolver for Sigil.

Maps function calls to implementations based on manifests and configuration.
"""

import inspect
import importlib
from typing import Callable, Optional, Any, Dict
from pathlib import Path

from ..core.models import Pin, ResolverMode, Manifest
from ..core.config import get_config
from ..workspace.workspace import Workspace
from ..tracking.tracker import get_tracker, track_call


class Resolver:
    """
    Resolves function calls to appropriate implementations.
    
    In production mode, only uses manifest-approved candidates.
    In dev mode, allows local overrides and hot-swaps.
    In off mode, always uses baseline implementations.
    """
    
    def __init__(self):
        self.active_manifest: Optional[Manifest] = None
        self.dev_overrides: Dict[str, Callable] = {}
        self.compiled_functions: Dict[str, Callable] = {}
    
    def load_manifest(self, manifest_path: Path) -> bool:
        """Load and verify a manifest."""
        try:
            import json
            
            with open(manifest_path) as f:
                manifest_data = json.load(f)
            
            # Basic validation (in production, would verify signature)
            required_fields = ["schema", "manifest_id", "pins", "signature"]
            if not all(field in manifest_data for field in required_fields):
                return False
            
            # Convert to Manifest object (simplified)
            self.active_manifest = manifest_data
            
            # Pre-compile approved functions
            self._compile_manifest_functions()
            
            return True
        except Exception as e:
            print(f"Error loading manifest: {e}")
            return False
    
    def _compile_manifest_functions(self):
        """Pre-compile functions from the active manifest."""
        if not self.active_manifest:
            return
        
        for pin_data in self.active_manifest.get("pins", []):
            function_id = pin_data["function_id"]
            candidate_id = pin_data["candidate"]
            
            # In a real implementation, would load the candidate source
            # and compile it to a callable function
            # For now, we'll store the mapping
            self.compiled_functions[function_id] = None
    
    def set_dev_override(self, function_id: str, func: Callable):
        """Set a dev-mode override for a function."""
        config = get_config()
        if config.resolver_mode == ResolverMode.DEV:
            self.dev_overrides[function_id] = func
    
    def resolve(self, original_func: Callable, pin: Pin, *args, **kwargs) -> Callable:
        """
        Resolve which implementation to use for a function call.
        
        Args:
            original_func: The original baseline function
            pin: The function pin
            *args, **kwargs: Function arguments (for logging/tracking)
            
        Returns:
            The function implementation to call
        """
        config = get_config()
        function_id = str(pin.function_id)
        
        # OFF mode: always use baseline
        if config.resolver_mode == ResolverMode.OFF:
            return original_func
        
        # DEV mode: check for overrides first
        if config.resolver_mode == ResolverMode.DEV:
            if function_id in self.dev_overrides:
                return self.dev_overrides[function_id]
            
            # Check for workspace candidates
            if pin.serve_workspace:
                workspace = Workspace(pin.serve_workspace, pin.spec_name)
                best_candidate = workspace.get_best_candidate(pin.function_id)
                if best_candidate:
                    compiled_func = self._compile_candidate(best_candidate)
                    if compiled_func:
                        return compiled_func
        
        # PROD mode: only use manifest-approved candidates
        if config.resolver_mode == ResolverMode.PROD:
            if self.active_manifest and function_id in self.compiled_functions:
                manifest_func = self.compiled_functions[function_id]
                if manifest_func:
                    return manifest_func
        
        # Fallback to baseline
        return original_func
    
    def _compile_candidate(self, candidate) -> Optional[Callable]:
        """
        Compile a candidate's source code to a callable function.
        
        This is a simplified version - a production implementation would
        need proper sandboxing and security measures.
        """
        try:
            # Create a namespace for execution
            namespace = {}
            
            # Execute the candidate source in the namespace
            exec(candidate.source_code, namespace)
            
            # Find the function in the namespace
            function_name = candidate.function_id.qualname.split('.')[-1]
            if function_name in namespace:
                return namespace[function_name]
            
        except Exception as e:
            print(f"Error compiling candidate {candidate.candidate_id}: {e}")
        
        return None


# Global resolver instance
_global_resolver: Optional[Resolver] = None


def get_resolver() -> Resolver:
    """Get the global resolver instance."""
    global _global_resolver
    if _global_resolver is None:
        _global_resolver = Resolver()
    return _global_resolver


def resolve_function(original_func: Callable, pin: Pin, *args, **kwargs) -> Callable:
    """
    Resolve which implementation to use for a function call.
    
    This is the main entry point called by the @improve decorator.
    """
    resolver = get_resolver()
    config = get_config()
    
    # Get the implementation to use
    resolved_func = resolver.resolve(original_func, pin, *args, **kwargs)
    
    # Track the call if tracking is enabled
    if config.tracker_enabled:
        # Determine candidate ID if using a candidate
        candidate_id = None
        if resolved_func != original_func:
            # This is a simplified way to identify if we're using a candidate
            # In practice, would need more sophisticated tracking
            candidate_id = "unknown_candidate"
        
        # The actual call will be made by the wrapper, we just return the function
        # Tracking happens in the wrapper
    
    return resolved_func