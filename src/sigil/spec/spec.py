"""
Spec management system for Sigil.

Specs define contexts for code optimization runs.
"""

import inspect
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime

from ..core.models import Pin, SigilConfig
from ..core.ids import FunctionID
from ..core.config import get_config


class Spec:
    """
    A specification that defines a context for code optimization.
    
    Specs track a set of pins (functions marked for improvement) and
    provide the context under which optimization runs happen.
    """
    
    def __init__(
        self, 
        name: str,
        description: Optional[str] = None,
        workspace_dir: Optional[Path] = None,
        evaluators: Optional[Dict[str, Callable]] = None
    ):
        self.name = name
        self.description = description or f"Spec: {name}"
        self.workspace_dir = workspace_dir or get_config().workspace_dir / "specs" / name
        self.evaluators = evaluators or {}
        self.pins: Dict[FunctionID, Pin] = {}
        self.created_at = datetime.now()
        
        # Ensure workspace directory exists
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
    
    def add_pin(
        self, 
        func: Callable, 
        eval_function: Optional[str] = None,
        serve_workspace: Optional[str] = None
    ) -> Pin:
        """Add a function pin to this spec."""
        function_id = FunctionID.from_function(func)
        
        # Get original source code
        try:
            original_source = inspect.getsource(func)
        except OSError:
            original_source = f"# Could not retrieve source for {func.__name__}"
        
        pin = Pin(
            function_id=function_id,
            spec_name=self.name,
            eval_function=eval_function,
            serve_workspace=serve_workspace,
            original_source=original_source
        )
        
        self.pins[function_id] = pin
        return pin
    
    def get_pin(self, func: Callable) -> Optional[Pin]:
        """Get pin for a function if it exists."""
        function_id = FunctionID.from_function(func)
        return self.pins.get(function_id)
    
    def add_evaluator(self, name: str, evaluator: Callable):
        """Add an evaluator function to this spec."""
        self.evaluators[name] = evaluator
    
    def get_evaluator(self, name: str) -> Optional[Callable]:
        """Get an evaluator by name."""
        return self.evaluators.get(name)
    
    def list_pins(self) -> List[Pin]:
        """List all pins in this spec."""
        return list(self.pins.values())
    
    def save(self):
        """Save spec metadata to workspace."""
        import json
        
        spec_file = self.workspace_dir / "spec.json"
        spec_data = {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "pins": [
                {
                    "function_id": str(pin.function_id),
                    "eval_function": pin.eval_function,
                    "serve_workspace": pin.serve_workspace,
                    "created_at": pin.created_at.isoformat()
                }
                for pin in self.pins.values()
            ]
        }
        
        with open(spec_file, "w") as f:
            json.dump(spec_data, f, indent=2)
    
    @classmethod
    def load(cls, name: str, workspace_dir: Optional[Path] = None) -> "Spec":
        """Load a spec from workspace."""
        import json
        
        if workspace_dir is None:
            workspace_dir = get_config().workspace_dir / "specs" / name
        
        spec_file = workspace_dir / "spec.json"
        if not spec_file.exists():
            raise FileNotFoundError(f"Spec {name} not found at {spec_file}")
        
        with open(spec_file) as f:
            spec_data = json.load(f)
        
        spec = cls(
            name=spec_data["name"],
            description=spec_data["description"],
            workspace_dir=workspace_dir
        )
        spec.created_at = datetime.fromisoformat(spec_data["created_at"])
        
        # Load pins (simplified reconstruction)
        for pin_data in spec_data.get("pins", []):
            # We can't fully reconstruct the Pin objects without the actual functions,
            # but we can create basic Pin objects for the CLI to work with
            from ..core.models import Pin
            from ..core.ids import FunctionID
            
            # Parse function ID from string representation
            function_id_str = pin_data["function_id"]
            # This is a simplified parsing - in practice would need more robust parsing
            parts = function_id_str.replace("sigil://", "").split("/")
            package_commit = parts[0].split("@")
            module_qual = parts[1].split(":")
            
            if len(package_commit) == 2 and len(module_qual) == 2:
                package = package_commit[0]
                commit = package_commit[1]
                module = module_qual[0]
                qualname_hash = module_qual[1].split("#")
                if len(qualname_hash) == 2:
                    qualname = qualname_hash[0]
                    abi_hash = qualname_hash[1]
                    
                    function_id = FunctionID(package, commit, module, qualname, abi_hash)
                    
                    pin = Pin(
                        function_id=function_id,
                        spec_name=spec.name,
                        eval_function=pin_data.get("eval_function"),
                        serve_workspace=pin_data.get("serve_workspace"),
                        original_source="# Source not available in loaded spec",
                        created_at=datetime.fromisoformat(pin_data["created_at"])
                    )
                    
                    spec.pins[function_id] = pin
        
        return spec
    
    def __repr__(self) -> str:
        return f"Spec(name='{self.name}', pins={len(self.pins)})"


# Global registry of active specs
_active_specs: Dict[str, Spec] = {}


def track(spec: Spec):
    """Register a spec as active for tracking."""
    _active_specs[spec.name] = spec
    spec.save()


def get_active_spec(name: str) -> Optional[Spec]:
    """Get an active spec by name."""
    return _active_specs.get(name)


def list_active_specs() -> List[Spec]:
    """List all active specs."""
    return list(_active_specs.values())