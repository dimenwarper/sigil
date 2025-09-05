"""Workspace management for storing and serving optimization solutions."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Solution(BaseModel):
    """Represents an optimized solution."""
    
    id: str
    function_name: str
    spec_name: str
    workspace_name: str
    code: str
    eval_score: Optional[float] = None
    metadata: Dict[str, Any] = {}
    created_at: datetime = datetime.now()
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class Workspaces:
    """Manages workspaces and solutions for optimization runs."""
    
    def __init__(self, root_path: Path = Path(".sigil")):
        self.root_path = root_path
        self.workspaces_path = root_path / "ws"
        self.workspaces_path.mkdir(parents=True, exist_ok=True)
    
    def get_workspace_path(self, spec_name: str, workspace_name: str = "default") -> Path:
        """Get the path for a specific workspace."""
        workspace_path = self.workspaces_path / spec_name / workspace_name
        workspace_path.mkdir(parents=True, exist_ok=True)
        return workspace_path
    
    def store_solution(self, solution: Solution) -> None:
        """Store a solution in the workspace."""
        workspace_path = self.get_workspace_path(solution.spec_name, solution.workspace_name)
        
        # Store solution metadata
        solutions_path = workspace_path / "solutions"
        solutions_path.mkdir(exist_ok=True)
        
        solution_file = solutions_path / f"{solution.id}.json"
        with open(solution_file, "w") as f:
            json.dump(solution.dict(), f, indent=2, default=str)
        
        # Store the actual code file
        code_path = workspace_path / "code"
        code_path.mkdir(exist_ok=True)
        
        code_file = code_path / f"{solution.function_name}_{solution.id}.py"
        with open(code_file, "w") as f:
            f.write(solution.code)
    
    def load_solution(self, spec_name: str, solution_id: str, workspace_name: str = "default") -> Optional[Solution]:
        """Load a specific solution."""
        workspace_path = self.get_workspace_path(spec_name, workspace_name)
        solution_file = workspace_path / "solutions" / f"{solution_id}.json"
        
        if not solution_file.exists():
            return None
        
        with open(solution_file, "r") as f:
            data = json.load(f)
        
        return Solution(**data)
    
    def list_solutions(self, spec_name: str, workspace_name: str = "default") -> List[Solution]:
        """List all solutions in a workspace."""
        workspace_path = self.get_workspace_path(spec_name, workspace_name)
        solutions_path = workspace_path / "solutions"
        
        if not solutions_path.exists():
            return []
        
        solutions = []
        for solution_file in solutions_path.glob("*.json"):
            with open(solution_file, "r") as f:
                data = json.load(f)
            solutions.append(Solution(**data))
        
        return sorted(solutions, key=lambda s: s.created_at, reverse=True)
    
    def get_best_solution(self, spec_name: str, function_name: str, workspace_name: str = "default") -> Optional[Solution]:
        """Get the best solution for a function based on eval_score."""
        solutions = self.list_solutions(spec_name, workspace_name)
        function_solutions = [s for s in solutions if s.function_name == function_name and s.eval_score is not None]
        
        if not function_solutions:
            return None
        
        return max(function_solutions, key=lambda s: s.eval_score or float('-inf'))
    
    def create_unified_diff(self, original_code: str, optimized_code: str) -> str:
        """Create a unified diff between original and optimized code."""
        import difflib
        
        original_lines = original_code.splitlines(keepends=True)
        optimized_lines = optimized_code.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            original_lines,
            optimized_lines,
            fromfile="original",
            tofile="optimized",
            lineterm=""
        )
        
        return ''.join(diff)
    
    def export_workspace(self, spec_name: str, workspace_name: str = "default", output_path: Optional[Path] = None) -> Path:
        """Export a workspace as a tar.gz archive."""
        if output_path is None:
            output_path = Path(f"{spec_name}_{workspace_name}_workspace.tar.gz")
        
        workspace_path = self.get_workspace_path(spec_name, workspace_name)
        
        shutil.make_archive(
            str(output_path.with_suffix("")),
            "gztar",
            str(workspace_path.parent),
            str(workspace_path.name)
        )
        
        return output_path


# Global workspace manager
_workspaces = Workspaces()


def get_workspaces() -> Workspaces:
    """Get the global workspaces manager."""
    return _workspaces