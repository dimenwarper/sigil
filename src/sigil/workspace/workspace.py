"""
Workspace management for Sigil.

Workspaces store candidates, diffs, and optimization run results.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..core.models import Candidate, EvaluationResult, Pin
from ..core.ids import FunctionID, CandidateID
from ..core.config import get_config


class Workspace:
    """
    A workspace stores candidates and optimization results.
    
    Workspaces are content-addressed and immutable once written.
    """
    
    def __init__(self, name: str, spec_name: str, workspace_dir: Optional[Path] = None):
        self.name = name
        self.spec_name = spec_name
        self.workspace_dir = workspace_dir or get_config().workspace_dir / "workspaces" / spec_name / name
        
        # Create workspace structure
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        (self.workspace_dir / "candidates").mkdir(exist_ok=True)
        (self.workspace_dir / "evaluations").mkdir(exist_ok=True)
        (self.workspace_dir / "metadata").mkdir(exist_ok=True)
        
        self.created_at = datetime.now()
        
        # Load or create metadata
        self._load_metadata()
    
    def _load_metadata(self):
        """Load workspace metadata."""
        metadata_file = self.workspace_dir / "metadata" / "workspace.json"
        
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
                self.created_at = datetime.fromisoformat(metadata["created_at"])
        else:
            self._save_metadata()
    
    def _save_metadata(self):
        """Save workspace metadata."""
        metadata_file = self.workspace_dir / "metadata" / "workspace.json"
        metadata = {
            "name": self.name,
            "spec_name": self.spec_name,
            "created_at": self.created_at.isoformat()
        }
        
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
    
    def store_candidate(self, candidate: Candidate) -> Path:
        """Store a candidate in the workspace."""
        candidate_dir = self.workspace_dir / "candidates" / str(candidate.candidate_id).replace("://", "_").replace(":", "_")
        candidate_dir.mkdir(parents=True, exist_ok=True)
        
        # Save candidate metadata
        metadata_file = candidate_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(candidate.model_dump(mode='json'), f, indent=2)
        
        # Save source code
        source_file = candidate_dir / "source.py"
        with open(source_file, "w") as f:
            f.write(candidate.source_code)
        
        # Save diff
        diff_file = candidate_dir / "diff.patch"
        with open(diff_file, "w") as f:
            f.write(candidate.diff_text)
        
        return candidate_dir
    
    def get_candidate(self, candidate_id: CandidateID) -> Optional[Candidate]:
        """Retrieve a candidate by ID."""
        candidate_dir = self.workspace_dir / "candidates" / str(candidate_id).replace("://", "_").replace(":", "_")
        metadata_file = candidate_dir / "metadata.json"
        
        if not metadata_file.exists():
            return None
        
        with open(metadata_file) as f:
            data = json.load(f)
        
        # Reconstruct the candidate
        return Candidate(
            candidate_id=CandidateID(data["candidate_id"].split(":")[-1]),
            function_id=FunctionID(
                package=data["function_id"].split("@")[0].split("://")[1],
                git_commit=data["function_id"].split("@")[1].split("/")[0],
                module=data["function_id"].split("/")[1].split(":")[0],
                qualname=data["function_id"].split(":")[1].split("#")[0],
                abi_hash=data["function_id"].split("#")[1]
            ),
            source_code=data["source_code"],
            diff_text=data["diff_text"],
            created_at=datetime.fromisoformat(data["created_at"]),
            generator=data["generator"],
            parent_candidate=CandidateID(data["parent_candidate"].split(":")[-1]) if data["parent_candidate"] else None,
            metadata=data["metadata"]
        )
    
    def store_evaluation(self, evaluation: EvaluationResult):
        """Store an evaluation result."""
        eval_file = (self.workspace_dir / "evaluations" / 
                    f"{str(evaluation.candidate_id).replace(':', '_')}_{evaluation.timestamp.isoformat()}.json")
        
        with open(eval_file, "w") as f:
            json.dump(evaluation.model_dump(mode='json'), f, indent=2)
    
    def get_evaluations(self, candidate_id: Optional[CandidateID] = None) -> List[EvaluationResult]:
        """Get evaluation results, optionally filtered by candidate."""
        evaluations = []
        eval_dir = self.workspace_dir / "evaluations"
        
        if not eval_dir.exists():
            return evaluations
        
        for eval_file in eval_dir.glob("*.json"):
            try:
                with open(eval_file) as f:
                    data = json.load(f)
                
                if candidate_id and data["candidate_id"] != str(candidate_id):
                    continue
                
                evaluation = EvaluationResult(
                    candidate_id=CandidateID(data["candidate_id"].split(":")[-1]),
                    function_id=FunctionID(
                        package=data["function_id"].split("@")[0].split("://")[1],
                        git_commit=data["function_id"].split("@")[1].split("/")[0],
                        module=data["function_id"].split("/")[1].split(":")[0],
                        qualname=data["function_id"].split(":")[1].split("#")[0],
                        abi_hash=data["function_id"].split("#")[1]
                    ),
                    metrics=data["metrics"],
                    passed=data["passed"],
                    error_message=data["error_message"],
                    execution_time=data["execution_time"],
                    timestamp=datetime.fromisoformat(data["timestamp"])
                )
                evaluations.append(evaluation)
            except (json.JSONDecodeError, KeyError):
                continue
        
        return sorted(evaluations, key=lambda e: e.timestamp)
    
    def list_candidates(self, function_id: Optional[FunctionID] = None) -> List[CandidateID]:
        """List all candidate IDs in the workspace."""
        candidates_dir = self.workspace_dir / "candidates"
        candidate_ids = []
        
        if not candidates_dir.exists():
            return candidate_ids
        
        for candidate_dir in candidates_dir.iterdir():
            if not candidate_dir.is_dir():
                continue
            
            metadata_file = candidate_dir / "metadata.json"
            if not metadata_file.exists():
                continue
            
            try:
                with open(metadata_file) as f:
                    data = json.load(f)
                
                if function_id and data["function_id"] != str(function_id):
                    continue
                
                candidate_id = CandidateID(data["candidate_id"].split(":")[-1])
                candidate_ids.append(candidate_id)
            except (json.JSONDecodeError, KeyError):
                continue
        
        return candidate_ids
    
    def get_best_candidate(self, function_id: FunctionID, metric: str = "score") -> Optional[Candidate]:
        """Get the best performing candidate for a function based on a metric."""
        candidates = [self.get_candidate(cid) for cid in self.list_candidates(function_id)]
        candidates = [c for c in candidates if c is not None]
        
        if not candidates:
            return None
        
        # Get evaluations for each candidate
        candidate_scores = {}
        for candidate in candidates:
            evaluations = self.get_evaluations(candidate.candidate_id)
            if evaluations:
                # Take the latest evaluation
                latest_eval = evaluations[-1]
                if metric in latest_eval.metrics:
                    candidate_scores[candidate] = latest_eval.metrics[metric]
        
        if not candidate_scores:
            return None
        
        # Return candidate with highest score
        return max(candidate_scores.keys(), key=lambda c: candidate_scores[c])
    
    def export_workspace(self, target_path: Path):
        """Export workspace as a tar archive."""
        import tarfile
        
        with tarfile.open(target_path, "w:gz") as tar:
            tar.add(self.workspace_dir, arcname=f"workspace_{self.name}")
    
    @classmethod
    def import_workspace(cls, archive_path: Path, workspace_dir: Optional[Path] = None) -> "Workspace":
        """Import workspace from a tar archive."""
        import tarfile
        
        if workspace_dir is None:
            workspace_dir = get_config().workspace_dir
        
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(workspace_dir)
        
        # Find the extracted workspace
        extracted_dirs = [d for d in workspace_dir.glob("workspace_*") if d.is_dir()]
        if not extracted_dirs:
            raise ValueError("No workspace found in archive")
        
        workspace_path = extracted_dirs[0]
        workspace_name = workspace_path.name.replace("workspace_", "")
        
        # Load metadata to get spec name
        metadata_file = workspace_path / "metadata" / "workspace.json"
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        return cls(workspace_name, metadata["spec_name"], workspace_path)