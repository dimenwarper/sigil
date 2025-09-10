"""
Core data models for Sigil using Pydantic.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_serializer

from .ids import FunctionID, CandidateID


class ResolverMode(str, Enum):
    """Resolver operational modes."""
    OFF = "off"      # Always use baseline
    DEV = "dev"      # Allow hot-swaps and overrides
    PROD = "prod"    # Only use manifest-approved candidates


class Pin(BaseModel):
    """A function marked for improvement."""
    model_config = {"arbitrary_types_allowed": True}
    
    function_id: FunctionID
    spec_name: str
    eval_function: Optional[str] = None  # Name of evaluation function
    serve_workspace: Optional[str] = None
    original_source: str
    created_at: datetime = Field(default_factory=datetime.now)
    
    @field_serializer('function_id')
    def serialize_function_id(self, function_id: FunctionID) -> str:
        return str(function_id)


class EvaluationResult(BaseModel):
    """Result of evaluating a candidate."""
    model_config = {"arbitrary_types_allowed": True}
    
    candidate_id: CandidateID
    function_id: FunctionID
    metrics: Dict[str, float]
    passed: bool
    error_message: Optional[str] = None
    execution_time: float
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @field_serializer('candidate_id')
    def serialize_candidate_id(self, candidate_id: CandidateID) -> str:
        return str(candidate_id)
    
    @field_serializer('function_id')
    def serialize_function_id(self, function_id: FunctionID) -> str:
        return str(function_id)


class Candidate(BaseModel):
    """A candidate implementation of a function."""
    model_config = {"arbitrary_types_allowed": True}
    
    candidate_id: CandidateID
    function_id: FunctionID
    source_code: str
    diff_text: str
    created_at: datetime = Field(default_factory=datetime.now)
    generator: str = "unknown"  # Which optimizer generated this
    parent_candidate: Optional[CandidateID] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_serializer('candidate_id')
    def serialize_candidate_id(self, candidate_id: CandidateID) -> str:
        return str(candidate_id)
    
    @field_serializer('function_id')
    def serialize_function_id(self, function_id: FunctionID) -> str:
        return str(function_id)
    
    @field_serializer('parent_candidate')
    def serialize_parent_candidate(self, parent_candidate: Optional[CandidateID]) -> Optional[str]:
        return str(parent_candidate) if parent_candidate else None


class SampleRecord(BaseModel):
    """A record of a function call and its evaluation."""
    model_config = {"arbitrary_types_allowed": True}
    
    trace_id: str
    function_id: FunctionID
    candidate_id: Optional[CandidateID] = None  # None for baseline
    resolver_mode: ResolverMode
    inputs_summary: Dict[str, Any]  # Sketched inputs
    outputs_summary: Dict[str, Any]  # Sketched outputs
    metrics: Dict[str, float]
    resources: Dict[str, float]  # wall_time, cpu_time, memory_mb
    timestamp: datetime = Field(default_factory=datetime.now)
    environment: Dict[str, str] = Field(default_factory=dict)


class ManifestPin(BaseModel):
    """Pin specification within a manifest."""
    model_config = {"arbitrary_types_allowed": True}
    
    function_id: FunctionID
    candidate: CandidateID
    metrics: Dict[str, float]
    evidence: Dict[str, Any]  # n_trials, confidence intervals, etc.


class Manifest(BaseModel):
    """Signed deployment manifest."""
    schema_version: str = "sigil/manifest@0"
    manifest_id: str
    spec: str
    created_at: datetime = Field(default_factory=datetime.now)
    author: str
    workspace: str
    codebase: Dict[str, str]  # repo, commit
    evaluator_set: List[Dict[str, Any]]
    policy: Dict[str, Any]
    pins: List[ManifestPin]
    signature: Optional[Dict[str, str]] = None


class SigilConfig(BaseModel):
    """Global Sigil configuration."""
    workspace_dir: Path = Field(default=Path.cwd() / ".sigil")
    tracker_enabled: bool = False
    resolver_mode: ResolverMode = ResolverMode.OFF
    default_spec: Optional[str] = None
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    max_iterations: int = 10
    sandbox_timeout: int = 30  # seconds
    sandbox_strategy: str = "subprocess"  # or "pyodide"
    sandbox_cpu_seconds: int = 1
    sandbox_wall_seconds: int = 2
    sandbox_memory_mb: int = 512
    sandbox_disable_network: bool = True
    # Optional Node project directory for Pyodide npm install
    pyodide_node_dir: Optional[Path] = None
