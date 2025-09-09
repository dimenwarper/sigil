"""
Sample tracking system for Sigil.

Tracks function calls, inputs, outputs, and evaluations in an append-only log.
"""

import json
import os
import uuid
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from contextlib import contextmanager

from ..core.models import SampleRecord, ResolverMode, FunctionID, CandidateID
from ..core.config import get_config


class SampleTracker:
    """
    Append-only tracker for function calls and evaluations.
    """
    
    def __init__(self, workspace_dir: Optional[Path] = None):
        self.workspace_dir = workspace_dir or get_config().workspace_dir
        self.log_file = self.workspace_dir / "samples.jsonl"
        self.is_active = False
        
        # Ensure directory exists
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
    
    def start(self):
        """Start tracking samples."""
        self.is_active = True
        
        # Log tracking start
        self._write_log({
            "event": "tracking_started",
            "timestamp": datetime.now().isoformat(),
            "pid": os.getpid(),
            "python_version": sys.version
        })
    
    def stop(self):
        """Stop tracking samples."""
        if self.is_active:
            # Log tracking stop
            self._write_log({
                "event": "tracking_stopped", 
                "timestamp": datetime.now().isoformat(),
                "pid": os.getpid()
            })
        
        self.is_active = False
    
    def record_sample(
        self,
        function_id: FunctionID,
        inputs: Dict[str, Any],
        outputs: Any,
        execution_time: float,
        candidate_id: Optional[CandidateID] = None,
        resolver_mode: ResolverMode = ResolverMode.OFF,
        error: Optional[str] = None
    ):
        """Record a function call sample."""
        if not self.is_active:
            return
        
        trace_id = str(uuid.uuid4())
        
        # Create input/output summaries (sketches, not full data)
        inputs_summary = self._create_summary(inputs)
        outputs_summary = self._create_summary({"result": outputs, "error": error})
        
        # Basic metrics
        metrics = {
            "success": error is None,
            "execution_time": execution_time
        }
        
        # Resource usage (basic)
        resources = {
            "wall_time": execution_time,
            "cpu_time": time.process_time(),  # Approximate
            "memory_mb": 0  # Could use psutil for actual memory tracking
        }
        
        # Environment info
        environment = {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "platform": sys.platform
        }
        
        record = SampleRecord(
            trace_id=trace_id,
            function_id=function_id,
            candidate_id=candidate_id,
            resolver_mode=resolver_mode,
            inputs_summary=inputs_summary,
            outputs_summary=outputs_summary,
            metrics=metrics,
            resources=resources,
            environment=environment
        )
        
        self._write_log(record.model_dump())
    
    def _create_summary(self, data: Dict[str, Any], max_items: int = 10) -> Dict[str, Any]:
        """Create a privacy-aware summary of data."""
        summary = {}
        
        for key, value in data.items():
            if isinstance(value, (int, float, bool)):
                summary[key] = {"type": type(value).__name__, "value": value}
            elif isinstance(value, str):
                summary[key] = {
                    "type": "str", 
                    "length": len(value),
                    "sample": value[:50] + "..." if len(value) > 50 else value
                }
            elif isinstance(value, (list, tuple)):
                summary[key] = {
                    "type": type(value).__name__,
                    "length": len(value),
                    "sample_types": [type(x).__name__ for x in value[:5]]
                }
            elif isinstance(value, dict):
                summary[key] = {
                    "type": "dict",
                    "keys": list(value.keys())[:max_items],
                    "length": len(value)
                }
            else:
                summary[key] = {"type": type(value).__name__}
        
        return summary
    
    def _write_log(self, record: Dict[str, Any]):
        """Write a record to the append-only log."""
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
    
    def read_samples(self, function_id: Optional[FunctionID] = None, limit: Optional[int] = None) -> list:
        """Read samples from the log, optionally filtered."""
        if not self.log_file.exists():
            return []
        
        samples = []
        count = 0
        
        with open(self.log_file) as f:
            for line in f:
                if limit and count >= limit:
                    break
                
                try:
                    record = json.loads(line.strip())
                    
                    # Skip non-sample events
                    if "event" in record:
                        continue
                    
                    # Filter by function_id if specified
                    if function_id and record.get("function_id") != str(function_id):
                        continue
                    
                    samples.append(record)
                    count += 1
                except json.JSONDecodeError:
                    continue
        
        return samples


# Global tracker instance
_global_tracker: Optional[SampleTracker] = None


def get_tracker() -> SampleTracker:
    """Get the global sample tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = SampleTracker()
    return _global_tracker


def start_tracking():
    """Start sample tracking globally."""
    tracker = get_tracker()
    tracker.start()
    
    # Update config
    config = get_config()
    config.tracker_enabled = True


def stop_tracking():
    """Stop sample tracking globally."""
    tracker = get_tracker()
    tracker.stop()
    
    # Update config
    config = get_config()
    config.tracker_enabled = False


def is_tracking() -> bool:
    """Check if tracking is currently active."""
    return get_tracker().is_active


@contextmanager
def track_call(function_id: FunctionID, candidate_id: Optional[CandidateID] = None):
    """Context manager to track a function call."""
    start_time = time.time()
    error = None
    
    try:
        yield
    except Exception as e:
        error = str(e)
        raise
    finally:
        execution_time = time.time() - start_time
        
        if is_tracking():
            get_tracker().record_sample(
                function_id=function_id,
                inputs={},  # Would need to capture from context
                outputs=None,  # Would need to capture from context
                execution_time=execution_time,
                candidate_id=candidate_id,
                error=error
            )