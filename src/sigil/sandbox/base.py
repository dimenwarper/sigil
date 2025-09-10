from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol


@dataclass
class SandboxLimits:
    """Limits used by sandboxes.

    Units:
    - cpu_seconds: CPU time seconds (soft upper bound where supported)
    - wall_seconds: wall-clock timeout (parent enforced)
    - memory_mb: address space limit in MB (POSIX rlimit where supported)
    - nofile: max open files
    - fsize_mb: max file size writes in MB
    - disable_network: attempts to block Python-level networking inside sandbox
    - max_output_bytes: cap stdout/stderr collection
    """

    cpu_seconds: int = 1
    wall_seconds: int = 2
    memory_mb: int = 512
    nofile: int = 64
    fsize_mb: int = 16
    disable_network: bool = True
    max_output_bytes: int = 64 * 1024


@dataclass
class SandboxResult:
    ok: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)


class Sandbox(Protocol):
    def run_function(
        self,
        *,
        code: str,
        func_name: str,
        args: Optional[list] = None,
        kwargs: Optional[dict] = None,
        limits: Optional[SandboxLimits] = None,
    ) -> SandboxResult:  # pragma: no cover - interface only
        ...

