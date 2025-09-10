"""
Sandbox interface and strategies.

Two initial strategies are provided:
- SubprocessSandbox: POSIX rlimits + wall-time with optional network blocking
- PyodideSandbox: placeholder Node-based runner (requires vendored Pyodide)
"""

from .base import Sandbox, SandboxLimits, SandboxResult
from .subprocess import SubprocessSandbox
from .pyodide import PyodideSandbox

def get_sandbox(strategy: str = "subprocess") -> Sandbox:
    """Factory for a sandbox instance.

    Strategies: "subprocess", "pyodide".
    """
    s = (strategy or "subprocess").lower()
    if s == "pyodide":
        return PyodideSandbox()
    return SubprocessSandbox()

__all__ = [
    "Sandbox",
    "SandboxLimits",
    "SandboxResult",
    "SubprocessSandbox",
    "PyodideSandbox",
    "get_sandbox",
]
