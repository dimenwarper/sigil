from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .base import Sandbox, SandboxLimits, SandboxResult
from ..core.config import get_config


class PyodideSandbox:
    """Node + Pyodide sandbox (placeholder).

    Requires Node and a local pyodide runner script + assets.
    If unavailable, returns a structured error so callers can fallback.
    """

    def __init__(self, node_executable: Optional[str] = None):
        self.node = node_executable or shutil.which("node")
        self.runner_path = Path(__file__).with_name("pyodide_runner.mjs")

    def run_function(
        self,
        *,
        code: str,
        func_name: str,
        args: Optional[list] = None,
        kwargs: Optional[dict] = None,
        limits: Optional[SandboxLimits] = None,
    ) -> SandboxResult:
        if not self.node or not self.runner_path.exists():
            return SandboxResult(ok=False, error="pyodide sandbox unavailable (node or runner missing)")

        limits = limits or SandboxLimits()
        payload = {
            "code": code,
            "func_name": func_name,
            "args": args or [],
            "kwargs": kwargs or {},
            "limits": asdict(limits),
        }

        env = os.environ.copy()
        # Avoid network proxies; runner should not fetch resources
        for k in list(env.keys()):
            if k.lower().endswith("_proxy"):
                env.pop(k, None)

        # Choose working directory for Node so that 'pyodide' can be resolved
        # from a project-local node_modules if the user ran 'sigil sandbox setup-pyodide'
        config = get_config()
        node_cwd = None
        try:
            node_dir = getattr(config, "pyodide_node_dir", None)
            if node_dir:
                node_cwd = str(node_dir)
        except Exception:
            node_cwd = None

        try:
            proc = subprocess.Popen(
                [self.node, str(self.runner_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=node_cwd,
            )
        except Exception as e:
            return SandboxResult(ok=False, error=f"failed to start node: {e}")

        try:
            out, err = proc.communicate(json.dumps(payload), timeout=limits.wall_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            return SandboxResult(ok=False, error=f"timeout after {limits.wall_seconds}s (pyodide)")

        try:
            data = json.loads(out.strip() or "{}")
        except Exception:
            return SandboxResult(ok=False, error="invalid pyodide runner output", stdout=out, stderr=err)

        return SandboxResult(
            ok=bool(data.get("ok")),
            result=data.get("result"),
            error=data.get("error"),
            stdout=data.get("stdout") or "",
            stderr=data.get("stderr") or "",
            stats={"returncode": proc.returncode},
        )
