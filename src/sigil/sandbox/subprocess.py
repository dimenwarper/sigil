from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .base import Sandbox, SandboxLimits, SandboxResult


class SubprocessSandbox:
    """Sandbox using a child Python process with POSIX rlimits and timeouts."""

    def __init__(self, python_executable: Optional[str] = None):
        self.python = python_executable or sys.executable
        # Path to the runner script
        self.runner_path = Path(__file__).with_name("_runner_subprocess.py")

    def run_function(
        self,
        *,
        code: str,
        func_name: str,
        args: Optional[list] = None,
        kwargs: Optional[dict] = None,
        limits: Optional[SandboxLimits] = None,
    ) -> SandboxResult:
        args = args or []
        kwargs = kwargs or {}
        limits = limits or SandboxLimits()

        payload = {
            "code": code,
            "func_name": func_name,
            "args": args,
            "kwargs": kwargs,
            "limits": asdict(limits),
        }

        env = os.environ.copy()
        # Sanitize potential proxy vars to reduce accidental network
        for k in list(env.keys()):
            if k.lower().endswith("_proxy"):
                env.pop(k, None)

        # Use a temporary working directory
        with tempfile.TemporaryDirectory(prefix="sigil_sbx_") as tmpdir:
            cmd = [self.python, "-S", "-u", str(self.runner_path)]

            # Use a new session to make killing process groups easier on POSIX
            start_new_session = platform.system() != "Windows"

            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=tmpdir,
                env=env,
                start_new_session=start_new_session,
            )

            try:
                out, err = proc.communicate(json.dumps(payload), timeout=limits.wall_seconds)
            except subprocess.TimeoutExpired:
                # Kill the child process and its group if possible
                try:
                    if start_new_session:
                        os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
                finally:
                    proc.kill()
                return SandboxResult(ok=False, error=f"timeout after {limits.wall_seconds}s")

            # Runner returns a single JSON line on stdout
            try:
                data = json.loads(out.strip() or "{}")
            except Exception:
                return SandboxResult(ok=False, error="invalid runner output", stdout=out[: limits.max_output_bytes], stderr=err[: limits.max_output_bytes])

            return SandboxResult(
                ok=bool(data.get("ok")),
                result=data.get("result"),
                error=data.get("error"),
                stdout=(data.get("stdout") or "")[: limits.max_output_bytes],
                stderr=(data.get("stderr") or "")[: limits.max_output_bytes],
                stats={"returncode": proc.returncode},
            )

