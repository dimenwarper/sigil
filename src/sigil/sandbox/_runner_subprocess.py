"""
Subprocess sandbox runner.

Reads JSON from stdin with fields:
{ code: str, func_name: str, args: list, kwargs: dict, limits: {...} }

Applies rlimits (POSIX), optional network/subprocess blocking, executes the
function, and prints a JSON result to stdout.
"""

from __future__ import annotations

import io
import json
import os
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr


def _apply_limits(lim: dict):
    # POSIX-only rlimits
    try:
        import resource  # type: ignore

        mb = 1024 * 1024
        cpu = int(lim.get("cpu_seconds", 1))
        mem = int(lim.get("memory_mb", 512)) * mb
        nofile = int(lim.get("nofile", 64))
        fsize = int(lim.get("fsize_mb", 16)) * mb

        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        # RLIMIT_AS can be flaky on macOS but still helps; ignore failures
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (nofile, nofile))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
        except Exception:
            pass
    except Exception:
        # Non-POSIX systems: best-effort only
        pass


def _disable_network_if_needed(disable: bool):
    if not disable:
        return
    try:
        import socket  # type: ignore

        def _block(*_a, **_k):  # noqa: ANN001, ANN002
            raise OSError("network disabled by sandbox")

        socket.socket = _block  # type: ignore
        socket.create_connection = _block  # type: ignore
        socket.getaddrinfo = _block  # type: ignore
    except Exception:
        pass

    try:
        import subprocess as _sp  # type: ignore

        class _BlockedPopen(_sp.Popen):  # type: ignore
            def __init__(self, *a, **k):  # noqa: D401, ANN002, ANN003
                raise OSError("subprocess disabled by sandbox")

        _sp.Popen = _BlockedPopen  # type: ignore
    except Exception:
        pass


def main():  # pragma: no cover - invoked as a child
    raw = sys.stdin.read()
    try:
        req = json.loads(raw)
    except Exception:
        print(json.dumps({"ok": False, "error": "invalid json request"}))
        return

    code = req.get("code", "")
    func_name = req.get("func_name")
    args = req.get("args") or []
    kwargs = req.get("kwargs") or {}
    limits = req.get("limits") or {}
    max_output_bytes = int(limits.get("max_output_bytes", 64 * 1024))

    _apply_limits(limits)
    _disable_network_if_needed(bool(limits.get("disable_network", True)))

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        # Execute code in a clean namespace
        ns: dict = {}
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(code, ns)
            if func_name not in ns or not callable(ns[func_name]):
                raise RuntimeError(f"function '{func_name}' not found after exec")
            result = ns[func_name](*args, **kwargs)

        out = stdout_buf.getvalue()[:max_output_bytes]
        err = stderr_buf.getvalue()[:max_output_bytes]
        print(json.dumps({"ok": True, "result": result, "stdout": out, "stderr": err}))
    except Exception:
        out = stdout_buf.getvalue()[:max_output_bytes]
        err = stderr_buf.getvalue()[:max_output_bytes]
        tb = traceback.format_exc()
        msg = (tb + "\n" + err)[-max_output_bytes:]
        print(json.dumps({"ok": False, "error": msg, "stdout": out, "stderr": err}))


if __name__ == "__main__":
    main()

