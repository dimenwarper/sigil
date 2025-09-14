import hashlib
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_addressed_path(root: Path, digest: str) -> Path:
    return root / digest[:2] / digest[2:4] / digest


@dataclass
class CmdResult:
    returncode: int
    stdout: str
    stderr: str
    duration_s: float


def run_cmd(
    cmd: str,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    timeout_s: Optional[int] = None,
) -> CmdResult:
    start = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        executable=os.environ.get("SHELL", "/bin/bash"),
    )
    try:
        out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        duration = time.time() - start
        return CmdResult(returncode=124, stdout=out, stderr=err + "\nTIMEOUT", duration_s=duration)
    duration = time.time() - start
    return CmdResult(returncode=proc.returncode, stdout=out, stderr=err, duration_s=duration)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_text(p: Path, content: str) -> None:
    ensure_dir(p.parent)
    p.write_text(content)


def read_text(p: Path) -> str:
    return p.read_text()


def now_utc_compact() -> str:
    return time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())


def parse_regex_value(pattern: str, text: str) -> Optional[str]:
    # pattern format: "regex:(...)"
    m = re.match(r"^regex:(.*)$", pattern, re.DOTALL)
    if not m:
        return None
    expr = m.group(1)
    mm = re.search(expr, text, re.MULTILINE)
    return mm.group(1) if mm and mm.groups() else (mm.group(0) if mm else None)


def safe_env(base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    # Minimal, sanitized environment for subprocesses
    env = dict(base or os.environ)
    # Remove potentially sensitive vars by simple heuristics
    for k in list(env.keys()):
        lk = k.lower()
        if any(x in lk for x in ("token", "secret", "key", "password")):
            env.pop(k, None)
    return env


def which(prog: str) -> Optional[str]:
    for p in os.environ.get("PATH", "").split(os.pathsep):
        cand = Path(p) / prog
        if cand.exists() and os.access(cand, os.X_OK):
            return str(cand)
    return None

