from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import shutil
import os
import tempfile

from .spec import Spec
from .utils import content_addressed_path, ensure_dir, sha256_hex, write_text


@dataclass(frozen=True)
class Candidate:
    id: str
    patch_text: str


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: List[str]  # includes leading markers ' ', '+', '-'


@dataclass
class FileDiff:
    path_old: str
    path_new: str
    hunks: List[Hunk]


@dataclass
class UnifiedDiff:
    files: List[FileDiff]


def parse_unified_diff(diff_text: str) -> UnifiedDiff:
    lines = diff_text.splitlines()
    i = 0
    files: List[FileDiff] = []
    while i < len(lines):
        line = lines[i]
        if line.startswith("--- "):
            path_old = line[4:].strip()
            i += 1
            if i >= len(lines) or not lines[i].startswith("+++ "):
                raise ValueError("Invalid unified diff: missing +++ after ---")
            path_new = lines[i][4:].strip()
            i += 1
            hunks: List[Hunk] = []
            while i < len(lines) and lines[i].startswith("@@"):
                m = re.match(r"@@ -([0-9]+)(?:,([0-9]+))? \+([0-9]+)(?:,([0-9]+))? @@", lines[i])
                if not m:
                    raise ValueError(f"Invalid hunk header: {lines[i]}")
                old_start = int(m.group(1))
                old_count = int(m.group(2) or 1)
                new_start = int(m.group(3))
                new_count = int(m.group(4) or 1)
                i += 1
                body: List[str] = []
                while i < len(lines) and (lines[i].startswith(" ") or lines[i].startswith("+") or lines[i].startswith("-")):
                    body.append(lines[i])
                    i += 1
                hunks.append(Hunk(old_start, old_count, new_start, new_count, body))
            files.append(FileDiff(path_old, path_new, hunks))
        else:
            # skip non-diff lines (e.g., git headers). Advance.
            i += 1
    return UnifiedDiff(files)




def validate_patch_against_pins(diff_text: str, spec: Spec, parent_root: Path) -> Tuple[bool, str]:
    """
    Validate that all changes in diff_text occur only within pin-declared files.
    """
    ud = parse_unified_diff(diff_text)
    # Build mapping: file -> pin ids
    pin_file_map: Dict[str, List[str]] = {}
    for p in spec.pins:
        for f in (p.files or []):
            pin_file_map.setdefault(f, []).append(p.id)

    for f in ud.files:
        # Normalize paths dropping possible prefixes like a/ b/
        def norm(p: str) -> str:
            s = p.split()[-1]
            if s.startswith("a/") or s.startswith("b/"):
                return s[2:]
            return s

        rel_path = norm(f.path_old if f.path_old != "/dev/null" else f.path_new)
        # Check file is in pin list
        if rel_path not in pin_file_map:
            return False, f"File not pinned for edits: {rel_path}"
        abs_path = parent_root / rel_path
        if not abs_path.exists():
            return False, f"Target file does not exist in parent: {rel_path}"
    return True, "ok"


def canonicalize_diff(diff_text: str) -> str:
    # Lightweight normalization: ensure trailing newline and LF line endings
    text = diff_text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.endswith("\n"):
        text += "\n"
    return text


def candidate_store_path(run_dir: Path, digest_hex: str) -> Path:
    return content_addressed_path(run_dir / "candidates", digest_hex)


def store_candidate(
    run_dir: Path,
    patch_text: str,
    seed: Optional[int] = None,
    parent_id: Optional[str] = None,
    prompt_json: Optional[Dict] = None,
) -> Tuple[str, Path]:
    canon = canonicalize_diff(patch_text)
    digest = sha256_hex(canon.encode("utf-8"))
    cdir = candidate_store_path(run_dir, digest)
    ensure_dir(cdir)
    write_text(cdir / "patch.diff", canon)
    if parent_id is not None:
        write_text(cdir / "parent", parent_id)
    if seed is not None:
        write_text(cdir / "seed", str(seed))
    if prompt_json is not None:
        import json

        write_text(cdir / "prompt.json", json.dumps(prompt_json, indent=2))
    return digest, cdir


def apply_unified_diff_in_tree(diff_text: str, tree_root: Path) -> None:
    """Apply a minimal subset of unified diffs to files under tree_root.

    Supports modifications (no file create/delete/rename)."""
    ud = parse_unified_diff(diff_text)
    for f in ud.files:
        # Normalize path
        def norm(p: str) -> str:
            s = p.split()[-1]
            if s.startswith("a/") or s.startswith("b/"):
                return s[2:]
            return s

        rel_path = norm(f.path_new if f.path_new != "/dev/null" else f.path_old)
        target = tree_root / rel_path
        if not target.exists():
            raise FileNotFoundError(f"Patch refers to missing file: {rel_path}")
        orig_lines = target.read_text().splitlines(keepends=True)
        out: List[str] = []
        idx = 1  # 1-based index over original
        for h in f.hunks:
            # Copy unchanged up to start of hunk
            while idx < h.old_start:
                out.append(orig_lines[idx - 1])
                idx += 1
            # Apply hunk lines
            for entry in h.lines:
                tag = entry[:1]
                content = entry[1:] + "\n"
                if tag == ' ':
                    # context: must match original
                    out.append(orig_lines[idx - 1])
                    idx += 1
                elif tag == '-':
                    # delete original line
                    idx += 1
                elif tag == '+':
                    out.append(content)
                else:
                    raise ValueError(f"Invalid hunk marker: {tag}")
        # Copy remainder
        while idx <= len(orig_lines):
            out.append(orig_lines[idx - 1])
            idx += 1
        target.write_text("".join(out))


def make_patched_worktree(src_repo: Path, diff_text: str) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="sigil-work-"))
    # Shallow copy tree (no VCS or venv)
    def should_copy(p: Path) -> bool:
        parts = set(p.parts)
        if any(x in parts for x in {".git", ".venv", "__pycache__", ".pytest_cache"}):
            return False
        return True

    for root, dirs, files in os.walk(src_repo):
        root_p = Path(root)
        # prune
        dirs[:] = [d for d in dirs if should_copy(root_p / d)]
        rel = root_p.relative_to(src_repo)
        (tmp / rel).mkdir(parents=True, exist_ok=True)
        for fn in files:
            src = root_p / fn
            if should_copy(src):
                dst = tmp / rel / fn
                shutil.copy2(src, dst)
    apply_unified_diff_in_tree(diff_text, tmp)
    return tmp
