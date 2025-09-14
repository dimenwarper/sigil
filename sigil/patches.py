from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .spec import Spec
from .utils import content_addressed_path, ensure_dir, sha256_hex, write_text


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


def _find_allowed_regions(file_path: Path, pin_ids: List[str]) -> List[Tuple[int, int]]:
    # Regions are marked with lines like: "# SIGIL:BEGIN <id>" and "# SIGIL:END <id>"
    text = file_path.read_text().splitlines()
    regions: List[Tuple[int, int]] = []
    stack: Dict[str, int] = {}
    for idx, line in enumerate(text, start=1):
        m1 = re.search(r"SIGIL:BEGIN\s+([A-Za-z0-9_\-\.]+)", line)
        if m1:
            pid = m1.group(1)
            if pid in pin_ids:
                stack[pid] = idx
        m2 = re.search(r"SIGIL:END\s+([A-Za-z0-9_\-\.]+)", line)
        if m2:
            pid = m2.group(1)
            if pid in pin_ids and pid in stack:
                start = stack.pop(pid)
                regions.append((start, idx))
    return sorted(regions)


def _line_in_regions(line_no: int, regions: List[Tuple[int, int]]) -> bool:
    for a, b in regions:
        if a <= line_no <= b:
            return True
    return False


def validate_patch_against_pins(diff_text: str, spec: Spec, parent_root: Path) -> Tuple[bool, str]:
    """
    Validate that all changes in diff_text occur only within pin-declared files and
    within allowed regions marked by SIGIL markers that reference pin ids.
    """
    ud = parse_unified_diff(diff_text)
    # Build mapping: file -> allowed regions from relevant pins
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
        regions = _find_allowed_regions(abs_path, pin_file_map[rel_path])
        if not regions:
            return False, f"No SIGIL regions found for pinned ids in {rel_path}"
        # Walk hunks tracking original line cursor to validate +/- locations
        for h in f.hunks:
            oline = h.old_start
            nline = h.new_start
            for entry in h.lines:
                tag = entry[:1]
                if tag == ' ':
                    oline += 1
                    nline += 1
                elif tag == '-':
                    if not _line_in_regions(oline, regions):
                        return False, f"Deletion at {rel_path}:{oline} outside SIGIL regions"
                    oline += 1
                elif tag == '+':
                    # insertion position should be within or directly adjacent to a region by current original cursor
                    check_line = max(1, oline)
                    if not _line_in_regions(check_line, regions) and not _line_in_regions(check_line - 1, regions):
                        return False, f"Insertion near {rel_path}:{check_line} outside SIGIL regions"
                    nline += 1
                else:
                    # unknown marker
                    return False, f"Invalid hunk line marker in diff: {entry[:10]}"
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
    parent_id: str,
    seed: Optional[int] = None,
    prompt_json: Optional[Dict] = None,
) -> Tuple[str, Path]:
    canon = canonicalize_diff(patch_text)
    digest = sha256_hex(canon.encode("utf-8"))
    cdir = candidate_store_path(run_dir, digest)
    ensure_dir(cdir)
    write_text(cdir / "patch.diff", canon)
    write_text(cdir / "parent", parent_id)
    if seed is not None:
        write_text(cdir / "seed", str(seed))
    if prompt_json is not None:
        import json

        write_text(cdir / "prompt.json", json.dumps(prompt_json, indent=2))
    return digest, cdir
