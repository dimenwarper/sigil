"""Simple Markdown formatter that removes trailing whitespace but keeps layout flexible."""

from __future__ import annotations

import re


def format_markdown(text: str) -> str:
    lines = []
    for raw in text.splitlines():
        # Collapse consecutive spaces but leave leading markers alone.
        prefix_match = re.match(r"^([#>*\-\s]*)(.*)$", raw)
        if not prefix_match:
            lines.append(raw.rstrip())
            continue
        prefix, body = prefix_match.groups()
        body = " ".join(part for part in body.split(" ") if part)
        lines.append(f"{prefix}{body}".rstrip())
    return "\n".join(lines) + "\n"
