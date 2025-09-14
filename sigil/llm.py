from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from pydantic import BaseModel, Field


class PatchRequest(BaseModel):
    spec_uri: str
    objective: str = Field(default="Propose an improvement as a unified diff.")
    files: Dict[str, str]  # path -> full file content
    pins: Dict[str, str]  # pin_uri -> description
    constraints: Dict[str, Any] = Field(default_factory=dict)


class PatchResponse(BaseModel):
    patch: str = Field(description="A unified diff strictly limited to the pinned files.")
    rationale: Optional[str] = Field(default=None)


class LLMProvider:
    def propose(self, req: PatchRequest) -> PatchResponse:
        raise NotImplementedError


class OpenAICompatibleProvider(LLMProvider):
    """Very small wrapper for OpenAI-compatible chat completions.

    Reads config from env:
      - OPENAI_API_KEY
      - OPENAI_BASE_URL (optional; defaults to https://api.openai.com/v1)
      - OPENAI_MODEL (defaults to gpt-4o-mini)
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def propose(self, req: PatchRequest) -> PatchResponse:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert code optimizer. Return ONLY JSON matching the schema: "
                    "{\"patch\": string, \"rationale\": string}. The patch MUST be a unified diff."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Task: propose a unified diff improving code within the given pins only.\n"
                    f"Spec: {req.spec_uri}\n"
                    f"Objective: {req.objective}\n"
                    f"Pins: {json.dumps(req.pins)}\n"
                    f"Constraints: {json.dumps(req.constraints)}\n\n"
                    "Files (path -> content):\n" + "\n\n".join(f"=== {p} ===\n{c}" for p, c in req.files.items()) +
                    "\nReturn JSON only."
                ),
            },
        ]

        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": messages,
        }
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        content = data["choices"][0]["message"]["content"]
        # Attempt to parse JSON; if content has code fences, strip them
        try:
            text = content.strip()
            if text.startswith("```"):
                # remove first and last fence
                text = "\n".join(text.splitlines()[1:-1])
            obj = json.loads(text)
        except Exception as e:
            raise RuntimeError(f"LLM returned non-JSON: {content[:200]}...") from e
        return PatchResponse.model_validate(obj)


class StubProvider(LLMProvider):
    """A deterministic proposer for testing: increments constants within SIGIL regions.

    Not a real LLM; returns a minimal unified diff changing 'x = 1' to 'x = 2' when possible.
    """

    def propose(self, req: PatchRequest) -> PatchResponse:
        # Heuristic: find a python file containing 'x = 1' under a SIGIL region and flip to 2
        target_path = None
        original = None
        for p, content in req.files.items():
            if "SIGIL:BEGIN" in content and "x = 1" in content:
                target_path = p
                original = content
                break
        if not target_path or original is None:
            # default: no-op diff against the first file
            if not req.files:
                patch = ""
            else:
                first = next(iter(req.files.keys()))
                patch = f"--- a/{first}\n+++ b/{first}\n"
            return PatchResponse(patch=patch, rationale="no-op")

        # Build a simple unified diff replacing first occurrence
        lines = original.splitlines()
        old_idx = None
        for i, l in enumerate(lines, start=1):
            if l.strip() == "x = 1":
                old_idx = i
                break
        if old_idx is None:
            patch = f"--- a/{target_path}\n+++ b/{target_path}\n"
            return PatchResponse(patch=patch, rationale="no-op")

        hunk_header = f"@@ -{old_idx},1 +{old_idx},1 @@"
        patch = (
            f"--- a/{target_path}\n" f"+++ b/{target_path}\n" f"{hunk_header}\n" f"-x = 1\n" f"+x = 2\n"
        )
        return PatchResponse(patch=patch, rationale="increment constant inside pin region")

