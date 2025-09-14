from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .llm import LLMProvider, PatchRequest, PatchResponse, OpenAICompatibleProvider, StubProvider
from .spec import Spec, spec_uri, pin_uri


def build_patch_request(spec: Spec) -> PatchRequest:
    files: Dict[str, str] = {}
    pins: Dict[str, str] = {}
    for pin in spec.pins:
        pins[pin_uri(spec, pin)] = pin.id
        for f in (pin.files or []):
            path = spec.repo_root / f
            if path.exists():
                files[f] = path.read_text()
    return PatchRequest(
        spec_uri=spec_uri(spec),
        objective="Improve performance while preserving correctness; return a unified diff strictly within pin regions.",
        files=files,
        pins=pins,
        constraints={"edit_format": "unified_diff"},
    )


def get_provider(kind: str) -> LLMProvider:
    if kind == "openai":
        return OpenAICompatibleProvider()
    elif kind == "stub":
        return StubProvider()
    else:
        raise ValueError(f"Unknown provider: {kind}")


class BaseOptimizer:
    def propose(self, spec: Spec, provider: LLMProvider, num: int = 1) -> List[PatchResponse]:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class SimpleOptimizer(BaseOptimizer):
    def propose(self, spec: Spec, provider: LLMProvider, num: int = 1) -> List[PatchResponse]:
        req = build_patch_request(spec)
        responses: List[PatchResponse] = []
        n = max(1, int(num))
        for _ in range(n):
            responses.append(provider.propose(req))
        return responses

