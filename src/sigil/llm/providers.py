"""
LLM provider clients.

Currently supports pydantic_ai with OpenAI as the first provider.
Falls back to a deterministic mock if dependencies or keys are missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from ..core.config import get_config


class LLMClientProtocol(Protocol):
    def generate(self, prompt: str, *, system_prompt: str = "", temperature: Optional[float] = None) -> str:  # pragma: no cover - thin wrapper
        ...


@dataclass
class MockLLMClient:
    """Fallback client used when real providers are unavailable."""

    def generate(self, prompt: str, *, system_prompt: str = "", temperature: Optional[float] = None) -> str:
        # Keep the previous simple behavior to ensure local usage still works
        mock_improvement = (
            "# Mock improvement (pydantic_ai not available)\n"
            "def improved_function():\n"
            "    return None\n"
        )
        return mock_improvement


class PydanticAIClient:
    """pydantic_ai-backed client using provider-specific models."""

    def __init__(self, provider: str, model_name: str):
        self.provider = provider
        self.model_name = model_name

        try:
            from pydantic_ai import Agent  # type: ignore
            self._Agent = Agent
        except Exception as e:  # noqa: BLE001 - we want to capture ImportError and others
            raise RuntimeError("pydantic_ai is not installed") from e

        # Map provider to pydantic_ai model wrapper
        if provider.lower() == "openai":
            try:
                from pydantic_ai.models.openai import OpenAIModel  # type: ignore
            except Exception as e:  # noqa: BLE001
                raise RuntimeError("pydantic_ai OpenAI model is unavailable") from e
            self._model = OpenAIModel(self.model_name)
        else:
            # Extendable: add other providers here
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def generate(self, prompt: str, *, system_prompt: str = "", temperature: Optional[float] = None) -> str:
        # pydantic_ai agents currently take a model and a system prompt;
        # temperature can be configured on provider-specific models (not required here).
        agent = self._Agent(self._model, system_prompt=system_prompt or "")
        result = agent.run(prompt)

        # Try common result attributes
        text = getattr(result, "final", None)
        if text is None:
            data = getattr(result, "data", None)
            text = data if isinstance(data, str) else None
        if text is None:
            text = str(result)
        return text


def get_llm_client() -> LLMClientProtocol:
    """Factory returning the best-available LLM client.

    - If pydantic_ai and provider model are available, returns PydanticAIClient.
    - Otherwise, returns MockLLMClient.
    """
    cfg = get_config()
    provider = (cfg.llm_provider or "openai").lower()
    model_name = cfg.llm_model or "gpt-4o-mini"

    try:
        return PydanticAIClient(provider=provider, model_name=model_name)
    except Exception:
        # Graceful fallback to mock so local flows keep working
        return MockLLMClient()

