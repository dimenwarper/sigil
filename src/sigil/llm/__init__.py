"""
LLM client integration for Sigil.

Provides a small abstraction to call real LLMs via pydantic_ai when
available, with a safe fallback to the existing mock generator.
"""

from .providers import get_llm_client, LLMClientProtocol

__all__ = ["get_llm_client", "LLMClientProtocol"]

