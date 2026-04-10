"""
LLM Provider Abstraction — Plugin Pattern
==========================================
Any LLM provider (OpenRouter, OpenAI, Anthropic, Ollama, vLLM)
implements this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class LLMMessage:
    role: str       # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    content: str
    reasoning: str = ""       # Thinking/reasoning tag (if available)
    raw: dict = None         # Raw API response


class BaseLLMProvider(ABC):
    """Interface that EVERY LLM provider must implement."""

    @abstractmethod
    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int = 2000,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        """
        Generates a text response.

        Args:
            messages: Message list in {role, content} format
            max_tokens: Token limit in response
            temperature: Sampling temperature
        Returns:
            LLMResponse with content and metadata
        """
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generates semantic embedding (for vector search)."""
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Model ID used."""
        ...

    async def close(self) -> None:
        """Resource cleanup (optional)."""
        pass
