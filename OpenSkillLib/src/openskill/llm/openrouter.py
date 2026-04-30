"""
OpenRouterProvider — Flexible API-based LLM Provider
===================================================
Conecta o OpenSkill a centenas de modelos via OpenRouter.
Ideal para o pipeline MemCollab (Weak vs Strong models).
"""

from __future__ import annotations

import json
import httpx
import structlog
from typing import Optional

from openskill.llm.base import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse
)
from openskill.utils.config import get_openrouter_key

log = structlog.get_logger()


class OpenRouterProvider(BaseLLMProvider):
    """
    Adapter para a API do OpenRouter.
    """

    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
            self,
            api_key: Optional[str] = None,
            default_model: str = "openai/gpt-4o-mini",
            timeout: float = 120.0
    ):
        self.api_key = api_key or get_openrouter_key()
        self.default_model = default_model
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "X-Title": "OpenSkill Library",
                "Content-Type": "application/json"
            },
            timeout=timeout
        )

    @property
    def model_id(self) -> str:
        return self.default_model

    async def generate(
            self,
            messages: list[LLMMessage],
            max_tokens: int = 2000,
            temperature: float = 0.7,
            **kwargs
    ) -> LLMResponse:
        """Gera resposta via OpenRouter."""

        # Permite sobrescrever o modelo por chamada (essencial para MemCollab)
        target_model = kwargs.get("model", self.default_model)

        payload = {
            "model": target_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        try:
            resp = await self._client.post(self.URL, json=payload)
            resp.raise_for_status()
            data = resp.json()

            choice = data["choices"][0]["message"]
            content = choice.get("content", "")

            # Extração de Reasoning (OpenRouter retorna em campo separado em alguns modelos)
            reasoning = data.get("choices", [{}])[0].get("reasoning", "")

            return LLMResponse(
                content=content,
                reasoning=reasoning,
                raw=data
            )
        except Exception as e:
            log.error("openrouter.error", error=str(e))
            raise

    async def embed(self, text: str) -> list[float]:
        """
        Gera embeddings usando o modelo de embedding padrão via API.
        Nota: OpenRouter redireciona para o modelo configurado (ex: text-embedding-3-small).
        """
        # Endpoint de embedding do OpenRouter (OpenAI-compatible)
        embed_url = "https://openrouter.ai/api/v1/embeddings"
        payload = {
            "model": "openai/text-embedding-3-small",
            "input": text
        }

        resp = await self._client.post(embed_url, json=payload)
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    async def close(self):
        await self._client.aclose()