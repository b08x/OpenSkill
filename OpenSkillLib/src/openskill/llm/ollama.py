"""
OllamaProvider — 100% Local and Free LLM
============================================
Connects to Ollama running locally on the developer's machine.
Zero API cost, zero data sent out.
"""

from __future__ import annotations

import httpx
from typing import AsyncIterator
import numpy as np

from openskill.llm.base import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
)


class OllamaProvider(BaseLLMProvider):
    """
    Provider that connects to local Ollama (http://localhost:11434).

    Ollama installation:
        curl -fsSL https://ollama.com/install.sh | sh
        ollama pull qwen2.5-coder:7b
    """

    OLLAMA_URL = "http://localhost:11434"

    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        embed_model: str = "nomic-embed-text",
        base_url: str | None = None,
        timeout: float = 180.0,
    ):
        self.model = model
        self.embed_model = embed_model
        self.base_url = (base_url or self.OLLAMA_URL).rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def model_id(self) -> str:
        return f"ollama/{self.model}"

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url, timeout=self.timeout
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int = 2000,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        # Converts to Ollama format
        ollama_messages = [
            {"role": m.role, "content": m.content} for m in messages
        ]

        resp = await self.client.post(
            "/api/chat",
            json={
                "model": self.model,
                "messages": ollama_messages,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
                "stream": False,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        return LLMResponse(
            content=data["message"]["content"],
            reasoning="",
            raw=data,
        )

    async def embed(self, text: str) -> list[float]:
        resp = await self.client.post(
            "/api/embeddings",
            json={"model": self.embed_model, "prompt": text},
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    # ── Soft Latent Injection support ────────────────────────────────────────

    async def generate_with_embeddings(
        self,
        prompt: str,
        skill_embeddings: list[list[float]],
        max_tokens: int = 2000,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Generates using Soft Latent Injection (S-Path-RAG).

        Skill vectors are injected as soft prompts
        DIRECTLY into the embedding tensor before generation.

        Works with HuggingFace models via transformers.
        For pure Ollama, we fallback to text.
        """
        # Ollama does not support direct embedding injection,
        # so we concatenate vectors as special context
        import base64
        import json

        emb_json = json.dumps(skill_embeddings)
        emb_b64 = base64.b64encode(emb_json.encode()).decode()

        context_marker = f"<skill_vectors>{emb_b64}</skill_vectors>"

        return await self.generate(
            messages=[
                LLMMessage(
                    role="user",
                    content=f"{context_marker}\n\n{prompt}",
                )
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
