"""
CloudSaaSStore — Cloud Storage (Paid Mode)
====================================================
Sends all operations to the OpenSkill Cloud API.
This enables:
  1. Shared skills among team members
  2. Fleet Evolution as a Service (100 parallel agents)
  3. Centralized corporate knowledge graph
  4. Vector search in managed Redis/PgVector
"""

from __future__ import annotations

import httpx
from typing import Optional, TYPE_CHECKING

from openskill.storage.base import (
    BaseSkillStore,
    SkillMetadata,
    SkillGraphData,
)

if TYPE_CHECKING:
    pass


class CloudSaaSStore(BaseSkillStore):
    """
    Adapter that connects to OpenSkill Cloud API.

    Usage:
        store = CloudSaaSStore(
            api_key="osk_live_xxxxxxxxxxxx",
            workspace="acme-corp",
            base_url="https://api.openskill.ai",  # optional, default
        )
    """

    DEFAULT_BASE_URL = "https://api.openskill.ai"

    def __init__(
        self,
        api_key: str,
        workspace: str = "default",
        base_url: str | None = None,
        timeout: float = 60.0,
    ):
        self.api_key = api_key
        self.workspace = workspace
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "X-Workspace": self.workspace,
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _endpoint(self, path: str) -> str:
        return f"/v1/{self.workspace}{path}"

    # ── Graph ────────────────────────────────────────────────────────────────

    def get_graph(self) -> SkillGraphData:
        # SYNCHRONOUS for compatibility with S-Path-RAG local graph
        # In cloud production, this would be locally cached
        import json
        cache_path = f".openskill_graph_{self.workspace}.json"
        try:
            with open(cache_path) as f:
                return SkillGraphData.from_dict(json.load(f))
        except FileNotFoundError:
            return SkillGraphData()

    async def update_graph(self, graph: SkillGraphData) -> None:
        await self.client.put(
            self._endpoint("/graph"),
            json=graph.to_dict(),
        )

    # ── Skills CRUD ──────────────────────────────────────────────────────────

    async def save_skill(
        self,
        skill_id: str,
        markdown: str,
        metadata: SkillMetadata,
    ) -> None:
        await self.client.post(
            self._endpoint("/skills"),
            json={
                "skill_id": skill_id,
                "markdown": markdown,
                "metadata": metadata.to_dict(),
            },
        )

    async def get_skill_md(self, skill_id: str) -> Optional[str]:
        resp = await self.client.get(self._endpoint(f"/skills/{skill_id}/md"))
        if resp.status_code == 404:
            return None
        return resp.json().get("markdown")

    async def get_skill_meta(self, skill_id: str) -> Optional[SkillMetadata]:
        resp = await self.client.get(self._endpoint(f"/skills/{skill_id}"))
        if resp.status_code == 404:
            return None
        data = resp.json()
        # The endpoint returns "metadata" as a flat dict
        return SkillMetadata.from_dict(data.get("metadata", data))

    async def list_skills(self) -> list[SkillMetadata]:
        resp = await self.client.get(self._endpoint("/skills"))
        return [
            SkillMetadata.from_dict(item)
            for item in resp.json().get("skills", [])
        ]

    async def delete_skill(self, skill_id: str) -> None:
        await self.client.delete(self._endpoint(f"/skills/{skill_id}"))

    async def save_embedding(
        self,
        skill_id: str,
        embedding: list[float],
        qvector: dict,
        model_name: str,    # Added
        dimension: int,      # Added
        provider: str        # Added
    ) -> None:
        """Sends the vector profile to the OpenSkill Cloud API."""
        await self.client.post(
            self._endpoint(f"/skills/{skill_id}/embedding"),
            json={
                "embedding": embedding,
                "qvector": qvector,
                "model_name": model_name,
                "dimension": dimension,
                "provider": provider
            },
        )
