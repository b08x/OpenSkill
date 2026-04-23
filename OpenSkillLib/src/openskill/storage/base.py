"""
Storage Abstraction Layer — Adapter Pattern
==========================================
ALL storage adapters implement this interface.

This is what makes OpenSkill an Open Core product:
  - LocalDiskStore  → Free, 100% local, runs on dev's laptop
  - CloudSaaSStore  → Paid, managed cloud infrastructure
"""
# --- openskill/storage/base.py ---

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, fields
import json
from enum import Enum

class SkillType(str, Enum):
    ACTIVE = "active"     # Requires code invocation (EvoSkills Python Scripts)
    PASSIVE = "passive"   # Requires only latent space injection (S-Path RAG Cross-Attention)
    HYBRID = "hybrid"     # Has strong semantic rules AND utility scripts

@dataclass
class SkillVectorProfile:
    model: str = "unknown"
    dimension: int = 0
    provider: str = "unknown"
    qvector: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None

    def to_dict(self) -> dict:
        return {
            "model": self.model, "dimension": self.dimension,
            "provider": self.provider, "qvector": self.qvector,
            "embedding": self.embedding
        }


@dataclass
class SkillMetadata:
    # Required fields with default fallback to not break 'from_dict'
    id: str = ""
    title: str = "Untitled"
    skill_type: SkillType = SkillType.PASSIVE
    domain: str = "General"
    category: str = "General"
    subcategory: str = "General"

    # --- (Game Mechanics) ---
    level: int = 1  # Increases with each co-evolution (EvoSkills)
    xp: float = 0.0  # Accumulates based on production success rate
    last_success_rate: Optional[float] = None
    evolution_count: int = 0  # How many times Trace2Skill did a "respec" or patch
    mana_cost: int = 0  # Token estimate (Hotbar cost)

    task: str = ""
    filename: str = ""
    created_at: str = ""
    weak_model: str = ""
    strong_model: str = ""

    trajectory_count: int = 0
    last_evolved_at: Optional[str] = None

    # Multi-folder support (paths relative to skill bundle root)
    reference_files: list[str] = field(default_factory=list)  # List of reference doc filenames
    template_files: list[str] = field(default_factory=list)  # List of template filenames
    asset_files: list[str] = field(default_factory=list)    # List of asset filenames

    vectors: dict[str, SkillVectorProfile] = field(default_factory=dict)
    embedding: Optional[list[float]] = None
    qvector: Optional[dict] = None

    def to_dict(self) -> dict:
        d = dataclass_asdict_filter_none(self)
        d['skill_type'] = self.skill_type.value  # Serializes Enum
        if self.vectors:
            d["vectors"] = {k: v.to_dict() if hasattr(v, 'to_dict') else v for k, v in self.vectors.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SkillMetadata":
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(cls)}

        # Safely converts vector dictionary
        vectors_raw = d.get("vectors", {})
        vectors = {}
        if isinstance(vectors_raw, dict):
            for k, v in vectors_raw.items():
                if isinstance(v, dict):
                    # Filters profile fields as well
                    p_fields = {f.name for f in dataclasses.fields(SkillVectorProfile)}
                    p_data = {pk: pv for pk, pv in v.items() if pk in p_fields}
                    vectors[k] = SkillVectorProfile(**p_data)

        if 'skill_type' in d:
            d['skill_type'] = SkillType(d['skill_type'])

        # Filters skill fields
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        if "vectors" in filtered: del filtered["vectors"]

        return cls(**filtered, vectors=vectors)


@dataclass
class SkillGraphData:
    """Skill graph (nodes + edges)."""
    nodes: dict[str, dict] = field(default_factory=dict)
    edges: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"nodes": self.nodes, "edges": self.edges}

    @classmethod
    def from_dict(cls, d: dict) -> "SkillGraphData":
        return cls(nodes=d.get("nodes", {}), edges=d.get("edges", []))


def dataclass_asdict_filter_none(obj) -> dict:
    import dataclasses
    d = {}
    for f in dataclasses.fields(obj):
        val = getattr(obj, f.name)
        # If it's Ellipsis or None, we ignore it.
        if val is not None and val is not Ellipsis:
            d[f.name] = val
    return d


class BaseSkillStore(ABC):
    """
    Abstract interface for skill storage.

    Implement this interface to create a new adapter:
      1. LocalDiskStore    — saves to .md/.json files on disk
      2. CloudSaaSStore    — calls SaaS REST API
      3. RedisStore         — vectors in Redis (future example)
      4. PgVectorStore      — vectors in PostgreSQL + pgvector
    """

    @abstractmethod
    async def save_skill(
        self,
        skill_id: str,
        markdown: str,
        metadata: SkillMetadata,
    ) -> None:
        """Saves a skill (markdown + metadata)."""
        ...

    @abstractmethod
    async def get_skill_md(self, skill_id: str) -> Optional[str]:
        """Returns the Markdown content of a skill."""
        ...

    @abstractmethod
    async def get_skill_meta(self, skill_id: str) -> Optional[SkillMetadata]:
        """Returns the metadata of a skill."""
        ...

    @abstractmethod
    async def get_skill_bundle(self, skill_id: str) -> Optional[dict]:
        """Returns the complete skill bundle including all folders (reference, template, assets)."""
        ...

    @property
    def workspace_path(self) -> Optional[Path]:
        """
        Returns the local workspace path, if applicable.
        Returns None for pure cloud storage that doesn't support local neural weight caching.
        """
        return None

    @abstractmethod
    async def list_skills(self) -> list[SkillMetadata]:
        """Lists all skills in the store."""
        ...

    @abstractmethod
    async def delete_skill(self, skill_id: str) -> None:
        """Removes a skill from the store."""
        ...

    @abstractmethod
    def get_graph(self) -> SkillGraphData:
        """Returns the skill graph."""
        ...

    @abstractmethod
    async def update_graph(self, graph: SkillGraphData) -> None:
        """Updates the skill graph."""
        ...

    @abstractmethod
    async def save_embedding(
            self,
            skill_id: str,
            embedding: list[float],
            qvector: dict,
            model_name: str,
            dimension: int,
            provider: str
    ) -> None:
        """Saves the quantized vector (TurboQuant) of a skill."""
        ...

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_uri(cls, uri: str, **kwargs) -> "BaseSkillStore":
        """
        Factory that returns the correct adapter based on URI.

        Examples:
          LocalDiskStore.from_uri("local:./skills")
          CloudSaaSStore.from_uri("cloud://osk_live_xxx?workspace=acme")
          RedisStore.from_uri("redis://localhost:6379/0")
        """
        scheme = uri.split("://")[0] if "://" in uri else "local"

        if scheme == "local":
            path = uri.replace("local://", "").strip() or "./skills"
            return cls(path=path, **kwargs)  # type: ignore[call-arg]
        elif scheme == "cloud":
            api_key = kwargs.get("api_key", "")
            workspace = kwargs.get("workspace", "default")
            return cls(api_key=api_key, workspace=workspace, **kwargs)  # type: ignore[call-arg]
        else:
            raise ValueError(f"Unknown storage scheme: {scheme}")
