"""
LocalDiskStore — 100% Local and Free Storage
======================================================
Saves skills as .md/.json files on the developer's disk.
Does not require an API key, does not send data to any server.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from openskill.storage.base import (
    BaseSkillStore,
    SkillMetadata,
    SkillGraphData, SkillVectorProfile, SkillType,
)

if TYPE_CHECKING:
    pass


class LocalDiskStore(BaseSkillStore):
    """
    Storage adapter that saves everything to the local disk.

    Directory structure:
        <path>/
        ├── skills/
        │   ├── raft_consensus_a493a4b2.md
        │   ├── paxos_01a3f8cd.md
        │   └── ...
        │   ├── raft_consensus_a493a4b2.json
        │   ├── paxos_01a3f8cd.json
        │   └── ...
        └── skill_graph.json
    """

    def __init__(
        self,
        path: str | Path = "./skills_output",
        create: bool = True,
    ):
        self.root = Path(path).resolve()
        self.skills_dir = self.root / "skills"

        if create:
            self.skills_dir.mkdir(parents=True, exist_ok=True)
            self._init_graph()

    def _init_graph(self) -> None:
        graph_path = self.root / "skill_graph.json"
        if not graph_path.exists():
            graph_path.write_text(
                json.dumps({"nodes": {}, "edges": []}),
                encoding="utf-8",
            )

    # ── Graph ────────────────────────────────────────────────────────────────

    def get_graph(self) -> SkillGraphData:
        graph_path = self.root / "skill_graph.json"
        try:
            data = json.loads(graph_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            data = {"nodes": {}, "edges": []}
        return SkillGraphData.from_dict(data)

    async def update_graph(self, graph: SkillGraphData) -> None:
        graph_path = self.root / "skill_graph.json"
        graph_path.write_text(
            json.dumps(graph.to_dict(), indent=2),
            encoding="utf-8",
        )

    # ── Skills CRUD ──────────────────────────────────────────────────────────

    async def save_skill(
            self,
            skill_id: str,
            markdown: str,
            metadata: SkillMetadata,
    ) -> None:
        """
        Mandatory implementation of the BaseSkillStore interface.
        Redirects to the new EvoSkills bundle system passing empty executable code.
        """
        await self.save_skill_bundle(skill_id, markdown, metadata, executable_code="")

    async def save_skill_bundle(
        self,
        skill_id: str,
        markdown: str,
        metadata: SkillMetadata,
        executable_code: str = ""
    ) -> None:
        """
        New Structure (EvoSkills Bundle):
        skills_output/
        └── <skill_id>/
            ├── SKILL.md            # Declarative Knowledge (MemCollab)
            ├── meta.json           # S-Path Vectors and Level (RPG Sheet)
            └── scripts/
                └── utils.py        # Active Code (EvoSkills)
        """
        bundle_dir = self.skills_dir / skill_id
        bundle_dir.mkdir(parents=True, exist_ok=True)

        # Update MD file route in metadata
        metadata.filename = f"{skill_id}/SKILL.md"

        # 1. Save the "Aura" (Passive rules in Markdown)
        md_path = bundle_dir / "SKILL.md"
        md_path.write_text(markdown, encoding="utf-8")

        # 2. Save "Status" (Meta.json)
        meta_path = bundle_dir / "meta.json"
        meta_path.write_text(
            json.dumps(metadata.to_dict(), indent=2),
            encoding="utf-8",
        )

        # 3. Save "Active Magic" (Python Script)
        if executable_code or metadata.skill_type in [SkillType.ACTIVE, SkillType.HYBRID]:
            scripts_dir = bundle_dir / "scripts"
            scripts_dir.mkdir(exist_ok=True)
            code_path = scripts_dir / "utils.py"
            code_path.write_text(executable_code, encoding="utf-8")

    async def get_skill_meta(self, skill_id: str) -> Optional[SkillMetadata]:
        sid = skill_id.strip()
        # 1. New format: skills/{skill_id}/meta.json (save_skill_bundle)
        bundle_meta = self.skills_dir / sid / "meta.json"
        if bundle_meta.exists():
            try:
                return SkillMetadata.from_dict(json.loads(bundle_meta.read_text(encoding="utf-8")))
            except Exception:
                pass
        # 2. Legacy: {skill_id}.json in skills/ root
        for f in self.skills_dir.glob(f"{sid}.json"):
            try:
                return SkillMetadata.from_dict(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass
        # 3. Broad fallback (anywhere)
        for f in self.root.rglob("meta.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if data.get("id") == sid:
                    return SkillMetadata.from_dict(data)
            except Exception:
                pass
        return None

    async def get_skill_md(self, skill_id: str) -> Optional[str]:
        meta = await self.get_skill_meta(skill_id)
        if meta is None: return None

        # Search for Markdown wherever it is
        paths = [self.skills_dir / meta.filename, self.root / meta.filename]
        for p in paths:
            if p.exists():
                return p.read_text(encoding="utf-8")
        return None

    async def list_skills(self) -> list[SkillMetadata]:
        metas: list[SkillMetadata] = []
        # rglob searches EVERYTHING (meta.json files inside folders and .json at root)
        files = list(self.root.rglob("meta.json")) + list(self.root.glob("*.json"))

        seen_ids = set()
        for f in files:
            if f.name == "skill_graph.json": continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if "id" not in data: continue

                meta = SkillMetadata.from_dict(data)
                if meta.id not in seen_ids:
                    metas.append(meta)
                    seen_ids.add(meta.id)
            except Exception:
                continue
        return sorted(metas, key=lambda m: m.created_at, reverse=True)

    async def delete_skill(self, skill_id: str) -> None:
        meta = await self.get_skill_meta(skill_id)
        if meta:
            (self.skills_dir / meta.filename).un_exists(missing_ok=True)
        (self.skills_dir / f"{skill_id}.json").unlink(missing_ok=True)

    @property
    def workspace_path(self) -> Optional[Path]:
        return self.root

    async def save_embedding(
            self,
            skill_id: str,
            embedding: list[float],
            qvector: dict,
            model_name: str,
            dimension: int,
            provider: str
    ) -> None:
        meta = await self.get_skill_meta(skill_id)
        if meta is None: return

        # Create or update specific profile
        profile_key = f"{model_name}_{dimension}".replace("/", "_")
        meta.vectors[profile_key] = SkillVectorProfile(
            model=model_name,
            dimension=dimension,
            provider=provider,
            qvector=qvector,
            embedding=embedding
        )

        # Maintain compatibility with old fields
        meta.qvector = qvector
        meta.embedding = embedding

        # FIXED: save in bundle (meta.json) if it exists, otherwise in legacy (.json)
        # Before it always saved in {skill_id}.json, ignoring the bundle — bootstrap
        # did not find vectors because it read from skills/{id}/meta.json.
        bundle_meta = self.skills_dir / skill_id / "meta.json"
        if bundle_meta.exists():
            meta_path = bundle_meta
        else:
            # Legacy: skill created before bundle system
            meta_path = self.skills_dir / f"{skill_id}.json"

        meta_path.write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")