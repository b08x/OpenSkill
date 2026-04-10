"""
OpenSkillClient — Unified High-Level API
==============================================
The interface that EVERY end-user will use.

from openskill import OpenSkillClient, LocalDiskStore

client = OpenSkillClient(store=LocalDiskStore("./skills"))
skill = await client.craft("How to implement Raft consensus?")
answer = await client.retrieve("How to handle leader election in Raft?")
"""

from __future__ import annotations

import json
import uuid
import structlog
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from openskill.core.loadout import SkillLoadout
from openskill import LocalDiskStore
from openskill.storage.base import (
    BaseSkillStore,
    SkillMetadata,
    SkillGraphData,
)
from openskill.llm.base import BaseLLMProvider
from openskill.llm.openrouter import OpenRouterProvider

from openskill.core.crafter import SkillCrafter
from openskill.core.evolver import SkillEvolver
from openskill.core.vector import TurboQuantizer
from openskill.core.graph import SkillGraph, register_skill_in_graph

from openskill.retrieval.retriever import OpenSkillRetriever, RetrievalGuidance

log = structlog.get_logger()


class OpenSkillClient:
    """
    Unified client that abstracts the entire pipeline.

    Most common usage mode:

        from openskill import OpenSkillClient, LocalDiskStore

        client = OpenSkillClient(
            store=LocalDiskStore("./skills"),
            llm=OpenRouterProvider(api_key="sk-or-..."),
        )

        # Create skill via MemCollab
        skill = await client.craft(
            task="Implement Raft consensus protocol",
            weak_model="openai/gpt-4o-mini",
            strong_model="anthropic/claude-3-5-sonnet",
        )

        # Search guidance via TurboQuant + S-Path-RAG
        guidance = await client.retrieve(
            "How to handle network partitions in distributed systems?"
        )

        # Evolve skill via Trace2Skill
        evolved = await client.evolve(
            skill_id=skill.id,
            trajectories=[...],  # or tasks=[...] to auto-generate
        )
    """

    def __init__(
        self,
        store: BaseSkillStore | None = None,
        llm: BaseLLMProvider | None = None,
        *,
        skill_dir: str | Path = "./skills_output",
        default_weak_model: str = "openai/gpt-4o-mini",
        default_strong_model: str = "anthropic/claude-3-5-sonnet",
        embed: bool = True,
    ):
        # Storage: if not provided, use LocalDiskStore by default
        self.store = store or LocalDiskStore(skill_dir)

        # LLM Provider: if not provided, try OpenRouter (requires API key)
        self.llm = llm

        # Model defaults
        self.default_weak_model = default_weak_model
        self.default_strong_model = default_strong_model

        # Flags
        self.embed = embed

        # Internal components (lazy)
        self._crafter: Optional[SkillCrafter] = None
        self._evolver: Optional[SkillEvolver] = None
        self._quantizer: Optional[TurboQuantizer] = None
        self._graph: Optional[SkillGraph] = None
        self._retriever: Optional[OpenSkillRetriever] = None

    # ── Lazy init of components ─────────────────────────────────────────────

    @property
    def crafter(self) -> SkillCrafter:
        if self._crafter is None:
            self._crafter = SkillCrafter(llm=self.llm)
        return self._crafter

    @property
    def evolver(self) -> SkillEvolver:
        if self._evolver is None:
            self._evolver = SkillEvolver(llm=self.llm)
        return self._evolver

    @property
    def quantizer(self) -> TurboQuantizer:
        if self._quantizer is None:
            self._quantizer = TurboQuantizer()
        return self._quantizer

    @property
    def graph(self) -> SkillGraph:
        if self._graph is None:
            self._graph = SkillGraph(store=self.store)
        return self._graph

    @property
    def retriever(self) -> OpenSkillRetriever:
        if self._retriever is None:
            self._retriever = OpenSkillRetriever(store=self.store, llm=self.llm)
        return self._retriever

    # ── Public API ──────────────────────────────────────────────────────────
    async def prepare_quest_loadout(self, task_query: str) -> 'SkillLoadout':
        """
        The process of sitting at the bonfire before the Boss.
        1. S-Path RAG searches for knowledge in the Graph (Spellbook).
        2. The Loadout (Hotbar) selects and equips whatever fits.
        """
        # 1. Retrieve guidelines from the Graph (S-Path RAG)
        guidance = await self.retriever.retrieve(task_query, top_k=5, use_graph=True)

        # 2. Instantiate the Hotbar
        loadout = SkillLoadout(max_active_slots=3, max_passive_slots=5)

        # 3. Try to equip the returned skills (respecting slot limits)
        for i, skill_meta_dict in enumerate(guidance.skills_meta):
            meta = SkillMetadata.from_dict(skill_meta_dict)
            content = guidance.skill_contents[i]

            # Physical code path (EvoSkills Bundle)
            code_path = str(self.store.skills_dir / meta.id / "scripts")

            # Try to equip
            loadout.equip(meta, content, code_path=code_path)

        return loadout

    async def execute_quest(self, task_query: str):
        """The main gamified flow: Hotbar (Tokens) + Spellbook (Latent Vectors)."""
        from openskill.llm.base import LLMMessage

        # 1. Prepare the Hotbar (Decide what goes to Prompt and what goes to Geometry)
        loadout = await self.prepare_quest_loadout(task_query)

        print(
            f"\n[⚔️ Starting Quest with {len(loadout.equipped_active)} Active Skills and {len(loadout.equipped_passive)} Passive Auras]")

        # 2. Extract vectors from PASSIVE skills (To "bend the LLM geometry")
        passive_vectors = []
        for skill in loadout.equipped_passive:
            # Search for TurboQuant vector (Treating as Object)
            for profile_key, profile_data in skill.get('vectors', {}).items():
                dim = getattr(profile_data, 'dimension', 0)
                emb = getattr(profile_data, 'embedding', None)

                if dim == 384 and emb:
                    passive_vectors.append(emb)
                    break

        # 3. Get the "Instruction Manual" for ACTIVE skills (For the Prompt)
        active_prompt = loadout.generate_system_prompt_appendage()

        if active_prompt:
            print(f"[🛡️ Hotbar] Equipping Active Scripts: {[s['title'] for s in loadout.equipped_active]}")

        # 4. Build Agent Context
        system_content = "You are an AI Agent equipped with specialized skills to solve complex tasks."
        if active_prompt:
            system_content += f"\n{active_prompt}"

        messages = [
            LLMMessage(role="system", content=system_content),
            LLMMessage(role="user", content=task_query)
        ]

        # 5. Go to battle!
        # Check if the loaded LLM supports vector injection (Cross-Attention/S-Path)
        if hasattr(self.llm, 'generate_with_guidance') and passive_vectors:
            log.info("quest.execution", mode="S-PATH INJECTION + ACTIVE SKILLS")
            from openskill.retrieval.retriever import RetrievalGuidance

            # Assemble the latent orientation object
            guidance = RetrievalGuidance(
                query=task_query,
                best_path_ids=[s['id'] for s in loadout.equipped_passive],
                skill_vectors=passive_vectors,
                skill_alphas=[1.0 / len(passive_vectors)] * len(passive_vectors)  # Equal weights for now
            )
            # LocalSkillInjectedLLM will handle vector injection + generation
            response = await self.llm.generate_with_guidance(
                query=task_query,
                guidance=guidance,
                mode="auto"
            )
        else:
            # Fallback for OpenRouter/Ollama (No vector injection, only Active Skills in prompt)
            log.info("quest.execution", mode="API FALLBACK (ACTIVE SKILLS ONLY)")
            response = await self.llm.generate(messages=messages, max_tokens=2000)

        return response

    async def craft(
            self,
            task: str,
            *,
            weak_model: str | None = None,
            strong_model: str | None = None,
            skill_id: str | None = None,
            embed: bool | None = None,
    ) -> SkillMetadata:
        """
        EvoSkills + MemCollab pipeline:
        trajectory → contrastive analysis → co-evolutionary loop → Skill Bundle
        """
        sid = skill_id or str(uuid.uuid4())[:8]
        weak = weak_model or self.default_weak_model
        strong = strong_model or self.default_strong_model
        do_embed = embed if embed is not None else self.embed

        log.info("craft.start", skill_id=sid, task=task[:80])

        if self.llm is None:
            raise RuntimeError("LLM Provider not configured.")

        # 1. Generate dual trajectories (MemCollab)
        weak_traj, strong_traj = await self.crafter.generate_trajectories(task, weak, strong)

        # 2. Contrastive analysis (MemCollab)
        constraints = await self.crafter.contrastive_analysis(task, strong_traj, weak_traj)

        # 3. Co-Evolution Loop (EvoSkills)
        from openskill.core.verifier import SurrogateVerifier
        verifier = SurrogateVerifier(llm=self.llm)

        skill_data, executable_code = await self.crafter.co_evolve_skill_bundle(
            task, constraints, strong_traj, verifier
        )

        # (Warning: Removed the old line `self.crafter.synthesize_skill` that was here,
        # as it would overwrite the `skill_data` that was just evolved in EvoSkills!)

        # 4. Task classification
        classification = await self.crafter.classify_task(task)

        # 5. Render Markdown (Semantic Knowledge / SKILL.md)
        skill_md = self.crafter.render_markdown(
            skill_data, task, weak, strong, weak_traj, strong_traj, constraints
        )

        # 6. Build metadata (The "Character Sheet")
        from openskill.storage.base import SkillType
        from openskill.core.crafter import _slugify  # Import slugify we created

        # NEW: The skill ID is now the readable name in the Agent Skills standard!
        raw_title = skill_data.get("title", "skill")
        sid = _slugify(raw_title)

        safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in skill_data.get("title", "skill"))[:40]

        # Determine RPG taxonomy based on the existence of generated code
        s_type = SkillType.HYBRID if executable_code else SkillType.PASSIVE

        meta = SkillMetadata(
            id=sid,
            title=skill_data.get("title", "Unnamed"),
            skill_type=s_type,  # Defines how the skill will be treated by the Hotbar
            domain=skill_data.get("domain", "General"),
            category=classification.get("category", "General"),
            subcategory=classification.get("subcategory", "General"),
            task=task[:120],
            created_at=datetime.now(timezone.utc).isoformat(),
            weak_model=weak,
            strong_model=strong,
            evolution_count=0,
            trajectory_count=2,
            level=1  # Initial Level
        )

        # 7. Save Multi-file Skill Bundle
        if hasattr(self.store, 'save_skill_bundle'):
            await self.store.save_skill_bundle(sid, skill_md, meta, executable_code)
        else:
            await self.store.save_skill(sid, skill_md, meta)  # Fallback

        # 8. TurboQuant Embedding + register in graph
        if do_embed:

            # 1. Retrieve list of skills (which is a LIST of SkillMetadata)
            all_metas = await self.store.list_skills()

            # 2. Convert to dictionary {id: meta} as graph.py expects
            all_metas_dict = {m.id: m.to_dict() for m in all_metas}

            # 3. Register
            await self._embed_and_register(sid, skill_md, meta)

            await register_skill_in_graph(
                skill_id=sid,
                meta=meta,
                all_metas=all_metas_dict,  # Passing the dict now
                store=self.store,
                use_gnn=False
            )

        log.info("craft.done", skill_id=sid, title=meta.title, type=s_type.value)
        return meta

    async def evolve(
            self,
            skill_id: str,
            *,
            trajectories: list[dict] | None = None,
            tasks: list[str] | None = None,
            analyst_model: str | None = None,
    ) -> dict:
        analyst = analyst_model or self.default_strong_model

        # 1. Load skill and current metadata
        skill_md = await self.store.get_skill_md(skill_id)
        meta = await self.store.get_skill_meta(skill_id)

        if skill_md is None or meta is None:
            raise FileNotFoundError(f"Skill {skill_id} not found")

        # 2. Generate trajectories if necessary
        if not trajectories and tasks:
            trajectories = await self.evolver.generate_trajectories(
                analyst, skill_md, tasks
            )

        # 3. Fleet evolution
        result = await self.evolver.evolve(skill_md, trajectories or [])

        # 4. Update metadata (Increment evolution count)
        meta.evolution_count += result.patch_count
        meta.last_evolved_at = datetime.now(timezone.utc).isoformat()
        meta.last_success_rate = result.success_rate

        # 5. Persist evolved skill (now passing correctly loaded meta)
        await self.store.save_skill(skill_id, result.evolved_md, meta)

        # 6. Update graph
        # Ensure update_skill_node method exists in graph.py (as adjusted before)
        await self.graph.update_skill_node(skill_id, meta=meta.to_dict(), store=self.store)

        log.info("evolve.done", skill_id=skill_id, patches=result.patch_count)
        return result.__dict__

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 3,
        use_graph: bool = True,
    ) -> RetrievalGuidance: # Change from dict to RetrievalGuidance
        return await self.retriever.retrieve(query, top_k=top_k, use_graph=use_graph)

    async def list_skills(self) -> list[SkillMetadata]:
        """List all skills."""
        return await self.store.list_skills()

    async def get_skill(self, skill_id: str) -> dict:
        """Returns complete skill (metadata + markdown)."""
        md = await self.store.get_skill_md(skill_id)
        meta = await self.store.get_skill_meta(skill_id)
        if md is None:
            raise FileNotFoundError(f"Skill {skill_id} not found")
        graph = self.store.get_graph()
        neighbors = [
            e for e in graph.edges
            if e["from"] == skill_id or e["to"] == skill_id
        ]
        return {
            **(meta.to_dict() if meta else {}),
            "content": md,
            "graph_neighbors": neighbors,
        }

    async def pull(self, skill_name: str) -> SkillMetadata:
        """
        Download a skill from OpenSkill Hub (public registry).

        Usage:  skill = await client.pull("raft/high-latency-consensus")
        """
        # Delegates to CloudSaaSStore or download from public hub
        raise NotImplementedError("Hub integration coming soon")

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _embed_and_register(self, skill_id: str, skill_md: str, meta: SkillMetadata) -> None:
        try:
            vec = await self.quantizer.embed(self.llm, skill_md)
            embedding_list = vec.tolist()
            qv = self.quantizer.quantize(vec)

            # Persist vector profile via save_embedding (clean path)
            await self.store.save_embedding(
                skill_id=skill_id,
                embedding=embedding_list,
                qvector=qv.to_dict(),
                model_name=self.llm.model_id,
                dimension=len(embedding_list),
                provider="OpenAI" if "openai" in self.llm.model_id.lower() else "LocalMiniLM",
            )

            # Register in graph
            all_metas = await self.store.list_skills()
            all_metas_dict = {m.id: m.to_dict() for m in all_metas}
            await register_skill_in_graph(
                skill_id=skill_id,
                meta=meta,
                all_metas=all_metas_dict,
                store=self.store,
            )
            log.info("embed_and_register.success", skill_id=skill_id)
        except Exception as e:
            log.warning("embed_and_register.failed", skill_id=skill_id, error=str(e))
