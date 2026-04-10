"""
SkillEvolver — Trace2Skill: Fleet-Based Parallel Evolution
==========================================================
Full implementation of the Trace2Skill pipeline (arXiv:2603.25158).

Unlike traditional RAG, the Evolver doesn't just retrieve; it REWRITES
reasoning guidelines based on empirical evidence of failures and successes.

Mechanics:
  1. Batching: Splits N trajectories among a fleet of sub-agents.
  2. Patch Proposal: Each sub-agent proposes a 'Skill Patch' (JSON Diff).
  3. Hierarchical Merge: A merge operator consolidates patches, keeping
     only those that are prevalent (signal > noise).
  4. Application: Applies structured changes to the original Markdown.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Any

import structlog
from openskill.llm.base import BaseLLMProvider, LLMMessage

log = structlog.get_logger()

# ── Trace2Skill Settings ─────────────────────────────────────────────

FLEET_BATCH_SIZE = 4  # Trajectories per sub-agent
MAX_FLEET_SIZE = 10  # Parallelism limit to avoid rate limiting
MIN_PREVALENCE = 0.3  # Only accepts patches seen in >30% of the batch trajectories


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class SkillPatch:
    """Represents a suggested change in a Skill section."""
    section: str  # 'Invariants', 'Violations', 'Constraints', etc.
    op: str  # 'append', 'replace', 'insert', 'remove'
    content: str  # The new text
    target: str = ""  # Anchor text for replace/insert
    justification: str = ""  # Why is this change necessary?
    prevalence: float = 0.5  # How common was the observed pattern (0.0 to 1.0)


@dataclass
class EvolutionResult:
    """Final result of the evolution process."""
    evolved_md: str
    patches_applied: list[SkillPatch]
    success_rate: float
    fleet_size: int
    patch_count: int


# ── Prompts ──────────────────────────────────────────────────────────────────

EVOLVER_SYSTEM_PROMPT = (
    "You are a Skill Evolution Sub-Agent (Trace2Skill framework).\n"
    "Analyze execution trajectories and propose targeted patches to improve a skill document.\n\n"
    "RULES:\n"
    "1. Only propose patches backed by OBSERVABLE patterns in the trajectories.\n"
    "2. Focus on patterns that repeat across MULTIPLE trajectories.\n"
    "3. Use the format 'avoid X; enforce Y' for constraints.\n"
    "4. Prevalence: 1.0 if seen in all trajectories, 0.5 if in half, etc.\n"
    "Output ONLY a JSON array of patches."
)

MERGE_SYSTEM_PROMPT = (
    "You are a Skill Merge Coordinator.\n"
    "Consolidate multiple patch sets into one coherent, non-redundant set.\n\n"
    "STRATEGY:\n"
    "1. Deduplicate: Merge similar suggestions into the best-worded one.\n"
    "2. Conflict Resolution: If patches contradict, keep the one with higher prevalence.\n"
    "3. Pattern Elevation: If multiple sub-agents found the same issue, increase its prevalence.\n"
    "Output ONLY a JSON array of merged patches."
)


# ── Main Class ─────────────────────────────────────────────────────────

class SkillEvolver:
    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm

    async def evolve(
            self,
            skill_md: str,
            trajectories: list[dict]
    ) -> EvolutionResult:
        """
        Executes the complete fleet evolution pipeline.

        trajectories: list of {"task": str, "trajectory": str, "success": bool}
        """
        if not trajectories:
            log.warning("evolver.no_trajectories")
            return EvolutionResult(skill_md, [], 0.0, 0, 0)

        success_rate = sum(1 for t in trajectories if t.get("success")) / len(trajectories)

        # 1. Stage 2: Parallel Patch Proposal
        batches = self._create_batches(trajectories)
        log.info("evolver.dispatch_fleet", num_batches=len(batches))

        patch_groups = await asyncio.gather(*[
            self._analyze_batch(skill_md, batch) for batch in batches
        ])
        patch_groups = [g for g in patch_groups if g]  # Remove failures

        # 2. Stage 3: Hierarchical Consolidation (Merge)
        consolidated_patches = await self._hierarchical_merge(skill_md, patch_groups)

        # 3. Stage 4: Patch Application
        evolved_md = self._apply_patches(skill_md, consolidated_patches)

        return EvolutionResult(
            evolved_md=evolved_md,
            patches_applied=consolidated_patches,
            success_rate=success_rate,
            fleet_size=len(batches),
            patch_count=len(consolidated_patches)
        )

    async def generate_trajectories(
            self,
            model: str,
            skill_md: str,
            tasks: list[str]
    ) -> list[dict]:
        """Uses the LLM to run the skill against tasks and generate test trajectories."""

        async def _run_task(task: str):
            prompt = f"Using the following SKILL GUIDE, solve the task.\n\nSKILL:\n{skill_md}\n\nTASK:\n{task}"
            # Note: The agent must report if it was successful at the end
            res = await self.llm.generate([LLMMessage(role="user", content=prompt)])
            success = "RESULT: SUCCESS" in res.content.upper()
            return {"task": task, "trajectory": res.content, "success": success}

        return await asyncio.gather(*[_run_task(t) for t in tasks[:MAX_FLEET_SIZE]])

    # ── Private Helpers ──────────────────────────────────────────────────────

    def _create_batches(self, trajectories: list[dict]) -> list[list[dict]]:
        """Splits trajectories into batches for the fleet."""
        size = FLEET_BATCH_SIZE
        return [trajectories[i:i + size] for i in range(0, len(trajectories), size)][:MAX_FLEET_SIZE]

    async def _analyze_batch(self, skill_md: str, batch: list[dict]) -> list[SkillPatch]:
        """Sub-agent analyzes a specific batch."""
        traj_str = ""
        for i, t in enumerate(batch):
            status = "SUCCESS" if t['success'] else "FAILURE"
            traj_str += f"\n--- Trajectory {i} [{status}] ---\nTask: {t['task']}\nTrace: {t['trajectory'][:1000]}...\n"

        user_msg = f"CURRENT SKILL:\n{skill_md}\n\nBATCH TO ANALYZE:\n{traj_str}\n\nPropose JSON patches."

        try:
            raw = await self.llm.generate([
                LLMMessage(role="system", content=EVOLVER_SYSTEM_PROMPT),
                LLMMessage(role="user", content=user_msg)
            ], max_tokens=2000, temperature=0.2)

            data = self._extract_json(raw.content)
            if isinstance(data, list):
                return [SkillPatch(**p) for p in data if self._is_valid_patch(p)]
            return []
        except Exception as e:
            log.error("evolver.subagent_error", error=str(e))
            return []

    async def _hierarchical_merge(self, skill_md: str, groups: list[list[SkillPatch]]) -> list[SkillPatch]:
        """Consolidates patches from all sub-agents (Recursive)."""
        if not groups: return []
        if len(groups) == 1: return groups[0]

        # To simplify, we perform a global merge. 
        # On a large scale (>100 patches), we would merge in pairs (tree).
        all_patches_json = json.dumps([p.__dict__ for group in groups for p in group], indent=2)

        user_msg = f"SKILL CONTEXT:\n{skill_md[:500]}...\n\nPATCHES TO MERGE:\n{all_patches_json}"

        raw = await self.llm.generate([
            LLMMessage(role="system", content=MERGE_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_msg)
        ], max_tokens=3000, temperature=0.1)

        data = self._extract_json(raw.content)
        if isinstance(data, list):
            # Filter by minimum prevalence to ensure quality
            return [SkillPatch(**p) for p in data if p.get('prevalence', 0) >= MIN_PREVALENCE]
        return groups[0]  # Fallback to the first group if it fails

    def _apply_patches(self, skill_md: str, patches: list[SkillPatch]) -> str:
        """Applies changes to the Markdown (Based on Regex/Heuristic)."""
        lines = skill_md.split("\n")

        for patch in patches:
            # Attempts to find the section (e.g., ## Normative Constraints)
            section_header = f"## {patch.section}"

            # 1. Append (simplest and safest)
            if patch.op == "append":
                found = False
                for i, line in enumerate(lines):
                    if section_header.lower() in line.lower():
                        # Inserts after the header or at the end of the section
                        lines.insert(i + 1, f"- {patch.content}")
                        found = True
                        break
                if not found:
                    lines.append(f"\n{section_header}\n- {patch.content}")

            # 2. Replace/Remove (requires target)
            elif patch.op in ["replace", "remove"] and patch.target:
                for i, line in enumerate(lines):
                    if patch.target.lower() in line.lower():
                        if patch.op == "replace":
                            lines[i] = f"- {patch.content}"
                        else:
                            lines.pop(i)
                        break

        # Adds evolution log to the end of the document
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        lines.append(f"\n---\n*Evolved on {now} via Trace2Skill ({len(patches)} prevalent patterns identified).*")

        return "\n".join(lines)

    def _extract_json(self, text: str) -> Any:
        """Robust helper for extracting JSON."""
        try:
            m = re.search(r'\[.*\]', text, re.DOTALL)
            if m: return json.loads(m.group(0))
            return json.loads(text)
        except:
            return None

    def _is_valid_patch(self, p: dict) -> bool:
        """Validates if the dictionary has the minimum fields of a SkillPatch."""
        return all(k in p for k in ["section", "op", "content"])
