"""
SkillCrafter — MemCollab: Contrastive Trajectory Distillation
=============================================================
Full implementation of the MemCollab pipeline (arXiv:2603.23234).

Core Principle:
  Agent memories are agent-specific (biases, style, heuristics).
  To create transferable memories between DIFFERENT models,
  we contrast trajectories from a STRONG vs. WEAK model on the SAME problem.
  What is INVARIANT between both = transferable principle.
  What is specific to the weak model = bias to be eliminated.

5-Stage Pipeline:
  1. Dual Trajectory Generation   → Generates τ_weak and τ_strong for the same task
  2. Contrastive Analysis          → Extracts constraints (invariants + violations)
  3. Task Classification           → Category/Subcategory (for task-aware retrieval)
  4. Skill Synthesis              → Assembles structured skill JSON
  5. Markdown Render               → Produces final SKILL.md

Does not perform I/O — receives LLMProvider and returns pure data.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import structlog

from openskill.core.verifier import SurrogateVerifier
from openskill.llm.base import BaseLLMProvider, LLMMessage

log = structlog.get_logger()


def _slugify(text: str) -> str:
    """Converts 'Optimal Fibonacci' to 'optimal-fibonacci' (Agent Skills standard)."""
    import re
    # Remove special characters, replace spaces with hyphens, and convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s-]', '', text).strip().lower()
    text = re.sub(r'[\s-]+', '-', text)
    return text[:64] # Standard requires max 64 chars

# ── Output Data Classes ──────────────────────────────────────────────────────

@dataclass
class DualTrajectories:
    """Pair of trajectories for the same problem."""
    task: str
    weak_model: str
    strong_model: str
    weak_trajectory: str
    strong_trajectory: str
    weak_success: bool = False
    strong_success: bool = False

    @property
    def preferred(self) -> tuple[str, bool]:
        """Returns (preferred trajectory, whether it came from the strong model)."""
        if self.strong_success:
            return self.strong_trajectory, True
        if self.weak_success:
            return self.weak_trajectory, False
        # No success — use strong as it is more capable
        return self.strong_trajectory, True

    @property
    def unpreferred(self) -> tuple[str, bool]:
        """Returns (unpreferred trajectory, whether it came from the weak model)."""
        if self.strong_success:
            return self.weak_trajectory, False
        if self.weak_success:
            return self.strong_trajectory, True
        return self.weak_trajectory, False


@dataclass
class TaskClassification:
    """Result of task classification for task-aware retrieval."""
    category: str
    subcategory: str


@dataclass
class ExtractedConstraints:
    """
    Constraints extracted from contrastive analysis.
    Format: list of strings "When X, enforce Y; avoid Z"
    """
    items: list[str]
    reasoning: str = ""  # LLM reasoning trace

    def __len__(self) -> int:
        return len(self.items)

    def __bool__(self) -> bool:
        return len(self.items) > 0

    def to_list(self) -> list[str]:
        return self.items


@dataclass
class SkillData:
    """Structured skill data (before rendering)."""
    title: str
    domain: str
    description: str
    category: str
    subcategory: str
    invariants: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    when_to_apply: str = ""
    example_pattern: str = ""
    source_constraints: list[str] = field(default_factory=list)


# ── System Prompts ───────────────────────────────────────────────────────

CRAFTER_SYSTEM_PROMPT = (
    "You are an expert skill architect for AI reasoning agents.\n"
    "Generate structured, reusable skill documents in JSON format.\n"
    "Skills encode NORMATIVE CONSTRAINTS — rules that agents MUST follow "
    "to produce correct reasoning.\n\n"
    "Output ONLY valid JSON. No markdown fences. No preamble."
)

TRAJECTORY_SYSTEM_PROMPT = (
    "You are a reasoning agent solving the given task step-by-step.\n"
    "Show your FULL reasoning, intermediate steps, code/formulas, and final answer.\n"
    "Be precise. Every step must be logged."
)

TRAJECTORY_USER_TEMPLATE = "Task:\n{task}\n\nSolve this problem completely."


CONSTRASTIVE_SYSTEM_PROMPT = (
    "You are an expert analyst extracting reusable REASONING MEMORY from\n"
    "contrastive multi-step reasoning trajectories.\n\n"
    "Extract:\n"
    "1) Reusable failure-aware reasoning constraints\n"
    "2) High-level reasoning strategies that characterize CORRECT reasoning\n\n"
    "Each strategy MUST:\n"
    "- Be written as ONE sentence\n"
    "- Follow this FORMAT: 'When ... , enforce ... ; avoid ...'\n"
    "- Be ABSTRACT and REUSABLE across different problems\n"
    "- NOT reference specific numbers, constants, or task details\n\n"
    "Output ONLY a numbered list of strategies. No explanations. No preamble. "
    "No markdown."
)

CONSTRASTIVE_USER_TEMPLATE = (
    "TASK:\n{task}\n\n"
    "PREFERRED TRAJECTORY (correct reasoning):\n{preferred}\n\n"
    "UNPREFERRED TRAJECTORY (incorrect or suboptimal reasoning):\n{unpreferred}\n\n"
    "Extract 3-8 reusable reasoning constraints in the format:\n"
    "'When [condition], enforce [principle]; avoid [failure pattern]'"
)

SKILL_BUNDLE_USER_TEMPLATE = (
    "TASK DOMAIN: {task}\n"
    "CONSTRAINTS:\n{constraints}\n"
    "STRONG AGENT APPROACH:\n{strong_trajectory}\n\n"
    "You are creating a Multi-file EvoSkill Bundle. You must provide TWO outputs:\n"
    "1. A JSON object with the SKILL.md metadata (title, invariants, etc).\n"
    "2. Executable Python code containing utility functions to solve this task.\n\n"
    "Format your response exactly like this:\n"
    "```json\n{{ ... }}\n```\n"
    "```python\n# Your executable code here\n```"
)

REFINE_CODE_TEMPLATE = (
    "Your previous code attempt for task '{task}' failed the verification.\n\n"
    "CURRENT CODE:\n```python\n{current_code}\n```\n\n"
    "DIAGNOSTIC FEEDBACK:\n{diagnostic}\n\n"
    "Fix the errors and provide the updated executable Python code wrapped in ```python ... ```."
)


CLASSIFY_SYSTEM_PROMPT = (
    "You are an expert Task Classifier for an AI Memory System.\n"
    "Classify the task into 'category' and 'subcategory'.\n\n"
    "Categories and subcategories:\n"
    "  Mathematics: Algebra, Geometry, Combinatorics, Number Theory, Calculus, Probability\n"
    "  Programming: Algorithms, Data Structures, Debugging, System Design, Web Dev\n"
    "  Logic/Reasoning: Puzzles, Planning, Fallacy Detection\n"
    "  General: General\n\n"
    "Respond ONLY with valid JSON: {{\"category\": \"...\", \"subcategory\": \"...\"}}"
)

CLASSIFY_USER_TEMPLATE = "TASK:\n{task}"


SYNTHESIZE_USER_TEMPLATE = (
    "TASK DOMAIN: {task}\n\n"
    "EXTRACTED CONSTRAINTS:\n{constraints}\n\n"
    "STRONG AGENT APPROACH (reference):\n{strong_trajectory}\n\n" 
    "Generate a SKILL with this exact JSON structure:\n"
    "{{"
    '  "title": "short skill name",\n'
    '  "domain": "Mathematics | Programming | Logic/Reasoning | General",\n'
    '  "description": "what this skill teaches in one sentence",\n'
    '  "invariants": ["essential principle 1", ...],\n'
    '  "violations": ["forbidden pattern 1", ...],\n'
    '  "constraints": ["enforce X; avoid Y", ...],\n'
    '  "when_to_apply": "trigger description",\n'
    '  "example_pattern": "brief abstract example"\n'
    "}}"
)

# ── LLM Helpers (private) ──────────────────────────────────────────────────

async def _call_llm(
    llm: BaseLLMProvider,
    messages: list[LLMMessage],
    max_tokens: int,
    temperature: float = 0.7,
) -> str:
    """Basic LLM call with error handling."""
    try:
        resp = await llm.generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.content.strip()
    except Exception as e:
        log.error("crafter.llm.error", error=str(e))
        raise
def _strip_think(text: str) -> str:
    """
    Removes <think>...</think> reasoning blocks common in CoT (Chain of Thought) models.
    This prevents the model's thinking log from interfering with JSON or list parsing.
    """
    if not text:
        return ""
    # Removes <think> tags and everything inside them (non-greedy)
    # flags=re.DOTALL allows '.' to capture line breaks
    # flags=re.IGNORECASE handles variations like <THINK>
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()

class SkillCrafter:
    """
    MemCollab pipeline orchestrator.
    Transforms a raw task into a refined Skill.md.
    """

    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm

    async def generate_trajectories(
            self, task: str, weak_model: str, strong_model: str
    ) -> tuple[str, str]:
        """Generates tau_w and tau_s trajectories in parallel."""
        import asyncio

        async def _run(model_id: str):
            messages = [
                LLMMessage(role="system", content=TRAJECTORY_SYSTEM_PROMPT),
                LLMMessage(role="user", content=TRAJECTORY_USER_TEMPLATE.format(task=task)),
            ]
            # Note: Here the BaseLLMProvider must support model switching if it's OpenRouter,
            # or ignore it if it's a fixed local model.
            resp = await self.llm.generate(messages, max_tokens=3000)
            return resp.content

        # Parallel execution for performance
        return await asyncio.gather(_run(weak_model), _run(strong_model))

    async def co_evolve_skill_bundle(
            self,
            task: str,
            constraints: list[str],
            strong_trajectory: str,
            verifier: 'SurrogateVerifier',
            max_iters: int = 5
    ) -> tuple[dict, str]:
        """
        Algorithm 1 (EvoSkills): Co-Evolution Loop.
        Generates logic, writes code, tests, and refines iteratively.

        FIX: passes skill_code to generate_tests for correct function name inference. 
             diagnostic is never empty again.
        """
        constraints_str = "\n".join([f"- {c}" for c in constraints])

        messages = [
            LLMMessage(role="system", content=CRAFTER_SYSTEM_PROMPT),
            LLMMessage(role="user", content=SKILL_BUNDLE_USER_TEMPLATE.format(
                task=task,
                constraints=constraints_str,
                strong_trajectory=strong_trajectory[:800]
            )),
        ]

        # 1st Generation (One-Shot)
        raw_response = await _call_llm(self.llm, messages, max_tokens=3000)
        skill_json = self._extract_json(raw_response) or {"title": "Unnamed", "domain": "General"}
        skill_code = self._extract_python_code(raw_response)

        # Passes skill_code so the verifier can infer the main function name
        log.info("evoskills.generating_surrogate_tests", task=task[:30])
        test_code = await verifier.generate_tests(task, skill_code=skill_code)

        # Iterative Evolution Loop (co-evolutionary loop)
        for i in range(max_iters):
            log.info("evoskills.verifying_iteration", iteration=i + 1, max=max_iters)

            success, diagnostic = verifier.evaluate_in_sandbox(skill_code, test_code)

            if success:
                log.info("evoskills.verification_passed", iteration=i + 1)
                break

            # diagnostic now ALWAYS contains useful information
            log.warning(
                "evoskills.verification_failed",
                iteration=i + 1,
                diagnostic=diagnostic[:200],
            )

            if not skill_code.strip():
                # Empty code — generate from scratch instead of attempting refinement
                log.warning("evoskills.empty_code_regenerating", iteration=i + 1)
                raw_response = await _call_llm(self.llm, messages, max_tokens=3000)
                skill_json = self._extract_json(raw_response) or skill_json
                skill_code = self._extract_python_code(raw_response)
                # Update tests for the new code
                test_code = await verifier.generate_tests(task, skill_code=skill_code)
                continue

            # Refinement with real feedback from the verifier (Eq 5, 7 from EvoSkills)
            refine_msgs = [
                LLMMessage(
                    role="system",
                    content=(
                        "You are a code refinement agent. Fix the Python code based on "
                        "the test diagnostics below. Return ONLY the corrected Python code "
                        "wrapped in ```python ... ``` blocks. Do not add explanations."
                    ),
                ),
                LLMMessage(role="user", content=REFINE_CODE_TEMPLATE.format(
                    task=task,
                    current_code=skill_code,
                    diagnostic=diagnostic,
                )),
            ]
            refine_resp = await _call_llm(self.llm, refine_msgs, max_tokens=2500)
            new_code = self._extract_python_code(refine_resp)

            if new_code.strip():
                skill_code = new_code
                # Update tests if the function name has changed
                test_code = await verifier.generate_tests(task, skill_code=skill_code)

        return skill_json, skill_code

    def _extract_python_code(self, text: str) -> str:
        import re
        m = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
        return m.group(1).strip() if m else ""

    async def contrastive_analysis(
            self, task: str, preferred: str, unpreferred: str
    ) -> list[str]:
        """Extracts invariant lessons between the two attempts."""
        messages = [
            LLMMessage(role="system", content=CONSTRASTIVE_SYSTEM_PROMPT),
            LLMMessage(role="user", content=CONSTRASTIVE_USER_TEMPLATE.format(
                task=task, preferred=preferred, unpreferred=unpreferred
            )),
        ]
        raw = await _call_llm(self.llm, messages, max_tokens=2000, temperature=0.3)
        cleaned = _strip_think(raw)

        # Simple numbered list parsing
        items = re.findall(r'^\d+[\.\)]\s*(.*)', cleaned, re.MULTILINE)
        return items if items else [cleaned]

    async def classify_task(self, task: str) -> dict:
        """Determines the category for future geometric retrieval."""
        messages = [
            LLMMessage(role="system", content=CLASSIFY_SYSTEM_PROMPT),
            LLMMessage(role="user", content=CLASSIFY_USER_TEMPLATE.format(task=task)),
        ]
        raw = await _call_llm(self.llm, messages, max_tokens=200, temperature=0.0)
        return self._extract_json(raw) or {"category": "General", "subcategory": "General"}

    async def synthesize_skill(
            self,
            task: str,
            constraints: list[str],
            weak_trajectory: str,
            strong_trajectory: str
    ) -> dict:
        """Merges lessons into a structured Skill object."""
        constraints_str = "\n".join([f"- {c}" for c in constraints])

        messages = [
            LLMMessage(role="system", content=CRAFTER_SYSTEM_PROMPT),
            LLMMessage(role="user", content=SYNTHESIZE_USER_TEMPLATE.format(
                task=task,
                constraints=constraints_str,
                strong_trajectory=strong_trajectory[:800]  # Slice performed here!
            )),
        ]
        raw = await _call_llm(self.llm, messages, max_tokens=2500, temperature=0.5)
        return self._extract_json(raw) or {"title": "Extraction Error"}

    def render_markdown(
            self,
            skill: dict,
            task: str,
            weak_model: str,
            strong_model: str,
            weak_traj: str,
            strong_traj: str,
            constraints: list[str]
    ) -> str:
        """Renders the SKILL.md file 100% compatible with the Agent Skills standard."""

        title = skill.get('title', 'Unnamed Skill')
        slug_name = _slugify(title)

        # Prevents YAML breakage and limits size
        description = skill.get('description', f"Workflow and utilities for {task}").strip()
        when_to_apply = skill.get('when_to_apply', description).strip()
        yaml_desc = (description + " " + when_to_apply)[:1000]

        inv_md = "\n".join(f"- {i}" for i in skill.get("invariants", []))
        viol_md = "\n".join(f"- ⚠️ {v}" for v in skill.get("violations", []))
        con_md = "\n".join(f"- {c}" for c in skill.get("constraints", []))

        # IMPORTANT: no indentation at the beginning
        return f"""---
    name: {slug_name}
    description: "{yaml_desc}"
    metadata:
      domain: {skill.get('domain', 'General')}
      generated_by: OpenSkill EvoSkills Framework
      weak_agent: {weak_model}
      strong_agent: {strong_model}
    ---

    # {title}

    > {description}

    ## Available Scripts
    - `scripts/utils.py` - Core utility functions for this skill.

    ## Reasoning Invariants
    {inv_md}

    ## Violation Patterns
    {viol_md}

    ## Normative Constraints
    {con_md}

    ---
    
    ## Example Pattern
    {skill.get('example_pattern', 'No example provided.')}
    Or
    ```python
    import sys
    # Always use relative paths when executing inside the skill directory
    sys.path.insert(0, 'scripts')
    from utils import * ```
    Code
    ---
    
    ## Source: MemCollab Analysis
    *Analysis performed by contrasting {strong_model} against {weak_model}.*
    """

    def _extract_json(self, text: str) -> Optional[dict]:
        """Extracts JSON in an ultra-robust way, cleaning reasoning blocks."""
        text = _strip_think(text)
        try:
            # 1. Tries to find ```json ... ``` block
            import re
            m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            if m:
                return json.loads(m.group(1))

            # 2. Tries to find the first { and the last } in the entire text
            m = re.search(r'(\{.*\})', text, re.DOTALL)
            if m:
                return json.loads(m.group(1))

            return json.loads(text)
        except Exception:
            # If it fails, tries to extract the title via simple Regex as a last resort
            title_match = re.search(r'"title":\s*"(.*?)"', text)
            if title_match:
                return {"title": title_match.group(1)}
            return None
