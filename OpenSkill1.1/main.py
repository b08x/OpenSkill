"""
OpenSkill — Unified Skill Distillation & Geometric Memory Engine
================================================================
Integrates four research papers into a production skill lifecycle system:

  MemCollab   (arXiv:2603.23234) — Contrastive trajectory distillation
              Creates agent-agnostic skills from weak/strong agent pair.

  Trace2Skill (arXiv:2603.25158) — Parallel fleet evolution
              Evolves skills from N execution trajectories via sub-agent fleet
              + hierarchical consolidation. Avoids sequential overfitting.

  TurboQuant  (arXiv:2504.19874) — Geometric skill memory
              Each skill is embedded and quantized (random rotation + Lloyd-Max
              + QJL residual). Retrieval becomes nearest-neighbor in latent space
              instead of symbolic category matching.

  S-Path-RAG  (arXiv:2603.23512) — Semantic skill graph retrieval
              Skills form a typed graph (PREREQUISITE_OF, EXTENDS, etc.).
              Retrieval finds a PATH through the graph via Neural-Socratic loop:
              embed query → seed expansion → beam path search → confidence check
              → adaptive graph expansion if below threshold.

Architecture:
  Creation    → /api/craft       (MemCollab)
  Evolution   → /api/evolve      (Trace2Skill)
  Retrieval   → /api/retrieve    (TurboQuant + S-Path-RAG)
  Graph       → /api/graph       (S-Path-RAG graph inspection)
  Classic     → /api/retrieve/v1 (original category-based, kept for comparison)
"""

import os
import json
import re
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from local_llm import generate_with_soft_latents
# ── New modules ───────────────────────────────────────────────────────────────
from skill_vector import compute_and_store_embedding, cosine_similarity_raw
from skill_vector import dequantize_vector, unpack_quantized
from skill_graph  import (
    register_skill_in_graph,
    graph_retrieve,
    load_graph,
    get_graph_stats,
)
from skill_evolution import (
    evolve_skill,
    generate_evolution_trajectories,
)

app = FastAPI(title="OpenSkill", version="2.0.0")

SKILLS_DIR = Path("skills_output")
SKILLS_DIR.mkdir(exist_ok=True)

OPENROUTER_URL   = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

WEAK_MODEL   = "openai/gpt-oss-120b"
STRONG_MODEL = "minimax/minimax-m2.7"


# ── Request / Response Schemas ────────────────────────────────────────────────

class CraftRequest(BaseModel):
    task:         str
    api_key:      str
    weak_model:   Optional[str] = None
    strong_model: Optional[str] = None
    embed:        bool = True  # Compute embedding + register in graph

class EvolveRequest(BaseModel):
    skill_id:     str
    api_key:      str
    # Provide pre-existing trajectories OR tasks to auto-generate them
    trajectories: Optional[list[dict]] = None  # [{task, trajectory, success}]
    tasks:        Optional[list[str]]  = None  # Auto-generate trajectories for these
    analyst_model: Optional[str]       = None

class RetrieveRequest(BaseModel):
    query:   str
    api_key: str
    model:   Optional[str] = None
    top_k:   Optional[int] = 3
    use_graph: bool = True  # Use S-Path-RAG graph retrieval (vs legacy category)

class GraphEdgeRequest(BaseModel):
    api_key:   str
    from_id:   str
    to_id:     str
    edge_type: str  # PREREQUISITE_OF | EXTENDS | RESOLVES_ERROR | SIMILAR_TO | CONTRADICTS
    weight:    float = 0.8
    reason:    str = ""


# ── Utilities ─────────────────────────────────────────────────────────────────

def strip_reasoning_tags(text: str) -> str:
    return re.sub(
        r'<(?:think|thought)>.*?(?:</(?:think|thought)>|$)',
        '', text, flags=re.DOTALL | re.IGNORECASE
    ).strip()


async def call_llm(api_key: str, model: str, messages: list,
                   max_tokens: int = 2000) -> str:
    key = api_key or OPENROUTER_API_KEY
    if not key:
        raise HTTPException(400, "No OpenRouter API key provided.")
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://openskill.local",
        "X-Title": "OpenSkill",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(OPENROUTER_URL, headers=headers, json=payload)
        if resp.status_code != 200:
            raise HTTPException(resp.status_code, f"OpenRouter error: {resp.text}")
        data = resp.json()

    choices = data.get("choices") or []
    if not choices:
        raise HTTPException(502, "OpenRouter returned no choices.")
    message = choices[0].get("message") or {}

    content_obj = message.get("content")
    content = ""
    if isinstance(content_obj, str):
        content = content_obj
    elif isinstance(content_obj, list):
        for item in content_obj:
            if isinstance(item, dict) and item.get("type") == "text":
                content += item.get("text", "")

    reasoning = message.get("reasoning") or ""
    if not reasoning and message.get("reasoning_details"):
        for d in message["reasoning_details"]:
            if isinstance(d, dict) and d.get("text"):
                reasoning += d["text"]

    text = ""
    if reasoning:
        text += f"<think>\n{reasoning}\n</think>\n\n"
    text += content

    if not text.strip():
        text = message.get("text") or (message.get("delta") or {}).get("content") or ""
    if not text.strip():
        raise HTTPException(502, f"Model returned empty content. Response: {data}")
    return text.strip()


def load_all_metas() -> dict:
    """Load all skill metadata dicts keyed by skill_id."""
    metas = {}
    for f in SKILLS_DIR.glob("*.json"):
        if f.name == "skill_graph.json":
            continue
        try:
            meta = json.loads(f.read_text(encoding="utf-8"))
            sid = meta.get("id")
            if sid:
                metas[sid] = meta
        except Exception:
            pass
    return metas


# ── MemCollab Pipeline ────────────────────────────────────────────────────────

async def generate_trajectory(api_key: str, model: str, task: str) -> str:
    return await call_llm(api_key, model, [
        {"role": "system", "content": (
            "You are a reasoning agent solving the given task step-by-step. "
            "Show your full reasoning, intermediate steps, code/formulas, and final answer."
        )},
        {"role": "user", "content": f"Task:\n{task}"}
    ], max_tokens=3000)


async def contrastive_analysis(api_key: str, model: str, task: str,
                                preferred: str, unpreferred: str) -> str:
    system = (
        "You are an expert analyst extracting reusable REASONING MEMORY from "
        "contrastive multi-step reasoning trajectories.\n\n"
        "Extract:\n"
        "1) Reusable failure-aware reasoning constraints\n"
        "2) High-level reasoning strategies that characterize correct reasoning\n\n"
        "Each strategy MUST:\n"
        "- Be written as one sentence\n"
        "- Follow this format: 'When ... , enforce ... ; avoid ...'\n"
        "- Be abstract and reusable across different problems\n\n"
        "Output ONLY a numbered list of strategies. No explanations. No preamble."
    )
    raw = await call_llm(api_key, model, [
        {"role": "system", "content": system},
        {"role": "user", "content": (
            f"TASK:\n{task}\n\n"
            f"PREFERRED TRAJECTORY:\n{preferred}\n\n"
            f"UNPREFERRED TRAJECTORY:\n{unpreferred}\n\n"
            "Extract 3-6 reusable reasoning constraints: 'When ..., enforce ...; avoid ...'"
        )},
    ], max_tokens=2000)
    return strip_reasoning_tags(raw)


async def classify_task(api_key: str, model: str, task: str) -> dict:
    system = (
        "You are an expert Task Classifier for an AI Memory System.\n"
        "Classify the task into 'category' and 'subcategory'.\n\n"
        "Categories: Mathematics, Programming, Logic/Reasoning, General.\n"
        "Subcategories for Mathematics: Algebra, Geometry, Combinatorics, Number Theory, Calculus, Probability.\n"
        "Subcategories for Programming: Algorithms, Data Structures, Debugging, System Design, Web Dev.\n"
        "Subcategories for Logic/Reasoning: Puzzles, Planning, Fallacy Detection.\n\n"
        "Respond ONLY with JSON: {\"category\": \"...\", \"subcategory\": \"...\"}"
    )
    raw = await call_llm(api_key, model, [
        {"role": "system", "content": system},
        {"role": "user", "content": f"TASK:\n{task}"}
    ], max_tokens=200)
    clean = strip_reasoning_tags(raw)
    try:
        m = re.search(r'(\{.*\})', clean, re.DOTALL)
        return json.loads(m.group(1) if m else clean)
    except Exception:
        return {"category": "General", "subcategory": "General"}


async def synthesize_skill(api_key: str, model: str, task: str,
                            constraints: str, weak_traj: str,
                            strong_traj: str) -> dict:
    system = (
        "You are a Skill Architect. Synthesize a universal, reusable SKILL document in JSON.\n"
        "The skill must be model-agnostic. Return ONLY valid JSON, no markdown fences."
    )
    user = (
        f"TASK DOMAIN: {task}\n\n"
        f"EXTRACTED CONSTRAINTS:\n{constraints}\n\n"
        f"STRONG AGENT APPROACH (reference):\n{strong_traj[:800]}\n\n"
        "Synthesize a SKILL with this exact JSON structure:\n"
        "{\n"
        '  "title": "short skill name",\n'
        '  "domain": "e.g. Mathematics / Coding / Reasoning",\n'
        '  "description": "what this skill teaches",\n'
        '  "invariants": ["essential principle 1", ...],\n'
        '  "violations": ["forbidden pattern 1", ...],\n'
        '  "constraints": ["enforce X; avoid Y", ...],\n'
        '  "when_to_apply": "trigger description",\n'
        '  "example_pattern": "brief abstract example"\n'
        "}"
    )
    raw = await call_llm(api_key, model, [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ], max_tokens=2500)
    clean = strip_reasoning_tags(raw)
    for pattern in [r'```(?:json)?\s*(\{.*?\})\s*```', r'(\{.*\})']:
        m = re.search(pattern, clean, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    return {
        "title": "Skill Extraction Error", "domain": "General",
        "description": clean[:500], "invariants": [], "violations": [],
        "constraints": constraints.split("\n") if constraints else [],
        "when_to_apply": task, "example_pattern": "",
    }


def render_skill_md(skill: dict, task: str, weak_model: str, strong_model: str,
                    weak_traj: str, strong_traj: str, constraints: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    inv_md  = "\n".join(f"- {i}" for i in skill.get("invariants",  []))
    viol_md = "\n".join(f"- ⚠️ {v}" for v in skill.get("violations", []))
    con_md  = "\n".join(f"- {c}" for c in skill.get("constraints", []))

    return f"""---
name: {skill.get('title', 'Unnamed Skill')}
domain: {skill.get('domain', 'General')}
generated: {now}
method: MemCollab Contrastive Trajectory Distillation
weak_agent: {weak_model}
strong_agent: {strong_model}
papers: >
  MemCollab (arXiv:2603.23234)
  Trace2Skill (arXiv:2603.25158)
  TurboQuant (arXiv:2504.19874)
  S-Path-RAG (arXiv:2603.23512)
---

# {skill.get('title', 'Unnamed Skill')}

> {skill.get('description', '')}

## When to Apply

{skill.get('when_to_apply', task)}

---

## Reasoning Invariants
*(Principles that MUST be preserved for correct reasoning)*

{inv_md}

---

## Violation Patterns
*(Forbidden patterns that lead to failure)*

{viol_md}

---

## Normative Constraints
*(Reusable rules: enforce ... ; avoid ...)*

{con_md}

---

## Example Pattern

```
{skill.get('example_pattern', 'See constraints above.')}
```

---

## Source: Contrastive Trajectory Analysis

### Task

```
{task}
```

### Weak Agent Trajectory (`{weak_model}`)
<details>
<summary>Expand weak agent reasoning</summary>

{weak_traj}

</details>

### Strong Agent Trajectory (`{strong_model}`)
<details>
<summary>Expand strong agent reasoning</summary>

{strong_traj}

</details>

### Raw Extracted Constraints
```
{constraints}
```

---

*Generated by OpenSkill — MemCollab + Trace2Skill + TurboQuant + S-Path-RAG*
"""


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.post("/api/craft")
async def craft_skill(req: CraftRequest, background_tasks: BackgroundTasks):
    """
    MemCollab pipeline: dual trajectory → contrastive analysis → Skill.md
    Then (async): embed skill → register in graph (TurboQuant + S-Path-RAG)
    """
    weak   = req.weak_model   or WEAK_MODEL
    strong = req.strong_model or STRONG_MODEL

    # Step 1: Generate dual trajectories
    try:
        weak_traj, strong_traj = await asyncio.gather(
            generate_trajectory(req.api_key, weak,   req.task),
            generate_trajectory(req.api_key, strong, req.task),
        )
    except Exception as e:
        raise HTTPException(500, f"Trajectory generation failed: {e}")

    # Step 2: Contrastive analysis (MemCollab)
    try:
        constraints = await contrastive_analysis(
            req.api_key, strong, req.task,
            preferred=strong_traj, unpreferred=weak_traj
        )
    except Exception as e:
        raise HTTPException(500, f"Contrastive analysis failed: {e}")

    # Step 3: Synthesize structured skill
    try:
        skill_data = await synthesize_skill(
            req.api_key, strong, req.task,
            constraints, weak_traj, strong_traj
        )
    except Exception as e:
        raise HTTPException(500, f"Skill synthesis failed: {e}")

    # Step 4: Task classification (MemCollab task-aware retrieval)
    try:
        classification = await classify_task(req.api_key, strong, req.task)
    except Exception:
        classification = {"category": "General", "subcategory": "General"}

    skill_data["category"]    = classification.get("category", "General")
    skill_data["subcategory"] = classification.get("subcategory", "General")

    # Step 5: Render Skill.md
    skill_md = render_skill_md(
        skill_data, req.task, weak, strong, weak_traj, strong_traj, constraints
    )

    # Step 6: Persist skill + metadata
    skill_id  = str(uuid.uuid4())[:8]
    safe_title = "".join(c if c.isalnum() or c in "-_" else "_"
                         for c in skill_data.get("title", "skill"))[:40]
    filename  = f"{safe_title}_{skill_id}.md"
    filepath  = SKILLS_DIR / filename
    filepath.write_text(skill_md, encoding="utf-8")

    meta = {
        "id":           skill_id,
        "title":        skill_data.get("title", "Unnamed"),
        "task":         req.task[:120],
        "category":     skill_data["category"],
        "subcategory":  skill_data["subcategory"],
        "domain":       skill_data.get("domain", "General"),
        "created_at":   datetime.now().isoformat(),
        "filename":     filename,
        "weak_model":   weak,
        "strong_model": strong,
        "evolution_count": 0,
        "trajectory_count": 2,
    }
    meta_path = SKILLS_DIR / f"{skill_id}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Step 7: Background — compute TurboQuant embedding + register in S-Path-RAG graph
    if req.embed:
        async def embed_and_register():
            try:
                all_metas = load_all_metas()
                updated_meta = await compute_and_store_embedding(
                    req.api_key, skill_md, meta, meta_path
                )
                all_metas[skill_id] = updated_meta
                await register_skill_in_graph(
                    req.api_key, skill_id, updated_meta, all_metas
                )
            except Exception as ex:
                print(f"[OpenSkill] Background embed/graph failed for {skill_id}: {ex}")

        background_tasks.add_task(embed_and_register)

    return {
        "skill_id":          skill_id,
        "title":             skill_data.get("title"),
        "domain":            skill_data.get("domain"),
        "category":          skill_data["category"],
        "subcategory":       skill_data["subcategory"],
        "filename":          filename,
        "weak_trajectory":   weak_traj,
        "strong_trajectory": strong_traj,
        "constraints":       constraints,
        "skill":             skill_data,
        "skill_md":          skill_md,
        "embedding_status":  "pending_background" if req.embed else "disabled",
    }


@app.post("/api/evolve/{skill_id}")
async def evolve_skill_endpoint(skill_id: str, req: EvolveRequest):
    """
    Trace2Skill evolution: improve an existing skill from execution trajectories.

    Two modes:
      1. Pass `trajectories` directly: [{task, trajectory, success}]
      2. Pass `tasks` list: system generates trajectories automatically

    Pipeline:
      Stage 1 — Generate/receive trajectories
      Stage 2 — Parallel fleet of sub-agents proposes patches
      Stage 3 — Hierarchical consolidation (prevalence-weighted)
      Stage 4 — Apply patches to Skill.md

    Returns: evolved skill with patch log and statistics.
    """
    if req.skill_id != skill_id:
        raise HTTPException(400, "skill_id mismatch")

    meta_path = SKILLS_DIR / f"{skill_id}.json"
    if not meta_path.exists():
        raise HTTPException(404, f"Skill {skill_id} not found")

    meta     = json.loads(meta_path.read_text(encoding="utf-8"))
    filepath = SKILLS_DIR / meta["filename"]
    if not filepath.exists():
        raise HTTPException(404, "Skill file not found")

    skill_md = filepath.read_text(encoding="utf-8")
    analyst  = req.analyst_model or STRONG_MODEL

    # Stage 1: Obtain trajectories
    trajectories = req.trajectories or []
    if not trajectories and req.tasks:
        try:
            trajectories = await generate_evolution_trajectories(
                req.api_key, analyst, skill_md, req.tasks
            )
        except Exception as e:
            raise HTTPException(500, f"Trajectory generation failed: {e}")

    if not trajectories:
        raise HTTPException(400, "Provide either 'trajectories' or 'tasks'.")

    # Stages 2–4: Trace2Skill fleet evolution
    try:
        result = await evolve_skill(
            api_key       = req.api_key,
            analyst_model = analyst,
            skill_md      = skill_md,
            trajectories  = trajectories,
        )
    except Exception as e:
        raise HTTPException(500, f"Evolution failed: {e}")

    # Persist evolved skill
    evolved_md = result["evolved_md"]
    filepath.write_text(evolved_md, encoding="utf-8")

    # Update metadata
    meta["evolution_count"]   = meta.get("evolution_count", 0) + 1
    meta["trajectory_count"]  = meta.get("trajectory_count", 2) + len(trajectories)
    meta["last_evolved_at"]   = datetime.now().isoformat()
    meta["last_success_rate"] = result["success_rate"]
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Re-embed after evolution (skill content changed)
    try:
        all_metas = load_all_metas()
        await compute_and_store_embedding(req.api_key, evolved_md, meta, meta_path)
    except Exception as ex:
        print(f"[OpenSkill] Re-embedding after evolution failed: {ex}")

    return {
        "skill_id":       skill_id,
        "title":          meta.get("title"),
        "evolution_count": meta["evolution_count"],
        "patch_count":    result["patch_count"],
        "fleet_size":     result["fleet_size"],
        "success_rate":   result["success_rate"],
        "patches_applied": result["patches_applied"],
        "evolved_md":     evolved_md,
    }


@app.post("/api/retrieve")
async def retrieve_skills(req: RetrieveRequest):
    """
    Unified retrieval endpoint.

    When use_graph=True (default):
      TurboQuant vector similarity → S-Path-RAG path search
      → Neural-Socratic loop → returns skill path + confidence

    When use_graph=False:
      Legacy MemCollab category-based retrieval (kept for comparison)
    """
    model     = req.model or STRONG_MODEL
    all_metas = load_all_metas()

    if not all_metas:
        return {
            "query": req.query,
            "mode": "graph" if req.use_graph else "category",
            "results": [],
            "message": "No skills in library yet. Create skills with /api/craft first.",
        }

    if req.use_graph:
        try:
            result = await graph_retrieve(
                api_key=req.api_key,
                query=req.query,
                all_metas=all_metas,
                top_k=req.top_k,
            )
        except Exception as e:
            raise HTTPException(500, f"Graph retrieval failed: {e}")

        # === THE MAGIC OF VECTOR INJECTION (SOFT LATENTS) ===
        skill_vectors = []
        enriched_skills = []

        for skill_meta in result.get("skills", []):
            sid = skill_meta.get("id", "")

            # Get the vector quantized by TurboQuant and De-quantize it
            if "qvector" in skill_meta:
                qvec, res, scale = unpack_quantized(skill_meta["qvector"])
                dim = skill_meta["qvector"]["dim"]
                # Reconstruct the original vector mathematically (without using text)
                v_approx = dequantize_vector(qvec, res, scale, dim)
                skill_vectors.append(v_approx)

            # Only for display in the UI
            filename = skill_meta.get("filename", "")
            content = (SKILLS_DIR / filename).read_text() if filename else ""
            enriched_skills.append({**skill_meta, "content": content})

        # Instead of sending the skills as text in the prompt, we inject the VECTORS
        # directly into the local LLM's brain and ask it to solve the query.
        prompt_resolucao = f"Resolve this problem applying the reasoning guidelines encoded in your latent space:\n\n{req.query}"

        final_answer = await asyncio.to_thread(
            generate_with_soft_latents,
            prompt_resolucao,
            skill_vectors
        )

        return {
            "query": req.query,
            "mode": "graph_soft_injection",
            "paths": result.get("paths", []),
            "skills": enriched_skills,
            "confidence": result.get("confidence", 0.0),
            "rounds": result.get("rounds", 1),
            "reasoning_trace": result.get("reasoning_trace", ""),
            "final_answer": final_answer,  # <--- The Generated Answer via Geometry!
        }

    else:
        # ── Legacy MemCollab category-based retrieval ────────────────────────
        classification = await classify_task(req.api_key, model, req.query)
        target_cat = classification.get("category", "")
        target_sub = classification.get("subcategory", "")

        scored = []
        for sid, meta in all_metas.items():
            score = 0
            if meta.get("category")    == target_cat: score += 2
            if meta.get("subcategory") == target_sub: score += 3
            if score > 0:
                md_path = SKILLS_DIR / meta.get("filename", "")
                content = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
                scored.append({"score": score, "meta": meta, "content": content})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return {
            "query":          req.query,
            "mode":           "category",
            "classification": classification,
            "results":        scored[:req.top_k],
        }


@app.get("/api/graph")
async def get_graph():
    """Inspect the S-Path-RAG skill graph structure."""
    graph = load_graph()
    stats = get_graph_stats(graph)
    return {
        "stats":   stats,
        "nodes":   graph.get("nodes", {}),
        "edges":   graph.get("edges", []),
    }


@app.post("/api/graph/edge")
async def add_manual_edge(req: GraphEdgeRequest):
    """Manually add a typed edge to the skill graph."""
    from skill_graph import load_graph, add_edge, save_graph
    valid_types = {"PREREQUISITE_OF", "EXTENDS", "RESOLVES_ERROR", "SIMILAR_TO", "CONTRADICTS"}
    if req.edge_type not in valid_types:
        raise HTTPException(400, f"edge_type must be one of {valid_types}")
    graph = load_graph()
    add_edge(graph, req.from_id, req.to_id, req.edge_type, req.weight, req.reason)
    save_graph(graph)
    return {"status": "ok", "edge": {
        "from": req.from_id, "to": req.to_id,
        "type": req.edge_type, "weight": req.weight,
    }}


@app.post("/api/embed/{skill_id}")
async def recompute_embedding(skill_id: str, api_key: str):
    """
    (Re)compute TurboQuant embedding for a skill and update graph edges.
    Useful after manual skill edits or when embedding was skipped.
    """
    meta_path = SKILLS_DIR / f"{skill_id}.json"
    if not meta_path.exists():
        raise HTTPException(404, f"Skill {skill_id} not found")

    meta     = json.loads(meta_path.read_text(encoding="utf-8"))
    filepath = SKILLS_DIR / meta["filename"]
    if not filepath.exists():
        raise HTTPException(404, "Skill file not found")

    skill_md  = filepath.read_text(encoding="utf-8")
    all_metas = load_all_metas()

    try:
        updated_meta = await compute_and_store_embedding(api_key, skill_md, meta, meta_path)
        all_metas[skill_id] = updated_meta
        await register_skill_in_graph(api_key, skill_id, updated_meta, all_metas)
    except Exception as e:
        raise HTTPException(500, f"Embedding failed: {e}")

    return {"status": "ok", "skill_id": skill_id, "embed_version": updated_meta.get("embed_version")}


@app.get("/api/skills")
async def list_skills():
    """List all skills with metadata (including graph & embedding status)."""
    skills = []
    for f in sorted(SKILLS_DIR.glob("*.json"),
                    key=lambda p: p.stat().st_mtime, reverse=True):
        if f.name == "skill_graph.json":
            continue
        try:
            meta = json.loads(f.read_text(encoding="utf-8"))
            meta["has_embedding"]   = "embedding" in meta
            meta["evolution_count"] = meta.get("evolution_count", 0)
            # Strip large embedding array from list response
            meta.pop("embedding", None)
            meta.pop("qvector",   None)
            skills.append(meta)
        except Exception:
            pass
    return skills


@app.get("/api/skills/{skill_id}")
async def get_skill(skill_id: str):
    meta_path = SKILLS_DIR / f"{skill_id}.json"
    if not meta_path.exists():
        raise HTTPException(404, "Skill not found")
    meta     = json.loads(meta_path.read_text(encoding="utf-8"))
    filepath = SKILLS_DIR / meta["filename"]
    content  = filepath.read_text(encoding="utf-8") if filepath.exists() else ""
    # Include graph neighbors for this skill
    graph    = load_graph()
    neighbors = [e for e in graph.get("edges", [])
                 if e["from"] == skill_id or e["to"] == skill_id]
    return {**meta, "content": content, "graph_neighbors": neighbors}


@app.get("/api/skills/{skill_id}/download")
async def download_skill(skill_id: str):
    meta_path = SKILLS_DIR / f"{skill_id}.json"
    if not meta_path.exists():
        raise HTTPException(404, "Skill not found")
    meta     = json.loads(meta_path.read_text(encoding="utf-8"))
    filepath = SKILLS_DIR / meta["filename"]
    if not filepath.exists():
        raise HTTPException(404, "Skill file not found")
    return FileResponse(str(filepath), filename="SKILL.md", media_type="text/markdown")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    html_path = Path("templates/index.html")
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("""
    <h2>OpenSkill v2.0</h2>
    <p>API running. See <a href='/docs'>/docs</a> for Swagger UI.</p>
    <ul>
      <li>POST /api/craft — Create skill (MemCollab)</li>
      <li>POST /api/evolve/{id} — Evolve skill (Trace2Skill)</li>
      <li>POST /api/retrieve — Retrieve skills (TurboQuant + S-Path-RAG)</li>
      <li>GET  /api/graph — Inspect skill graph</li>
      <li>POST /api/graph/edge — Add manual edge</li>
      <li>POST /api/embed/{id} — Recompute embedding</li>
    </ul>
    """)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
