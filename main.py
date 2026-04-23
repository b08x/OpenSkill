"""
SkillCrafter — MemCollab-inspired Universal Skill Distillation Engine
FastAPI backend using OpenRouter for weak/strong agent pairing.
"""

import os
import json
import re
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="SkillCrafter", version="1.0.0")

SKILLS_DIR = Path("skills_output")
SKILLS_DIR.mkdir(exist_ok=True)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# ──────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────
WEAK_MODEL  = "openai/gpt-oss-120b"
STRONG_MODEL = "minimax/minimax-m2.5"

# ──────────────────────────────────────────────
# Request / Response schemas
# ──────────────────────────────────────────────
class CraftRequest(BaseModel):
    task: str
    api_key: str
    weak_model: Optional[str] = None
    strong_model: Optional[str] = None
    context_docs: Optional[list[dict]] = None

class SkillEntry(BaseModel):
    id: str
    title: str
    task: str
    created_at: str
    filename: str
    weak_model: str
    strong_model: str

class RetrieveRequest(BaseModel):
    query: str
    api_key: str
    model: Optional[str] = None
    top_k: Optional[int] = 3


# ──────────────────────────────────────────────
# OpenRouter helper
# ──────────────────────────────────────────────
def strip_reasoning_tags(text: str) -> str:
    """
    Removes reasoning tags (<think>...</think> or <thought>...</thought>)
    leaving only the model's final response. Ideal for cleaning JSON or list outputs.
    """
    # re.DOTALL allows '.' to match newlines
    # Non-greedy (.*?) combined with the boundary (</think> or end of text $) ensures
    # we clean the block even if the model exceeds token limits and doesn't close the tag.
    cleaned = re.sub(r'<(?:think|thought)>.*?(?:</(?:think|thought)>|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


async def classify_task(api_key: str, model: str, task: str) -> dict:
    """
    Classifies the task into Category and Subcategory for Task-Aware Retrieval.
    Inspired by Table 7 of the MemCollab paper.
    """
    system = (
        "You are an expert Task Classifier for an AI Memory System.\n"
        "Analyze the user's task and classify it into a main 'category' and 'subcategory'.\n\n"
        "Allowed Categories: Mathematics, Programming, Logic/Reasoning, General.\n"
        "Subcategories for Mathematics: Algebra, Geometry, Combinatorics, Number Theory, Calculus, Probability.\n"
        "Subcategories for Programming: Algorithms, Data Structures, Debugging, System Design, Web Dev.\n"
        "Subcategories for Logic/Reasoning: Puzzles, Planning, Fallacy Detection.\n\n"
        "Respond ONLY with a valid JSON in this exact format:\n"
        "{\n"
        '  "category": "Main Category",\n'
        '  "subcategory": "Specific Subcategory"\n'
        "}"
    )

    raw = await call_llm(api_key, model, [
        {"role": "system", "content": system},
        {"role": "user", "content": f"TASK TO CLASSIFY:\n{task}"}
    ], max_tokens=300)

    clean_text = strip_reasoning_tags(raw)

    try:
        match = re.search(r'(\{.*\})', clean_text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return json.loads(clean_text)
    except Exception as e:
        print(f"Task Classification failed: {e}. Raw: {clean_text}")
        return {"category": "General", "subcategory": "General"}

# ──────────────────────────────────────────────
# OpenRouter helper
# ──────────────────────────────────────────────
async def call_llm(api_key: str, model: str, messages: list, max_tokens: int = 2000) -> str:
    key = api_key or OPENROUTER_API_KEY
    if not key:
        raise HTTPException(status_code=400,
                            detail="No OpenRouter API key provided. Set OPENROUTER_API_KEY env var or enter it in the UI.")
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://skillcrafter.local",
        "X-Title": "SkillCrafter",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }

    # Timeout increased to 180s as Reasoning Models take longer
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(OPENROUTER_URL, headers=headers, json=payload)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=f"OpenRouter error: {resp.text}")
        data = resp.json()

        choices = data.get("choices") or []
        if not choices:
            raise HTTPException(status_code=502, detail=f"OpenRouter returned no choices. Response: {data}")
        message = choices[0].get("message") or {}

        # 1. Robust extraction of "content" (can be string or a list in multi-modal models)
        content_obj = message.get("content")
        content = ""
        if isinstance(content_obj, str):
            content = content_obj
        elif isinstance(content_obj, list):
            for item in content_obj:
                if isinstance(item, dict) and item.get("type") == "text":
                    content += item.get("text", "")

        # 2. Extraction of native reasoning from OpenRouter (if applicable)
        reasoning = message.get("reasoning") or ""
        if not reasoning and message.get("reasoning_details"):
            for detail in message["reasoning_details"]:
                if isinstance(detail, dict) and detail.get("text"):
                    reasoning += detail["text"]

        # 3. Unify and force <think> pattern for easy handling in the rest of the code
        text = ""
        if reasoning:
            text += f"<think>\n{reasoning}\n</think>\n\n"
        if content:
            text += content

        # 4. Fallbacks (old models, streaming edge cases)
        if not text.strip():
            text = message.get("text") or (message.get("delta") or {}).get("content") or ""

        if not text.strip():
            raise HTTPException(status_code=502, detail=f"Model returned empty content. Full response: {data}")

        return text.strip()


# ──────────────────────────────────────────────
# MemCollab pipeline steps
# ──────────────────────────────────────────────

async def generate_trajectory(api_key: str, model: str, task: str) -> str:
    messages = [
        {"role": "system", "content": (
            "You are a reasoning agent solving the given task step-by-step. "
            "Show your full reasoning process, including intermediate steps, "
            "any code or formulas, and your final answer. Be detailed."
        )},
        {"role": "user", "content": f"Task:\n{task}"}
    ]
    # Returns raw output (including native reasoning) to be evaluated later
    return await call_llm(api_key, model, messages, max_tokens=3000)


async def contrastive_analysis(api_key: str, model: str, task: str,
                               preferred: str, unpreferred: str) -> str:
    system = (
        "You are an expert analyst extracting reusable REASONING MEMORY from "
        "contrastive multi-step reasoning trajectories.\n\n"
        "Your goal is NOT to solve the problem. Your goal is to extract:\n"
        "1) Reusable failure-aware reasoning constraints\n"
        "2) High-level reasoning strategies that characterize correct reasoning\n\n"
        "Each extracted strategy MUST:\n"
        "- Be written as one sentence\n"
        "- Follow this format exactly: 'When ... , enforce ... ; avoid ...'\n"
        "- Be abstract and reusable across different problems\n\n"
        "Output ONLY a numbered list of strategies. No explanations. No preamble."
    )
    user = (
        f"TASK:\n{task}\n\n"
        f"PREFERRED TRAJECTORY (correct/better):\n{preferred}\n\n"
        f"UNPREFERRED TRAJECTORY (incorrect/weaker):\n{unpreferred}\n\n"
        "Extract 3-6 reusable reasoning constraints following the format: "
        "'When ..., enforce ...; avoid ...'"
    )
    raw_result = await call_llm(api_key, model, [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ], max_tokens=2000)

    # Remove <think> tags from evaluator model to return purely the list.
    return strip_reasoning_tags(raw_result)


async def synthesize_skill(api_key: str, model: str, task: str,
                           constraints: str, weak_traj: str, strong_traj: str) -> dict:
    system = (
        "You are a Skill Architect. Given a task and extracted reasoning constraints, "
        "synthesize a universal, reusable SKILL document in structured JSON.\n\n"
        "The skill must be model-agnostic — applicable to any AI or human solving similar tasks.\n"
        "Return ONLY valid JSON, no markdown fences."
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
        '  "invariants": ["essential principle 1", "essential principle 2", ...],\n'
        '  "violations": ["forbidden pattern 1", "forbidden pattern 2", ...],\n'
        '  "constraints": ["enforce X; avoid Y", ...],\n'
        '  "when_to_apply": "trigger description",\n'
        '  "example_pattern": "brief abstract example"\n'
        "}"
    )
    raw = await call_llm(api_key, model, [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ], max_tokens=2500)

    # Step 1: Clean thoughts (Reasoning Tags)
    clean_text = strip_reasoning_tags(raw)

    # Step 2: Robust JSON extraction via Regex
    try:
        # A. Try to find Markdown block with JSON (works even with text before/after)
        md_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', clean_text, re.DOTALL | re.IGNORECASE)
        if md_match:
            return json.loads(md_match.group(1))

        # B. If no Markdown, try to find the main outer key { ... }
        bracket_match = re.search(r'(\{.*\})', clean_text, re.DOTALL)
        if bracket_match:
            return json.loads(bracket_match.group(1))

        # C. Last attempt: direct load on cleaned string
        return json.loads(clean_text)

    except Exception as e:
        print(f"Synthesize fallback triggered! Parsing error: {e}")
        # Return fallback security structure if truly failed
        return {
            "title": "Skill Extraction Error",
            "domain": "General",
            "description": f"Raw Output Captured:\n{clean_text[:500]}...",
            "invariants": [],
            "violations": [],
            "constraints": constraints.split("\n") if constraints else [],
            "when_to_apply": task,
            "example_pattern": ""
        }

def render_skill_md(skill: dict, task: str, weak_model: str, strong_model: str,
                    weak_traj: str, strong_traj: str, constraints: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    invariants_md = "\n".join(f"- {i}" for i in skill.get("invariants", []))
    violations_md = "\n".join(f"- ⚠️ {v}" for v in skill.get("violations", []))
    constraints_md = "\n".join(f"- {c}" for c in skill.get("constraints", []))

    return f"""---
name: {skill.get('title', 'Unnamed Skill')}
domain: {skill.get('domain', 'General')}
generated: {now}
method: MemCollab Contrastive Trajectory Distillation
weak_agent: {weak_model}
strong_agent: {strong_model}
---

# {skill.get('title', 'Unnamed Skill')}

> {skill.get('description', '')}

## When to Apply

{skill.get('when_to_apply', task)}

---

## Reasoning Invariants
*(Principles that MUST be preserved for correct reasoning)*

{invariants_md}

---

## Violation Patterns
*(Forbidden patterns that lead to failure)*

{violations_md}

---

## Normative Constraints
*(Reusable rules: enforce ... ; avoid ...)*

{constraints_md}

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

*Generated by SkillCrafter using MemCollab — Cross-Agent Memory Collaboration via Contrastive Trajectory Distillation*
*Paper: https://arxiv.org/abs/2603.23234*
"""


# ──────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────

@app.post("/api/craft")
async def craft_skill(req: CraftRequest):
    """Main MemCollab pipeline: dual trajectory → contrastive analysis → Skill.md"""
    weak = req.weak_model or WEAK_MODEL
    strong = req.strong_model or STRONG_MODEL

    # Step 1: Generate trajectories from both agents
    try:
        weak_traj = await generate_trajectory(req.api_key, weak, req.task)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weak agent failed: {str(e)}")

    try:
        strong_traj = await generate_trajectory(req.api_key, strong, req.task)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Strong agent failed: {str(e)}")

    # Step 2: Contrastive analysis (strong model analyzes both)
    # Prefer strong trajectory as positive
    try:
        constraints = await contrastive_analysis(
            req.api_key, strong, req.task,
            preferred=strong_traj,
            unpreferred=weak_traj
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Contrastive analysis failed: {str(e)}")

    # Step 3: Synthesize structured skill
    try:
        skill_data = await synthesize_skill(
            req.api_key, strong, req.task,
            constraints, weak_traj, strong_traj
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Skill synthesis failed: {str(e)}")

    # --- NEW: Classify Task (NOW IN THE RIGHT PLACE) ---
    try:
        classification = await classify_task(req.api_key, strong, req.task)
    except Exception:
        classification = {"category": "General", "subcategory": "General"}

    # As skill_data was already created in Step 3, we can now inject the category into it!
    skill_data["category"] = classification.get("category", "General")
    skill_data["subcategory"] = classification.get("subcategory", "General")

    # Step 4: Render Skill.md
    skill_md = render_skill_md(
        skill_data, req.task, weak, strong,
        weak_traj, strong_traj, constraints
    )

    # Step 5: Save to disk
    skill_id = str(uuid.uuid4())[:8]
    safe_title = "".join(c if c.isalnum() or c in "-_" else "_"
                         for c in skill_data.get("title", "skill"))[:40]
    filename = f"{safe_title}_{skill_id}.md"
    filepath = SKILLS_DIR / filename
    filepath.write_text(skill_md, encoding="utf-8")

    # Step 6: Save metadata
    meta = {
        "id": skill_id,
        "title": skill_data.get("title", "Unnamed"),
        "task": req.task[:120],
        "category": skill_data["category"],
        "subcategory": skill_data["subcategory"],
        "created_at": datetime.now().isoformat(),
        "filename": filename,
        "weak_model": weak,
        "strong_model": strong,
    }
    meta_path = SKILLS_DIR / f"{skill_id}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "skill_id": skill_id,
        "title": skill_data.get("title"),
        "domain": skill_data.get("domain"),
        "category": skill_data["category"],
        "subcategory": skill_data["subcategory"],
        "filename": filename,
        "weak_trajectory": weak_traj,
        "strong_trajectory": strong_traj,
        "constraints": constraints,
        "skill": skill_data,
        "skill_md": skill_md,
    }


@app.get("/api/skills")
async def list_skills():
    """List all generated skills"""
    skills = []
    for meta_file in sorted(SKILLS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            meta = json.loads(meta_file.read_text())
            skills.append(meta)
        except Exception:
            pass
    return skills


@app.post("/api/retrieve")
async def retrieve_skills(req: RetrieveRequest):
    """
    Task-Aware Retrieval: Classifies the question and searches for compatible Skills.
    """
    model = req.model or STRONG_MODEL

    # 1. Classifies the user's question
    classification = await classify_task(req.api_key, model, req.query)
    target_cat = classification.get("category", "")
    target_sub = classification.get("subcategory", "")

    # 2. Scans saved skills and calculates a relevance score
    scored_skills = []
    for meta_file in SKILLS_DIR.glob("*.json"):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            score = 0
            if meta.get("category") == target_cat:
                score += 2
            if meta.get("subcategory") == target_sub:
                score += 3

            if score > 0:
                # Carrega o MD para enviar junto
                md_path = SKILLS_DIR / meta["filename"]
                content = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
                scored_skills.append({
                    "score": score,
                    "meta": meta,
                    "content": content
                })
        except Exception:
            continue

    # 3. Ordena pelas mais relevantes e retorna o Top K
    scored_skills.sort(key=lambda x: x["score"], reverse=True)
    top_skills = scored_skills[:req.top_k]

    return {
        "classification": classification,
        "results": top_skills
    }


@app.get("/api/skills/{skill_id}/download")
async def download_skill(skill_id: str):
    """Download a skill as Skills.md"""
    # Find the skill
    meta_path = SKILLS_DIR / f"{skill_id}.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Skill not found")
    meta = json.loads(meta_path.read_text())
    filepath = SKILLS_DIR / meta["filename"]
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Skill file not found")
    return FileResponse(
        path=str(filepath),
        filename="Skills.md",
        media_type="text/markdown"
    )


@app.get("/api/skills/{skill_id}")
async def get_skill(skill_id: str):
    """Get skill metadata and content"""
    meta_path = SKILLS_DIR / f"{skill_id}.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Skill not found")
    meta = json.loads(meta_path.read_text())
    filepath = SKILLS_DIR / meta["filename"]
    content = filepath.read_text(encoding="utf-8") if filepath.exists() else ""
    return {**meta, "content": content}


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    html = Path("templates/index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)