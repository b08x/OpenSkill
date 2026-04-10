# SkillCrafter 🔬

**Universal Skill Distillation via MemCollab — Contrastive Trajectory Analysis**


DEMO PROTOTYPE:  https://openskill.onrender.com/


> *Based on: "MEMCOLLAB: Cross-Agent Memory Collaboration via Contrastive Trajectory Distillation" (arxiv:2603.23234)*
<img width="1891" height="834" alt="image" src="https://github.com/user-attachments/assets/f80ad317-26a7-4068-b2cf-6eb5d0408b1d" />
<img width="1525" height="812" alt="image" src="https://github.com/user-attachments/assets/2e5c0a82-a5ea-45ae-860d-5ed136ba88ed" />


---

## What is this?

SkillCrafter implements the MemCollab pipeline as a web app. You provide a task/problem domain and your OpenRouter API key. The system then:

1. **Runs a weak agent** (small model) on the task → trajectory A
2. **Runs a strong agent** (large model) on the same task → trajectory B
3. **Contrastive analysis** — the strong model compares both trajectories, extracting:
   - **Reasoning Invariants** `(i_k)` — what the strong agent did right
   - **Violation Patterns** `(v_k)` — what the weak agent did wrong
4. **Normative Constraints** — distilled as: `(enforce i_k; avoid v_k)`
5. **Skills.md** — a model-agnostic, downloadable skill file ready to inject into any agent

---

## Setup

** Download main.py, templates/index.html for UI. ** 

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open: **http://localhost:8000**

---

## Requirements

- **Python 3.10+**
- **OpenRouter API Key** — get one at https://openrouter.ai
  - Free models are set by default (Llama 3.1 8B + Gemma 3 27B)
  - Swap to any OpenRouter model string for better results

---

## Default Models (Free Tier)

| Role | Model |
|------|-------|
| 🤖 Weak Agent | `meta-llama/llama-3.1-8b-instruct:free` |
| 🧠 Strong Agent | `google/gemma-3-27b-it:free` |

You can change these in the UI or use paid models like:
- `anthropic/claude-3.5-sonnet` (strong)
- `meta-llama/llama-3.3-70b-instruct` (strong)
- `openai/gpt-4o-mini` (weak)

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /` | GET | Web UI |
| `POST /api/craft` | POST | Run full MemCollab pipeline |
| `GET /api/skills` | GET | List all distilled skills |
| `GET /api/skills/{id}` | GET | Get a specific skill |
| `GET /api/skills/{id}/download` | GET | Download as `Skills.md` |

---

## Output Format — Skills.md

Each generated skill follows this structure:

```markdown
---
name: Skill Title
domain: Mathematics / Coding / etc.
generated: 2026-03-25
method: MemCollab Contrastive Trajectory Distillation
---

## Reasoning Invariants
- Essential principle that leads to success...

## Violation Patterns
- ⚠️ Forbidden pattern that causes failure...

## Normative Constraints
- When dealing with X, enforce Y; avoid Z
- ...
```

This format is designed to be injected directly into any agent's system prompt or knowledge base.

---

## Testings: https://github.com/IhateCreatingUserNames2/OpenSkill/tree/main/SkillTesting/MemCollab 

## The MemCollab Insight

> *"Memory distilled from a single agent often preserves that agent's biases, limiting usefulness when transferred to other agents. By contrasting trajectories from different models, we extract shared invariants while suppressing agent-specific artifacts."*

The key insight: `τ = f(s, b)` where `s` = task-relevant structure, `b` = agent-specific bias.
Contrastive distillation extracts `m = ψ(s)` — pure reasoning structure, bias-free.

---

*Paper: https://arxiv.org/abs/2603.23234*
