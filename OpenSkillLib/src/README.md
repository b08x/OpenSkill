# OpenSkill — Geometric Skill Memory for LLM Agents

> **A unified lifecycle engine for creating, evolving, storing, and retrieving structured knowledge (Skills) for LLM-based agents — grounded in four research papers.**

---

## Table of Contents

1. [What is OpenSkill?](#what-is-openskill)
2. [Architecture Overview](#architecture-overview)
3. [Research Foundations](#research-foundations)
   - [MemCollab — Contrastive Trajectory Distillation](#memcollab)
   - [Trace2Skill — Fleet-Based Skill Evolution](#trace2skill)
   - [TurboQuant — Near-Optimal Vector Quantization](#turboquant)
   - [S-Path-RAG — Semantic Graph Retrieval](#s-path-rag)
   - [EvoSkills — Self-Evolving Skill Generation](#evoskills)
4. [Installation](#installation)
5. [Core Concepts](#core-concepts)
   - [Skills vs Tools](#skills-vs-tools)
   - [Skill Bundle Structure](#skill-bundle-structure)
   - [Skill Types (Active / Passive / Hybrid)](#skill-types)
6. [How to Use OpenSkill](#how-to-use-openskill)
   - [Creating New Skills](#creating-new-skills)
   - [Training the Path Scorer](#training-the-path-scorer)
   - [Distilling Skills from Trajectories](#distilling-skills-from-trajectories)
   - [Retrieving Skills at Inference](#retrieving-skills-at-inference)
   - [Using SKILL.md Files](#using-skillmd-files)
7. [Vector Injection — What It Is and Why It Matters](#vector-injection)
   - [Prefix Injection (Gap 2)](#prefix-injection-gap-2)
   - [Cross-Attention Injection (Gap 3)](#cross-attention-injection-gap-3)
   - [Training the Injectors](#training-the-injectors)
8. [CLI Reference](#cli-reference)
9. [Full Pipeline Walkthrough](#full-pipeline-walkthrough)
10. [Agent Pipeline Integration — OpenClaw Example](#agent-pipeline-integration)
11. [Architecture Diagram](#architecture-diagram)
12. [Disclaimer](#disclaimer)

---

## What is OpenSkill?

OpenSkill is a complete **skill lifecycle management system** for LLM agents. It treats *skills* — structured, reusable knowledge documents — as first-class citizens that can be:

- **Created** via dual-agent contrastive analysis (MemCollab)
- **Evolved** via parallel fleet distillation from execution traces (Trace2Skill)
- **Compressed** into compact geometric memory via near-optimal quantization (TurboQuant)
- **Retrieved** via semantic graph traversal with neural scoring (S-Path-RAG)
- **Injected** directly into an LLM's geometry without touching the prompt (Cross-Attention Injection)
- **Self-generated** and co-evolutionarily verified without human labels (EvoSkills)

The result is an agent that accumulates **transferable, model-agnostic procedural knowledge** that improves continuously — without any parameter updates to the underlying LLM.

```
Task → MemCollab → SKILL.md → TurboQuant vector → S-Path-RAG graph → Injection → Answer
         ↑                                                                          |
         └──────────────── Trace2Skill evolution ◄── execution traces ◄────────────┘
```

---

## Architecture Overview

```
openskill/
├── core/
│   ├── crafter.py      # MemCollab: dual-agent contrastive distillation
│   ├── evolver.py      # Trace2Skill: fleet-based parallel evolution
│   ├── vector.py       # TurboQuant: Lloyd-Max + QJL quantization
│   ├── graph.py        # S-Path-RAG: semantic shortest-path retrieval
│   ├── trainer.py      # Neural path scorer training (InfoNCE + BCE)
│   ├── gnn_encoder.py  # GNN residual encoder for graph enrichment
│   └── verifier.py     # EvoSkills: surrogate sandbox verifier
├── injection/
│   ├── soft.py         # SkillProjector: prefix token injection (Gap 2)
│   ├── cross_attention.py  # CrossAttentionInjector: K/V injection (Gap 3)
│   └── local_llm.py    # LocalSkillInjectedLLM: unified inference
├── retrieval/
│   └── retriever.py    # OpenSkillRetriever: orchestrates TurboQuant + graph
├── storage/
│   ├── local.py        # LocalDiskStore: file-based storage
│   └── cloud.py        # CloudSaaSStore: REST API storage
└── client.py           # OpenSkillClient: unified high-level API
```

---

## Research Foundations

### MemCollab

**Paper:** *MemCollab: Cross-Agent Memory Collaboration via Contrastive Trajectory Distillation* (arXiv:2603.23234)

**Core insight:** Memory distilled from a single agent's trajectories is *agent-specific* — it encodes that model's stylistic preferences and heuristics, not the underlying task structure. When you transfer this memory to a different model, performance degrades.

**Solution:** Run *two* agents (one weak, one strong) on the *same task*, then contrast their trajectories. What is **invariant** between the two is the transferable reasoning principle. What differs is agent-specific bias to be discarded.

**How OpenSkill implements it (`crafter.py`):**

1. `generate_trajectories(task, weak_model, strong_model)` — runs both agents in parallel via the LLM provider
2. `contrastive_analysis(task, preferred, unpreferred)` — extracts constraints in the form `"When X, enforce Y; avoid Z"` using a structured system prompt
3. `synthesize_skill(...)` — packages invariants, violations, and constraints into a structured JSON object
4. `render_markdown(...)` — produces the final `SKILL.md` with YAML frontmatter

The extracted constraints use **abstract reasoning forms** (e.g., conditioning, accumulation, case enumeration) that generalize across problem instances and even across LLM families.

```python
from openskill import OpenSkillClient, LocalDiskStore
from openskill.llm.openrouter import OpenRouterProvider

client = OpenSkillClient(
    store=LocalDiskStore("./skills_output"),
    llm=OpenRouterProvider(api_key="sk-or-..."),
)

# MemCollab: dual-agent contrastive distillation
skill = await client.craft(
    task="Implement a circuit breaker pattern for microservices",
    weak_model="openai/gpt-4o-mini",
    strong_model="anthropic/claude-3-5-sonnet",
)
print(skill.id, skill.title)
```

---

### Trace2Skill

**Paper:** *Trace2Skill: Distill Trajectory-Local Lessons into Transferable Agent Skills* (arXiv:2603.25158)

**Core insight:** Existing skill evolution methods update a skill *sequentially* as each new trajectory arrives. This is like an author who edits their manual while still learning — they react prematurely, before acquiring holistic domain understanding. Human experts instead **analyze broad experience in parallel** before writing.

**Solution:** Dispatch a *fleet* of parallel sub-agents, each analyzing a single trajectory independently. Then consolidate all their proposed patches simultaneously via **inductive hierarchical merging** — retaining only patterns that recur across diverse trajectories.

**How OpenSkill implements it (`evolver.py`):**

1. `generate_trajectories(skill_md, tasks)` — the agent runs the current skill against a set of test tasks, producing labeled trajectories (success/failure)
2. `_create_batches(trajectories)` — divides N trajectories among up to 10 sub-agent slots
3. `_analyze_batch(skill_md, batch)` — each sub-agent proposes `SkillPatch` objects (JSON diffs with section, op, content, prevalence)
4. `_hierarchical_merge(skill_md, groups)` — a merge operator deduplicates, resolves conflicts, and **elevates prevalent patterns** (those seen across many independent batches)
5. `_apply_patches(skill_md, patches)` — applies only patches above a minimum prevalence threshold (0.3 by default)

The **parallelism** is the key differentiator: all patches derive from the same frozen initial skill, preventing the sequential drift of online methods. The merge step performs inductive reasoning — systematic task properties that appear across many trajectories get elevated; idiosyncratic one-off observations get discarded.

```python
# Trace2Skill: evolve a skill from agent execution traces
result = await client.evolve(
    skill_id="a493a4b2",
    tasks=[
        "Handle leader election failure with network split",
        "Recover from follower crash with uncommitted entries",
        "Handle clock skew in election timeout",
    ]
)
print(f"Applied {result['patch_count']} patches, success rate: {result['success_rate']:.1%}")
```

---

### TurboQuant

**Paper:** *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate* (arXiv:2504.19874)

**Core insight:** Existing vector quantization methods either sacrifice distortion quality or require offline data-dependent tuning (k-means codebooks). TurboQuant achieves **near-optimal distortion** (within a factor of ~2.7 of the Shannon information-theoretic lower bound) in a fully online, data-oblivious manner.

**Two-stage algorithm:**

**Stage 1 — MSE-optimal quantization:**
- Apply a random rotation matrix Π to the input vector (making each coordinate follow a Beta distribution)
- In high dimensions, coordinates become approximately independent — so **optimal scalar quantization can be applied per-coordinate independently**
- Use Lloyd-Max 1D k-means to find the 2^b optimal centroids for the Beta distribution
- Store the centroid index (b bits per coordinate)

**Stage 2 — Unbiased inner product correction:**
- MSE-optimal quantizers introduce bias in inner product estimation
- Apply a 1-bit QJL (Quantized Johnson-Lindenstrauss) transform to the residual error
- This makes the combined estimator **unbiased** with near-optimal variance

**How OpenSkill implements it (`vector.py`):**

```python
# TurboQuantizer: compress a 1536d embedding to ~4 bits/channel
from openskill.core.vector import TurboQuantizer

q = TurboQuantizer(dimension=1536)
vec = await q.embed(llm, skill_markdown)  # get the embedding
qv = q.quantize(vec)                      # compress: Stage 1 (Lloyd-Max) + Stage 2 (QJL)
vec_approx = q.dequantize(qv)             # reconstruct for injection
similarity = q.similarity(qv_a, qv_b)    # inner product with QJL correction
```

The `QuantizedVector` object stores:
- `qvec` — int8 centroid indices (Stage 1)
- `residual` — int8 QJL signs (+1/-1) of the residual (Stage 2)
- `centroids` — the 16 Lloyd-Max optimal values for this bit-width
- `dim` — original dimension for dequantization

This is what makes the skill graph **memory-efficient**: a 1536-dimensional float32 embedding (6144 bytes) is compressed to ~768 bytes at 4 bits/channel while preserving geometric relationships for retrieval.

---

### S-Path-RAG

**Paper:** *S-Path-RAG: Semantic-Aware Shortest-Path Retrieval Augmented Generation for Multi-Hop Knowledge Graph QA* (arXiv:2603.23512)

**Core insight:** Standard RAG retrieves flat documents. For skills organized in a knowledge graph, the *path* between concepts matters — multi-hop reasoning requires traversing semantic relationships, not just ranking isolated nodes.

**Key innovations:**

**1. Semantically weighted path enumeration (Eq. 1–2):**
Each edge gets a weight combining structural cost, semantic similarity (cosine of node embeddings), and relation priors. Paths are scored by: `score(p; q) = -Σ w_e + λ_sem · sem(p, q)`

**2. Differentiable path scoring (Eq. 3–4):**
A PathScorerModel (MLP) computes `u_p = s_θ(p, q; Z)` for each candidate path. During training, Gumbel-Softmax relaxation provides differentiable discrete selection. The verifier head `v_η(p, q)` suppresses LLM-plausible but graph-unsupported paths.

**3. Soft latent injection (Eq. 5–6):**
Instead of verbalizing path lists into tokens, selected paths are combined as `z_ctx = Σ α_p · Enc_path(p)` and injected into the LLM via cross-attention. The language model attends to compact path representations, keeping token usage low.

**4. Neural-Socratic Graph Dialogue:**
An iterative loop where the LLM's uncertainty signal maps to graph expansions. If confidence < threshold after round t, the retriever expands seed nodes and re-enumerates paths.

**How OpenSkill implements it (`graph.py`, `trainer.py`, `retriever.py`):**

```python
# The graph is built automatically as skills are created
# Skills in the same category get SIMILAR_TO edges
# find_paths() runs the Neural-Socratic loop

guidance = await client.retrieve(
    "How to handle leader election failure in Raft?",
    top_k=3,
    use_graph=True
)
print(guidance.confidence)        # path score from neural scorer
print(guidance.skill_alphas)      # α_p per skill (Eq 5)
print(guidance.reasoning_trace)   # Socratic loop diagnostic
```

**Training the scorer** (separately, see [Training the Path Scorer](#training-the-path-scorer)):

The scorer is trained with:
- **InfoNCE loss** (contrastive: positive path vs all negatives in batch)
- **BCE verifier loss** (binary: does this path support the query?)
- **Gumbel-Softmax regularization** (entropy term to prevent collapse)

---

### EvoSkills

**Paper:** *EvoSkills: Self-Evolving Agent Skills via Co-Evolutionary Verification* (arXiv:2604.01687)

**Core insight:** Human-authored skills suffer from *human-machine cognitive misalignment* — workflows intuitive to humans don't match how LLMs actually reason. Letting agents create their own skills produces better-aligned procedural knowledge, but one-shot generation is unreliable.

**Solution:** A co-evolutionary loop between a **Skill Generator** and a **Surrogate Verifier** that are informationally isolated from each other:

1. The Skill Generator produces a multi-file skill bundle (SKILL.md + scripts/utils.py)
2. The Surrogate Verifier — a separate LLM session with no access to the generator's code — independently generates deterministic assert tests and evaluates the outputs
3. If the surrogate tests fail, the verifier provides structured diagnostics; the generator refines the skill
4. When the surrogate tests pass, a **Ground-Truth Oracle** re-executes in a fresh environment and returns only an opaque pass/fail signal
5. If the oracle fails, the verifier **escalates** its test suite (adds harder cases); the loop continues

The information isolation is critical: the verifier cannot inherit the generator's biases (e.g., a verifier that uses BLS for period detection cannot know the generator switched to TLS). This forces genuine coverage rather than confirmation bias.

**How OpenSkill implements it (`crafter.py`, `verifier.py`):**

```python
# co_evolve_skill_bundle runs the EvoSkills loop internally
skill_data, executable_code = await crafter.co_evolve_skill_bundle(
    task=task,
    constraints=constraints,
    strong_trajectory=strong_traj,
    verifier=SurrogateVerifier(llm=llm),
    max_iters=5
)
```

The resulting bundle has:
- `SKILL.md` — declarative procedural knowledge (what to do, when, and why)
- `scripts/utils.py` — tested, debugged utility functions ready for import

---

## Installation

OpenSkill is built with modern Python packaging standards. You can install it in editable mode to use the CLI directly.

### 1. Environment Setup
We recommend using [uv](https://github.com/astral-sh/uv) for lightning-fast dependency management:

```bash
# Clone the repository
git clone https://github.com/IhateCreatingUserNames2/OpenSkill.git
cd OpenSkill/OpenSkillLib

# Install core package
make install

# OR install with all optional dependencies (LLMs, GNNs, Server, LangChain)
make install-all
```

### 2. Configuration
Set your API key to enable distillation and retrieval features:

```bash
# Linux/macOS
export OPENROUTER_API_KEY="sk-or-v1-..."

# Windows (PowerShell)
$env:OPENROUTER_API_KEY="sk-or-v1-..."
```

### 3. Usage
Once installed, the `openskill` command is available globally in your terminal:

```bash
# Check version
openskill --version

# Run retrieval with local injection
openskill retrieve "How to design Circuit Breaker?" --local --mode auto

# List all skills
openskill list
```

---

## Core Concepts

### Skills vs Tools

| | Tool | Skill |
|---|---|---|
| **Structure** | Single function / API | Multi-file bundle (SKILL.md + scripts/) |
| **Scope** | One atomic action | Full workflow with decision points |
| **Knowledge** | Procedural (how) | Procedural + declarative (how + when + why) |
| **Failure handling** | None | Explicit error patterns and prevention rules |
| **Portability** | Model-specific | Model-agnostic (transfers across LLM families) |
| **Evolution** | Static | Iteratively improved via Trace2Skill |

### Skill Bundle Structure

```
skills_output/skills/<skill_id>/
├── SKILL.md          # Declarative knowledge: invariants, violations, constraints
├── meta.json         # Metadata: vectors, category, level, evolution count
└── scripts/
    └── utils.py      # Executable utility functions (EvoSkills output)
```

### Skill Types

| Type | Description | Hotbar slot |
|---|---|---|
| `PASSIVE` | Pure declarative knowledge, injected into LLM geometry | Passive (latent injection) |
| `ACTIVE` | Has executable scripts, loaded into prompt context | Active (token injection) |
| `HYBRID` | Both declarative rules AND executable code | Both slots |

---

## How to Use OpenSkill

### Creating New Skills

**Via Python API:**

```python
import asyncio
from openskill import OpenSkillClient, LocalDiskStore
from openskill.llm.openrouter import OpenRouterProvider

async def main():
    client = OpenSkillClient(
        store=LocalDiskStore("./skills_output"),
        llm=OpenRouterProvider(api_key="sk-or-..."),
    )

    # Creates skill via MemCollab (dual-agent contrastive distillation)
    # + EvoSkills (co-evolutionary verification of generated code)
    skill = await client.craft(
        task="Implement an exponential backoff retry strategy for HTTP requests",
        weak_model="openai/gpt-4o-mini",        # generates the "unpreferred" trajectory
        strong_model="anthropic/claude-3-5-sonnet",  # generates the "preferred" trajectory
    )
    print(f"Created skill: {skill.id} — {skill.title}")
    print(f"Type: {skill.skill_type.value}")

asyncio.run(main())
```

**Via CLI:**

```bash
openskill create "Implement exponential backoff retry for HTTP requests" \
    --api-key sk-or-... \
    --weak openai/gpt-4o-mini \
    --strong anthropic/claude-3-5-sonnet
```

**What happens internally:**
1. Both models generate full reasoning trajectories for the task
2. `contrastive_analysis()` extracts transferable constraints (MemCollab)
3. `co_evolve_skill_bundle()` generates and verifies executable code (EvoSkills)
4. The skill bundle is saved: `SKILL.md` + `scripts/utils.py` + `meta.json`
5. `TurboQuantizer.embed()` generates a compressed vector for the skill
6. The skill is registered in the knowledge graph with `SIMILAR_TO` edges to related skills

---

### Training the Path Scorer

The path scorer enables intelligent, semantic path selection in the knowledge graph. It must be trained before the neural retrieval mode (`confidence > 0`) activates.

**Step 1 — Embed all skills** (generates 1536d or 384d vectors):

```bash
# Remote (OpenAI 1536d — higher quality):
openskill embed <skill-id> --api-key sk-or-...

# Local (MiniLM 384d — free, no API):
openskill embed <skill-id> --local
```

**Step 2 — Train the scorer:**

```bash
# Auto-detects embedding dimension from your skills
python train_scorer.py --skill-dir ./skills_output --epochs 80

# With explicit dimension (if auto-detection fails):
python train_scorer.py --skill-dir ./skills_output --embed-dim 1536 --epochs 120
```

The trainer:
- Generates a synthetic dataset of (query, path, label) triples via `bootstrap_data.py`
- Uses domain-specific query templates + title-based queries as positives
- Creates hard negatives (same category), random negatives, and reversed negatives
- Trains with **InfoNCE** (contrastive ranking) + **BCE** (verifier) + **Gumbel-Softmax** regularization
- Saves to `skills_output/path_scorer.safetensors`

**Step 3 — Verify it works:**

```bash
python train_scorer.py --skill-dir ./skills_output --data-cache train_data.npz
# Should show: Verifier Accuracy: >60%, MRR: >0.5
```

**Full automated pipeline:**

```bash
python pil.py --api-key sk-or-... --skill-dir ./skills_output --epochs 80
# Runs: embed → retrain → test in sequence
```

---

### Distilling Skills from Trajectories

Once you have execution traces from an agent running in your environment, Trace2Skill can evolve an existing skill:

```python
# Provide raw execution traces
trajectories = [
    {
        "task": "Handle Raft leader election with network partition",
        "trajectory": "<full agent trace here>",
        "success": False  # agent failed — error analyst will diagnose
    },
    {
        "task": "Replicate log entries to quorum",
        "trajectory": "<full agent trace here>",
        "success": True   # agent succeeded — success analyst will extract patterns
    },
    # ... more trajectories
]

result = await client.evolve(
    skill_id="my-raft-skill",
    trajectories=trajectories
)
print(f"Fleet size: {result['fleet_size']}")
print(f"Patches applied: {result['patch_count']}")
print(f"Success rate: {result['success_rate']:.1%}")
```

Or let OpenSkill generate test trajectories automatically from a list of task descriptions:

```python
result = await client.evolve(
    skill_id="my-raft-skill",
    tasks=[
        "Handle split-brain scenario with 3-node cluster",
        "Recover from leader crash with uncommitted log entries",
        "Handle follower falling too far behind",
    ]
)
```

**What the fleet does:**
- Dispatches one sub-agent per trajectory in parallel (up to 10 concurrent)
- Success analysts do a single-pass extraction of generalizable behavior patterns
- Error analysts run an interactive agentic loop — they inspect files, compare against ground truth, and causally validate the root failure before proposing a patch
- The merge operator consolidates all patches, discarding those with prevalence < 0.3
- The evolved `SKILL.md` is saved back to disk

---

### Retrieving Skills at Inference

```python
# Basic retrieval
guidance = await client.retrieve(
    "How to prevent data loss when a Raft follower crashes?",
    top_k=3,
    use_graph=True    # use semantic graph + neural scorer
)

# guidance contains:
print(guidance.best_path_ids)    # skill IDs in ranked order
print(guidance.skill_contents)   # SKILL.md content of each skill
print(guidance.skill_vectors)    # dequantized TurboQuant vectors
print(guidance.skill_alphas)     # α_p weights per skill (Eq 5, S-Path-RAG)
print(guidance.confidence)       # neural path score (0–1)
print(guidance.reasoning_trace)  # Socratic loop diagnostic
```

If `confidence < 0.35`, the Neural-Socratic loop automatically expands the seed set and re-runs path enumeration (up to 3 rounds). This catches multi-hop evidence that a single-round retrieval would miss.

---

### Using SKILL.md Files

A `SKILL.md` is the declarative heart of a skill. It encodes *what the agent should do*, *when*, and *why*:

```markdown
---
name: Exponential Backoff Retry
domain: Programming
generated: 2026-04-08 10:30 UTC
method: MemCollab Contrastive Trajectory Distillation
---

# Exponential Backoff Retry

> Retry transient HTTP failures with exponentially increasing delays and jitter.

## When to Apply
When making HTTP requests that may fail transiently (5xx, 429, network timeouts).

## Reasoning Invariants
- Always compute delay as: base_delay * (2^attempt) + random_jitter
- Always cap maximum delay to prevent infinite waiting
- Always distinguish retryable (5xx, 429) from non-retryable (4xx) errors

## Violation Patterns
- ⚠️ Retrying immediately without backoff causes thundering herd
- ⚠️ Not adding jitter causes synchronized retry storms
- ⚠️ Retrying 4xx errors wastes resources — these are client errors

## Normative Constraints
- When retrying, enforce: delay = min(base * 2^n + jitter, max_delay); avoid fixed delays
- When classifying errors, enforce: only retry on 5xx/429/timeout; avoid retrying 4xx
```

**Loading a SKILL.md into an agent's context:**

```python
# Direct content retrieval
skill_md = await client.store.get_skill_md("my-skill-id")

# Or via guidance (retrieval result)
for content in guidance.skill_contents:
    # inject into system prompt, or use for vector injection
    pass
```

**For executable skills (HYBRID type):**

```python
import sys
sys.path.insert(0, './skills_output/skills/my-skill-id/scripts')
from utils import exponential_backoff_retry

result = exponential_backoff_retry(
    fn=my_http_call,
    max_retries=5,
    base_delay=1.0,
    max_delay=30.0
)
```

---

## Vector Injection

Vector injection is the mechanism by which skills influence LLM generation **geometrically** — without consuming token budget. Instead of putting skill content in the prompt (which increases cost, risks distraction, and can be ignored by the model), the skill's semantic vector is embedded directly into the LLM's hidden state.

This is the "soul" of the system: the agent's generation is *steered* toward the skill's reasoning domain at the representation level.

### Prefix Injection (Gap 2)

**Implemented in:** `injection/soft.py`, `core/trainer.py`

The `SkillProjector` maps skill vectors (384d or 1536d) into the LLM's hidden dimension and prepends them as *virtual prefix tokens* before the prompt tokens.

```
[skill_1_projected] [skill_2_projected] [skill_3_projected] <user> How to ... </user>
```

The LLM sees these as additional context at the start of the sequence. The `L_align` training objective (Eq. 7–8 from S-Path-RAG) ensures the model actually attends to them proportionally to their alpha_p weights.

**Training:**

```bash
python projector_trainer.py \
    --skill-dir ./skills_output \
    --model-id Qwen/Qwen3.5-2B \
    --data-cache train_data.npz \
    --epochs 30 \
    --lambda-align 0.5
```

**Using it:**

```bash
openskill retrieve "your query" --local --mode prefix
```

### Cross-Attention Injection (Gap 3)

**Implemented in:** `injection/cross_attention.py`, `cross_attn_trainer.py`

More powerful than prefix injection. `CrossAttentionInjector` installs **forward hooks** on the last 25% of the LLM's decoder layers. At each target layer, the hook computes:

```
Attn(Q_tok, K_graph, V_graph) = softmax(Q_tok @ K_graph^T / sqrt(d)) @ V_graph
```

Where:
- `Q_tok` — the layer's normal hidden states (token queries)
- `K_graph` — skill vectors projected to key space via `k_proj`
- `V_graph` — skill vectors projected to value space via `v_proj`

The output is gated by a learnable per-layer scalar: `hidden += tanh(gate) * cross_attn_output`

Gates are initialized to 0 (no effect) and grow during training. This means an untrained injector is perfectly safe — it has zero effect on generation until trained.

**What cross-attention injection actually does:**

It allows *every deep layer* of the LLM to "consult" the skills independently. Unlike prefix tokens (which can be forgotten as context accumulates), cross-attention operates at every layer simultaneously. The model's internal representations are continuously steered toward the skill domain throughout the forward pass.

**Training:**

```bash
python cross_attn_trainer.py \
    --skill-dir ./skills_output \
    --model-id Qwen/Qwen3.5-2B \
    --data-cache train_data.npz \
    --epochs 20 \
    --lambda-align 1.0
```

The trainer uses **Representation Engineering (RepE)**: instead of generating text and computing answer loss, it minimizes `MSE(R_injected[-1], R_persona[-1])` — making the model's final hidden state match the representation of an expert persona. This avoids repetition degeneration while teaching the model to "think like" the skill domain.

**Using it:**

```bash
openskill retrieve "your query" --local --mode cross_attention
```

**Auto mode** (recommended):

```bash
openskill retrieve "your query" --local --mode auto
# Selects: cross_attention (if trained) > prefix (if trained) > verbalization
```

### Training the Injectors

```python
# Full pipeline: embed → train scorer → train projector → train cross-attn
python pil.py --api-key sk-or-... --epochs 80           # embed + scorer
python projector_trainer.py --skill-dir ./skills_output  # prefix injector
python cross_attn_trainer.py --skill-dir ./skills_output # cross-attn injector
```

---

## CLI Reference

```bash
# Create a skill from a task description
openskill create "your task description" --api-key sk-or-...

# Retrieve skills for a query
openskill retrieve "your query" [--local] [--mode auto|verbalization|prefix|cross_attention]

# Evolve an existing skill with test tasks
openskill evolve <skill-id> --tasks "task1,task2,task3"

# Embed skills (generate TurboQuant vectors)
openskill embed <skill-id> --api-key sk-or-...    # remote 1536d
openskill embed <skill-id> --local                 # local 384d

# Rebuild the knowledge graph
openskill build-graph [--use-gnn]

# List all skills
openskill list

# Inspect graph topology
openskill graph

# Start MCP server (for Cursor/Windsurf IDE)
openskill serve --port 8000
```

---

## Full Pipeline Walkthrough

```bash
# 1. Create a library of skills in your domain
python bootstrap_library.py   # creates 10 skills via OpenRouter

# 2. Embed all skills into TurboQuant vectors
python pil.py --api-key sk-or-... --epochs 80

# 3. Train the path scorer (neural scoring)
python train_scorer.py --skill-dir ./skills_output --epochs 80

# 4. (Optional) Train the prefix injector
python projector_trainer.py --skill-dir ./skills_output --model-id Qwen/Qwen3.5-2B

# 5. (Optional) Train the cross-attention injector
python cross_attn_trainer.py --skill-dir ./skills_output --model-id Qwen/Qwen3.5-2B

# 6. Query the system
openskill retrieve "How to handle connection pool exhaustion in PostgreSQL?" \
    --local --mode auto

# 7. Evolve skills from execution experience
openskill evolve <skill-id> --tasks "connection pool under heavy load,pool timeout recovery"
```

---

## Agent Pipeline Integration

OpenSkill is designed to plug into any agent harness. Here is an example integration with an **OpenClaw-style** agentic pipeline — a multi-step reasoning agent that solves complex software engineering tasks.

### Architecture

```
User Task
    │
    ▼
┌─────────────────────────────────────────────────┐
│              OpenClaw Agent                      │
│                                                  │
│  1. Task Analysis                                │
│     └─► openskill.retrieve(task)                 │
│         → RetrievalGuidance                      │
│                                                  │
│  2. Hotbar Loading                               │
│     └─► SkillLoadout.equip(guidance.skills)      │
│         ACTIVE skills → prompt context           │
│         PASSIVE skills → latent vectors          │
│                                                  │
│  3. Reasoning Loop (ReAct)                       │
│     └─► LocalSkillInjectedLLM.generate_with_     │
│             guidance(query, guidance, mode=auto) │
│         ┌── mode=cross_attention                  │
│         │   → K_graph/V_graph injected at        │
│         │     each transformer layer             │
│         └── mode=verbalization (fallback)        │
│             → constraints in system prompt       │
│                                                  │
│  4. Post-Execution                               │
│     └─► collect traces                           │
│         openskill.evolve(skill_id, traces)       │
└─────────────────────────────────────────────────┘
```

### Code Example

```python
import asyncio
from openskill import OpenSkillClient, LocalDiskStore
from openskill.injection.local_llm import LocalSkillInjectedLLM

# ── Setup ──────────────────────────────────────────────────────────────────

store = LocalDiskStore("./skills_output")
llm = LocalSkillInjectedLLM(model_id="Qwen/Qwen3.5-2B", skill_dir="./skills_output")
client = OpenSkillClient(store=store, llm=llm)


# ── OpenClaw-style Agent ───────────────────────────────────────────────────

async def openclaw_agent(task: str) -> str:
    """
    Multi-step agent that uses OpenSkill to:
      1. Retrieve the most relevant skills for the task
      2. Load them into the Hotbar (active tokens + passive geometry)
      3. Generate with cross-attention injection
      4. Collect the trace for future Trace2Skill evolution
    """

    # STEP 1: Semantic skill retrieval (S-Path-RAG)
    guidance = await client.retrieve(task, top_k=3, use_graph=True)

    print(f"[Skills] Retrieved {len(guidance.best_path_ids)} skills")
    print(f"[Skills] Confidence: {guidance.confidence:.2f}")
    print(f"[Skills] Alphas: {[f'{a:.3f}' for a in guidance.skill_alphas]}")

    # STEP 2: Prepare the hotbar (classify skills into Active vs Passive)
    loadout = await client.prepare_quest_loadout(task)

    print(f"[Hotbar] Active slots: {[s['title'] for s in loadout.equipped_active]}")
    print(f"[Hotbar] Passive auras: {[s['title'] for s in loadout.equipped_passive]}")

    # STEP 3: Generate with guidance
    # auto mode: tries cross_attention → prefix → verbalization
    response = await llm.generate_with_guidance(
        query=task,
        guidance=guidance,
        mode="auto",
        max_new_tokens=1024,
    )

    print(f"[Mode] Generation mode: {response.raw.get('mode', 'unknown')}")

    # STEP 4: Collect trace for evolution (in a real system, capture success/failure)
    trace = {
        "task": task,
        "trajectory": response.content,
        "success": True   # determined by your evaluation harness
    }

    return response.content


# ── Batch evolution after agent runs ──────────────────────────────────────

async def collect_and_evolve(skill_id: str, traces: list[dict]):
    """
    After collecting execution traces, run Trace2Skill to improve the skill.
    Best practice: batch 50-200 traces before evolving for holistic analysis.
    """
    result = await client.evolve(
        skill_id=skill_id,
        trajectories=traces
    )
    print(f"[Evolution] Patches: {result['patch_count']}, "
          f"Success rate: {result['success_rate']:.1%}, "
          f"Fleet size: {result['fleet_size']}")


# ── Example run ────────────────────────────────────────────────────────────

async def main():
    # First: create skills for your domain
    skill = await client.craft(
        task="Implement a secure JWT authentication system with RS256 keys",
        weak_model="openai/gpt-4o-mini",
        strong_model="anthropic/claude-3-5-sonnet",
    )
    print(f"Created: {skill.id} — {skill.title}")

    # Then: use the agent with skill guidance
    answer = await openclaw_agent(
        "I need to implement RS256 JWT authentication for my FastAPI app. "
        "How do I generate the key pair and verify tokens securely?"
    )
    print("\nAnswer:")
    print(answer)

    # Later: evolve from experience
    example_traces = [
        {"task": "JWT with short expiry", "trajectory": "...", "success": True},
        {"task": "Refresh token rotation", "trajectory": "...", "success": False},
    ]
    await collect_and_evolve(skill.id, example_traces)


asyncio.run(main())
```

### Integration with MCP (Cursor / Windsurf)

Add to your IDE's MCP configuration:

```json
{
  "mcpServers": {
    "openskill": {
      "command": "python",
      "args": ["-m", "openskill.mcp.server"],
      "env": {
        "OPENROUTER_API_KEY": "sk-or-..."
      }
    }
  }
}
```

Then in Cursor, you can ask:
> *"Use OpenSkill to help me implement a circuit breaker for this service"*

The MCP server exposes four tools: `openskill_craft`, `openskill_retrieve`, `openskill_evolve`, and `openskill_list`.

### Integration with LangChain

```python
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from openskill.retrieval.langchain import OpenSkillRetriever

retriever = OpenSkillRetriever(
    skill_dir="./skills_output",
    top_k=3,
    format="constraints",   # returns only normative constraints
    use_ollama=True,
    ollama_model="qwen2.5-coder:7b",
)

qa = RetrievalQA.from_chain_type(
    llm=Ollama(model="qwen2.5-coder:7b"),
    retriever=retriever,
)

result = qa.invoke("How to implement Raft consensus over a high-latency network?")
print(result["result"])
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         OPENSKILL SYSTEM                                │
│                                                                         │
│  ┌────────────────┐    ┌────────────────┐    ┌────────────────────────┐ │
│  │   MemCollab    │    │  Trace2Skill   │    │       EvoSkills        │ │
│  │  (Creation)    │    │  (Evolution)   │    │  (Self-Generation)     │ │
│  │                │    │                │    │                        │ │
│  │ weak model ──► │    │ fleet of       │    │ Skill Generator        │ │
│  │ strong model   │    │ sub-agents     │    │       ⇅ co-evolve      │ │
│  │   └► contrast  │    │   └► patches   │    │ Surrogate Verifier     │ │
│  │      └► SKILL  │    │      └► merge  │    │       ⇅ oracle         │ │
│  └───────┬────────┘    └───────┬────────┘    └──────────┬─────────────┘ │
│          │                    │                         │               │
│          └──────────────────► SKILL.md ◄────────────────┘               │
│                               + scripts/utils.py                        │
│                                    │                                    │
│                             ┌──────▼──────┐                             │
│                             │ TurboQuant  │                             │
│                             │  (Lloyd-Max │                             │
│                             │   + QJL)    │                             │
│                             └──────┬──────┘                             │
│                                    │ compressed vector                  │
│                             ┌──────▼──────────┐                         │
│                             │   S-Path-RAG    │                         │
│                             │  Knowledge Graph│                         │
│                             │  + PathScorer   │                         │
│                             │  + GNN encoder  │                         │
│                             └──────┬──────────┘                         │
│                                    │ guidance (path_ids, alphas)        │
│                    ┌───────────────┼────────────────┐                   │
│                    │               │                │                   │
│             ┌──────▼───┐  ┌────────▼──────┐  ┌─────▼────────┐         │
│             │Verbalize  │  │SkillProjector │  │CrossAttention│         │
│             │(tokens)   │  │(prefix tokens)│  │Injector(K/V) │         │
│             └──────┬───┘  └────────┬──────┘  └─────┬────────┘         │
│                    │               │                │                   │
│                    └───────────────▼────────────────┘                   │
│                                 LLM                                     │
│                              (frozen)                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Design Principles

**No parameter updates required.** Skills improve the agent without fine-tuning the LLM. The underlying model stays frozen; only the skill content and injector weights change.

**Model-agnostic skills.** Skills distilled by a 35B model improve a 122B model. Skills evolved on Qwen transfer to GPT and Claude. The declarative knowledge encoded in SKILL.md is architecture-independent.

**Holistic over sequential.** Trace2Skill analyzes all trajectories in parallel rather than reacting to each one. This prevents overfitting to early or recent observations and produces skills that generalize to out-of-distribution tasks.

**Confidence-gated retrieval.** The Neural-Socratic loop only expands search when confidence is below threshold, preventing unnecessary computation when the first retrieval round is already sufficient.

**Graceful degradation.** Every injection mode has a fallback chain: cross-attention → prefix → verbalization → plain generation. The system works even with no trained injectors — it just uses the constraint text from SKILL.md instead.

## Disclaimer 


**All of this is EXPERIMENTAL, this is Prototype, a Proof of Concept.** 

**This code is based on the concepts described in academic papers, it may contains issues in replicating and integrating them.**

**Vector Injection is a Trick Thing, to do as S-pAth RAG did, one would need a LoRA to understand what it is being Injected, for NOW, i have DECIDED to only Steer it In the Optimal Direction, instead of Context Injection in the Attention matrix, but the code can perform that with a few twerks.** 

**This code contains bugs and errors , just with caution.** 

**The End Result i Expect to See in this Project is that it can Create a package, a skill  containing logical abstractions, scripts, markdowns, vectors, anything, everything in a package that can be used by any Agent, or LLM, local or remote, thru Context Alone, or Vector Injection. Not only that, but also Update that Skill, Evolve it, make it learn.**





