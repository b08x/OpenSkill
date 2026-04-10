# OpenSkill / SkillCrafter 🔬

**Universal Skill Distillation via Contrastive Trajectory Analysis & Geometric Memory**

## Project Overview
OpenSkill is an advanced framework for extracting, refining, and storing "Universal Skills" from LLM reasoning trajectories. It aims to create model-agnostic knowledge representations that can be shared across diverse LLM architectures without "style contamination."

### Key Methodologies
- **MemCollab (Contrastive Distillation):** Contrasts trajectories from a **Weak Agent** (e.g., Llama 3.1 8B) and a **Strong Agent** (e.g., GPT-4o, Claude 3.5) to extract **Reasoning Invariants** (logical must-haves) and **Violation Patterns** (failure modes to avoid).
- **Trace2Skill (Skill Evolution):** Iteratively evolves and patches skills based on empirical evidence from a "fleet" of execution traces.
- **TurbOQuant (Knowledge Quantization):** Compresses high-dimensional skill embeddings into **4-bit** representations with residuals, enabling efficient storage and near-optimal similarity search.
- **S-PATH RAG (Path-Aware Retrieval):** A graph-based retrieval system that navigates a **Skill Graph** using semantic-aware shortest-path algorithms to find the most relevant "logical handbook" for a task.

---

## Core Architecture
- **`OpenSkillLib/`**: The main library containing core logic.
  - `openskill.core.crafter`: Orchestrates the MemCollab distillation pipeline.
  - `openskill.core.evolver`: Implements the Trace2Skill evolution and patching logic.
  - `openskill.core.vector`: Implements the **TurboQuant** quantization engine.
  - `openskill.core.graph`: Implements the **S-PATH RAG** skill graph retrieval.
  - `openskill.llm`: Providers for various LLM backends (OpenRouter, Ollama, etc.).
- **`OpenSkill1.1/`**: A Proof-of-Concept implementation demonstrating the full evolution and quantization cycle.
- **`main.py` (root)**: A FastAPI-based web application providing a UI for the SkillCrafter pipeline.
- **`mcp_openSkill.py`**: Implementation of a Model Context Protocol (MCP) server for OpenSkill.
- **`SkillTesting/`**: Benchmark tasks and test trajectories used for verification and distillation.

---

## Building and Running

### Prerequisites
- Python 3.11+
- `OPENROUTER_API_KEY` environment variable (for the web UI and distillation).

### Setup
```bash
# Install dependencies
pip install -r requirements.txt
# For the core library (development mode)
pip install -e OpenSkillLib/src
```

### Running the Web Interface
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Access the UI at: `http://localhost:8000`

### Running the PoC (OpenSkill1.1)
```bash
python OpenSkill1.1/main.py
```

### Running the MCP Server
```bash
python mcp_openSkill.py
```

---

## Development Conventions
- **Skill Format:** Skills are stored as Markdown (`SKILL.md`) with YAML frontmatter, containing `Reasoning Invariants`, `Violation Patterns`, and `Normative Constraints` (format: `When [condition], enforce [principle]; avoid [failure]`).
- **Quantization:** Embeddings are 1536-dim (standard) or 384-dim (GNN-refined), quantized using the TurboQuantizer to 4-bit with residuals.
- **Logging:** Uses `structlog` for structured, machine-readable logging.
- **Versioning:** Skills track their evolution through `evolution_count` and timestamps in the `skill_graph.json`.
- **Testing:** Verification is performed via `SurrogateVerifier` in a sandboxed environment, often generating and running Python scripts to validate logic.
