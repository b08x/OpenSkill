# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenSkill is an advanced AI/ML framework for extracting, refining, and storing "Universal Skills" from LLM reasoning trajectories. It implements four research papers into a production skill lifecycle system:

- **MemCollab** (arXiv:2603.23234): Contrastive trajectory distillation between weak/strong agents
- **Trace2Skill** (arXiv:2603.25158): Parallel fleet evolution of skills  
- **TurboQuant** (arXiv:2504.19874): Geometric skill memory with 4-bit vector quantization
- **S-Path-RAG** (arXiv:2603.23512): Semantic skill graph retrieval with neural-socratic loops

## Architecture

The project has two main components:

### 1. OpenSkill1.1/ - Standalone Web Application
- **Entry point**: `main.py` - FastAPI application with integrated pipeline
- **API endpoints**: `/api/craft`, `/api/evolve`, `/api/retrieve`, `/api/graph`
- **Dependencies**: Basic FastAPI stack (see `OpenSkill1.1/requirements.txt`)

### 2. OpenSkillLib/src/ - Python Package Library
- **Entry point**: `openskill/` package with modular architecture
- **Core modules**: 
  - `openskill.core.crafter` - MemCollab distillation pipeline
  - `openskill.core.evolver` - Trace2Skill evolution logic
  - `openskill.core.vector` - TurboQuant quantization engine  
  - `openskill.core.graph` - S-Path-RAG skill graph retrieval
- **CLI**: Available via `openskill` command after installation
- **MCP Server**: `openskill-mcp` command for Model Context Protocol integration

## Common Development Commands

### Environment Setup
```bash
# Install core web application dependencies
pip install -r requirements.txt

# Install the library package (development mode)  
pip install -e OpenSkillLib/src

# Install with optional ML dependencies
pip install -e OpenSkillLib/src[hf,server,all]
```

### Running Applications
```bash
# Web UI (main application)
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# PoC implementation
python OpenSkill1.1/main.py

# MCP Server
python mcp_openSkill.py
# or after library install:
openskill-mcp

# Library server (alternative FastAPI implementation)
python -m openskill.server
```

### Docker Development
```bash
# Build and run web application
docker build -t openskill --target runtime-app .
docker run -p 8000:8000 openskill

# Build and run PoC with local inference
docker build -t openskill-poc --target runtime-poc .
docker run -p 8002:8002 openskill-poc
```

### Testing and Development
```bash
# Library development (from OpenSkillLib/src/)
make install  # Uses uv for fast installation
make test     # Run pytest suite
make clean    # Clean build artifacts

# Code quality
ruff check .     # Linting
mypy .          # Type checking
```

## Use Codemap CLI for Codebase Navigation

Codemap CLI is available for intelligent codebase visualization and navigation.

**Required Usage** - You MUST use `codemap --diff` to research changes different from default branch, and `git diff` + `git status` to research current working state.

### Quick Start

```bash
codemap .                    # Project tree
codemap --only py .          # Just Python files
codemap --exclude .git,__pycache__,.pytest_cache .  # Hide build artifacts
codemap --depth 2 .          # Limit depth
codemap --diff               # What changed vs main
codemap --deps .             # Dependency flow
```

### Options

| Flag | Description |
|------|-------------|
| `--depth, -d <n>` | Limit tree depth (0 = unlimited) |
| `--only <exts>` | Only show files with these extensions |
| `--exclude <patterns>` | Exclude files matching patterns |
| `--diff` | Show files changed vs main branch |
| `--ref <branch>` | Branch to compare against (with --diff) |
| `--deps` | Dependency flow mode |
| `--importers <file>` | Check who imports a file |
| `--skyline` | City skyline visualization |
| `--json` | Output JSON |

**Smart pattern matching** - no quotes needed:
- `.py` - any `.py` file
- `__pycache__` - any `/__pycache__/` directory
- `*test*` - glob pattern

### Diff Mode

See what you're working on:

```bash
codemap --diff
codemap --diff --ref development
```

### Use Context7 MCP for Loading Documentation

Context7 MCP is available to fetch up-to-date documentation with code examples.

**Recommended library IDs**:

- `/websites/hermes-agent_nousresearch` - Hermes Agent autonomous AI agent framework with built-in tools, browser capabilities, and MCP support (High reputation, Benchmark 89)
- `/nousresearch/hermes-agent` - Official Hermes Agent GitHub documentation with installation guides and CLI commands (High reputation)
- `/websites/fastapi_tiangolo` - FastAPI modern Python web framework for building high-performance APIs with automatic documentation (High reputation, 6652 snippets, Benchmark 89)
- `/websites/pydantic_dev` - Pydantic data validation library using Python type hints for robust data schemas (High reputation, 2547 snippets, Benchmark 86)
- `/huggingface/transformers` - Official Hugging Face Transformers library for state-of-the-art ML models across PyTorch, TensorFlow, and JAX (High reputation, 4634 snippets)
- `/llmstxt/huggingface_co_transformers_v5_2_0_llms_txt` - Comprehensive Transformers documentation with extensive code examples for NLP, CV, and multimodal tasks (High reputation, 22875 snippets)

## Key Configuration

### Environment Variables
- `OPENROUTER_API_KEY` - Required for LLM interactions via OpenRouter API
- `SKILLS_DIR` - Directory for skill storage (defaults to `skills_output/`)

### Model Configuration
- **Default Weak Model**: `openai/gpt-oss-120b` 
- **Default Strong Model**: `minimax/minimax-m2.7`
- **Configurable**: Both models can be overridden via API requests or environment

### Optional Dependencies
The library supports multiple deployment scenarios:
- `[hf]` - HuggingFace transformers for local inference
- `[server]` - FastAPI web server components
- `[redis]` - Redis-based skill storage
- `[pgvector]` - PostgreSQL vector storage
- `[mcp]` - Model Context Protocol integration
- `[all]` - All optional dependencies

## File Structure and Conventions

### Skill Format
Skills are stored as Markdown files with YAML frontmatter:
```markdown
---
name: Skill Title
domain: Mathematics/Coding/Reasoning
generated: 2026-03-25
method: MemCollab Contrastive Trajectory Distillation
---

## Reasoning Invariants
- Essential principles that must be preserved...

## Violation Patterns  
- ⚠️ Forbidden patterns that cause failure...

## Normative Constraints
- When [condition], enforce [principle]; avoid [failure]
```

### Vector Storage
- Embeddings: 1536-dim (standard) or 384-dim (GNN-refined)
- Quantization: 4-bit compression with residuals via TurboQuantizer
- Graph: Skills stored in typed graph with relationships (PREREQUISITE_OF, EXTENDS, etc.)

### Development Patterns
- **Logging**: Uses `structlog` for structured, machine-readable logs
- **Async**: Heavy use of asyncio for LLM API calls and parallel processing
- **Type Safety**: Pydantic models for API contracts and data validation
- **Modularity**: Clear separation between core algorithms, storage, and LLM providers

## Integration Points

### LLM Providers
- Primary: OpenRouter API (supports 200+ models)
- Local: Ollama integration for on-premise deployment
- Future: Direct HuggingFace transformers support

### Storage Backends
- Local: File-based storage for development
- Cloud: SaaS storage backend for production
- Vector: Redis or pgvector for high-performance retrieval

### External Integrations
- **LangChain**: Compatible skill injection
- **LlamaIndex**: Direct skill retrieval integration
- **MCP**: Model Context Protocol server for IDE/tool integration

## Troubleshooting

### Tensor Shape Mismatch in Skill Retrieval

**Symptom:** `/api/retrieve` endpoint returns 500 error with:
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x256 and 384x1024)
```

**Root Cause:** Dimensional mismatch between stored skill vectors and the embedding model's expected dimensions.

**Technical Details:**
The error occurs when skill vectors were created with a different embedding dimension than what the current embedding model produces:
- Stored skill vectors: 256D (from hardcoded `EMBED_DIM` in `skill_vector.py`)
- Current embedding model: 384D (from `all-MiniLM-L6-v2` SentenceTransformer)
- SkillProjector expects input matching current model dimensions

**Solution:** Dynamic Projector Pattern (Implemented in local_llm.py)
```python
# Auto-detect actual skill vector dimensions
actual_embed_dim = skills_tensor.shape[-1]

# Create projector matching actual input dimensions
if projector is None or projector.proj.in_features != actual_embed_dim:
    projector = SkillProjector(actual_embed_dim, LLM_HIDDEN_SIZE).to(device)
```

**Benefits:**
- ✅ Backward compatible with existing 256D quantized skills
- ✅ Forward compatible when embedding models change
- ✅ No need to regenerate existing skill vectors
- ✅ Eliminates need for synchronized `EMBED_DIM` constants across modules

**Prevention:** 
- Use single source of truth for embedding dimensions
- Consider this pattern when integrating multiple ML models with different tensor shapes
- Add validation to catch dimension mismatches early in development

### Common Issues and Solutions

**Issue:** HuggingFace models unavailable
- **Solution:** System automatically falls back to OpenRouter API
- **Verification:** Check console for "✗ Failed to load HF LLM" message

**Issue:** OpenRouter API key not configured
- **Solution:** Set `OPENROUTER_API_KEY` environment variable
- **Verification:** Check for "OPENROUTER_API_KEY not set" error

**Issue:** Skill quantization/dequantization errors
- **Solution:** Verify skill vector dimensions match current embedding model
- **Check:** Compare `skill_meta.qvector.dim` with current `EMBED_DIM`