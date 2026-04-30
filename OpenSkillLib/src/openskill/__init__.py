"""
OpenSkill — Geometric Skill Memory for LLM Agents
=================================================

Um sistema de ciclo de vida completo de skills para agentes LLM:

  MemCollab  (arXiv:2603.23234)
      Dual-agent contrastive trajectory distillation.
      Cria skills agent-agnostic a partir de pares fraco/forte.

  Trace2Skill (arXiv:2603.25158)
      Fleet-based parallel skill evolution.
      Evolução via frota de sub-agentes + consolidação hierárquica.

  TurboQuant (arXiv:2504.19874)
      Near-optimal vector quantization (4 bits/channel).
      Memória geométrica compressa com correção QJL.

  S-Path-RAG (arXiv:2603.23512)
      Semantic-aware shortest-path retrieval over skill graph.
      Neural-Socratic loop + soft latent injection.

Exemplo rápido:
    from openskill import OpenSkillClient, LocalDiskStore

    client = OpenSkillClient(store=LocalDiskStore("./skills"))
    skill = await client.craft(
        task="Implement Raft consensus over high-latency network",
        weak_model="openai/gpt-4o-mini",
        strong_model="anthropic/claude-3-5-sonnet",
    )
    guidance = await client.retrieve("How to handle network partitions in Raft?")
"""

# ── Lazy Imports ─────────────────────────────────────────────────────────────

def __getattr__(name: str):
    if name == "SkillCrafter":
        from openskill.core.crafter import SkillCrafter
        return SkillCrafter
    if name == "SkillEvolver":
        from openskill.core.evolver import SkillEvolver
        return SkillEvolver
    if name == "TurboQuantizer":
        from openskill.core.vector import TurboQuantizer
        return TurboQuantizer
    if name == "SkillGraph":
        from openskill.core.graph import SkillGraph
        return SkillGraph
    if name == "BaseSkillStore":
        from openskill.storage.base import BaseSkillStore
        return BaseSkillStore
    if name == "LocalDiskStore":
        from openskill.storage.local import LocalDiskStore
        return LocalDiskStore
    if name == "CloudSaaSStore":
        from openskill.storage.cloud import CloudSaaSStore
        return CloudSaaSStore
    if name == "BaseLLMProvider":
        from openskill.llm.base import BaseLLMProvider
        return BaseLLMProvider
    if name == "OpenRouterProvider":
        from openskill.llm.openrouter import OpenRouterProvider
        return OpenRouterProvider
    if name == "OllamaProvider":
        from openskill.llm.ollama import OllamaProvider
        return OllamaProvider
    if name == "OpenSkillRetriever":
        from openskill.retrieval.retriever import OpenSkillRetriever
        return OpenSkillRetriever
    if name == "SkillProjector":
        from openskill.injection.soft import SkillProjector
        return SkillProjector
    if name == "OpenSkillClient":
        from openskill.client import OpenSkillClient
        return OpenSkillClient
    
    raise AttributeError(f"module {__name__} has no attribute {name}")

__all__ = [
    # Core
    "SkillCrafter",
    "SkillEvolver",
    "TurboQuantizer",
    "SkillGraph",
    # Storage
    "BaseSkillStore",
    "LocalDiskStore",
    "CloudSaaSStore",
    # LLM providers
    "BaseLLMProvider",
    "OpenRouterProvider",
    "OllamaProvider",
    # Retrieval
    "OpenSkillRetriever",
    # Injection
    "SkillProjector",
    # High-level
    "OpenSkillClient",
    # Version
    "__version__",
]

__version__ = "0.1.0"
