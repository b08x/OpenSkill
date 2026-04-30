"""
OpenSkill Core — Pure Business Logic, Zero I/O
==============================================

Cada módulo contém apenas lógica de domínio, SEM dependência de:
  - Sistema de arquivos
  - APIs HTTP externas
  - Armazenamento

A injeção de dependência (storage, LLM provider) é feita pelo caller
(OpenSkillClient ou testes).

Módulos:
  crafter.py   — MemCollab: dual-agent contrastive distillation
  evolver.py   — Trace2Skill: fleet-based skill evolution
  vector.py    — TurboQuant: geometric vector quantization
  graph.py     — S-Path-RAG: semantic graph retrieval
"""

# ── Lazy Core Imports ────────────────────────────────────────────────────────

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
    
    raise AttributeError(f"module {__name__} has no attribute {name}")

__all__ = [
    "SkillCrafter",
    "SkillEvolver",
    "TurboQuantizer",
    "SkillGraph",
]
