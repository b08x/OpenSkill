# Graph Report - .  (2026-04-23)

## Corpus Check
- 65 files · ~154,572 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 667 nodes · 1612 edges · 56 communities detected
- Extraction: 49% EXTRACTED · 51% INFERRED · 0% AMBIGUOUS · INFERRED: 829 edges (avg confidence: 0.6)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Base Abstractions|Base Abstractions]]
- [[_COMMUNITY_App & Bootstrap Layer|App & Bootstrap Layer]]
- [[_COMMUNITY_Client & Core API|Client & Core API]]
- [[_COMMUNITY_Cross-Attention Injection|Cross-Attention Injection]]
- [[_COMMUNITY_Skill System|Skill System]]
- [[_COMMUNITY_Craft & Evolve APIs|Craft & Evolve APIs]]
- [[_COMMUNITY_Training Pipeline|Training Pipeline]]
- [[_COMMUNITY_ABC Interfaces|ABC Interfaces]]
- [[_COMMUNITY_Skill Legacy|Skill Legacy]]
- [[_COMMUNITY_Storage & Cloud|Storage & Cloud]]
- [[_COMMUNITY_LLM Providers|LLM Providers]]
- [[_COMMUNITY_Retrieval System|Retrieval System]]
- [[_COMMUNITY_MCP Server|MCP Server]]
- [[_COMMUNITY_Vector Operations|Vector Operations]]
- [[_COMMUNITY_Server Routes|Server Routes]]
- [[_COMMUNITY_Graph Structure|Graph Structure]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]

## God Nodes (most connected - your core abstractions)
1. `LLMMessage` - 90 edges
2. `BaseLLMProvider` - 81 edges
3. `OpenRouterProvider` - 67 edges
4. `LocalDiskStore` - 60 edges
5. `SurrogateVerifier` - 40 edges
6. `TurboQuantizer` - 39 edges
7. `SkillGraphData` - 35 edges
8. `OpenSkillClient` - 34 edges
9. `SkillGraph` - 34 edges
10. `LocalSkillInjectedLLM` - 34 edges

## Surprising Connections (you probably didn't know these)
- `generate_evolution_trajectories()` --calls--> `list()`  [INFERRED]
  OpenSkill1.1/skill_evolution.py → OpenSkillLib/src/openskill/cli/main.py
- `evolve_skill()` --calls--> `list()`  [INFERRED]
  OpenSkill1.1/skill_evolution.py → OpenSkillLib/src/openskill/cli/main.py
- `neural_socratic_retrieve()` --calls--> `list()`  [INFERRED]
  OpenSkill1.1/skill_graph.py → OpenSkillLib/src/openskill/cli/main.py
- `recompute_embedding()` --calls--> `register_skill_in_graph()`  [INFERRED]
  OpenSkill1.1/main.py → OpenSkillLib/src/openskill/core/graph.py
- `Removes reasoning tags (<think>...</think> or <thought>...</thought>)     leavin` --rationale_for--> `strip_reasoning_tags()`  [EXTRACTED]
  main.py → OpenSkill1.1/main.py

## Communities

### Community 0 - "Base Abstractions"
Cohesion: 0.05
Nodes (89): BaseLLMProvider, BaseSkillStore, LLMMessage, Skill graph (nodes + edges)., Abstract interface for skill storage.      Implement this interface to create, Interface that EVERY LLM provider must implement., Resource cleanup (optional)., SkillGraphData (+81 more)

### Community 1 - "App & Bootstrap Layer"
Cohesion: 0.04
Nodes (71): create_app(), OpenSkill FastAPI Server ======================== Ponto de entrada para a API, LLMResponse, BaseLLMProvider, BaseRetriever, _detect_embed_dim(), _embed_queries_async(), generate_bootstrap_dataset() (+63 more)

### Community 2 - "Client & Core API"
Cohesion: 0.05
Nodes (39): SkillVectorProfile, crafter(), evolver(), graph(), quantizer(), retriever(), craft_skill(), encode_graph_embeddings() (+31 more)

### Community 3 - "Cross-Attention Injection"
Cohesion: 0.05
Nodes (48): load_dataset(), Loads dataset saved in .npz., CrossAttentionInjector, detect_qwen_params(), load(), cross_attention.py — S-Path-RAG Gap 3: Cross-Attention Injection (Eq 6) =======, Precomputes K_graph and V_graph from skill vectors.         Must be called befo, Calculates Attn(Q_tok, K_graph, V_graph) for a layer.          Eq 6: softmax(Q (+40 more)

### Community 4 - "Skill System"
Cohesion: 0.06
Nodes (52): add_manual_edge(), get_graph(), Inspect the S-Path-RAG skill graph structure., Manually add a typed edge to the skill graph., add_edge(), add_node(), enumerate_paths(), get_adjacency() (+44 more)

### Community 5 - "Craft & Evolve APIs"
Cohesion: 0.09
Nodes (39): BaseModel, CraftRequest, evolve_skill(), EvolveRequest, call_llm(), classify_task(), contrastive_analysis(), craft_skill() (+31 more)

### Community 6 - "Training Pipeline"
Cohesion: 0.11
Nodes (20): _detect_dim_from_skills(), main(), train_scorer.py — Complete Path Scorer Training Script ========================, Detects embedding dimension by inspecting skill meta.json files.     Returns th, Full pipeline: data → train → evaluation., run_training(), _anneal_temperature(), evaluate_scorer() (+12 more)

### Community 7 - "ABC Interfaces"
Cohesion: 0.1
Nodes (4): ABC, dataclass_asdict_filter_none(), Storage Abstraction Layer — Adapter Pattern ===================================, Enum

### Community 8 - "Skill Legacy"
Cohesion: 0.18
Nodes (17): apply_patches_to_skill(), _call(), evolve_skill(), _extract_json(), generate_evolution_trajectories(), _hierarchical_consolidate(), _merge_patch_groups(), _propose_patches_for_batch() (+9 more)

### Community 9 - "Storage & Cloud"
Cohesion: 0.2
Nodes (3): from_dict(), BaseSkillStore, CloudSaaSStore

### Community 10 - "LLM Providers"
Cohesion: 0.12
Nodes (16): Error Analyst, Hierarchical Merging, MemCollab, ReAct Agent, Reasoning Invariants, S-PATH RAG, Skill Creation, Skill Deepening (+8 more)

### Community 11 - "Retrieval System"
Cohesion: 0.16
Nodes (10): generate_text(), generate_with_soft_latents(), get_embedding(), _openrouter_generate(), local_llm.py — LocalSkillInjectedLLM (Gap 3: cross-attention injection) =======, Default text generation (used for Trace2Skill and MemCollab)., S-Path-RAG Soft Latent Injection:     Attaches de-quantized quantized vectors DI, Generates a real embedding using SentenceTransformer. (+2 more)

### Community 12 - "MCP Server"
Cohesion: 0.22
Nodes (9): Cross-Attention Injection, Neural-Socratic Graph Dialogue, Path Scorer, Semantic-Aware Retrieval, api retrieve endpoint, KV Cache Quantization, MSE-Optimized Quantizer, QJL Transform (+1 more)

### Community 13 - "Vector Operations"
Cohesion: 0.4
Nodes (4): get_skill_details(), Search skills by query or category. Use to find the best skill for your problem., Returns the Skill.md content by ID for the agent to read the rules., search_skills()

### Community 14 - "Server Routes"
Cohesion: 0.67
Nodes (3): SkillsBench, Skill Generator, Surrogate Verifier

### Community 15 - "Graph Structure"
Cohesion: 0.67
Nodes (3): Contrastive Trajectory Distillation, Reasoning Invariants, Task-Aware Retrieval

### Community 16 - "Community 16"
Cohesion: 1.0
Nodes (2): api skills endpoint, loadSkillList function

### Community 17 - "Community 17"
Cohesion: 1.0
Nodes (2): api craft endpoint, craftSkill function

### Community 18 - "Community 18"
Cohesion: 1.0
Nodes (2): TurboQuant ref, TurboQuant Paper

### Community 19 - "Community 19"
Cohesion: 1.0
Nodes (2): MemCollab ref, MemCollab Paper

### Community 20 - "Community 20"
Cohesion: 1.0
Nodes (2): Strong Agent, Weak Agent

### Community 21 - "Community 21"
Cohesion: 1.0
Nodes (2): S-PATH-RAG ref, S-PATH-RAG Paper

### Community 22 - "Community 22"
Cohesion: 1.0
Nodes (0): 

### Community 23 - "Community 23"
Cohesion: 1.0
Nodes (0): 

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (0): 

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (1): True if gates diverged from zero.

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (1): Carrega projector treinado do disco.

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (1): Heurística: se os pesos divergiram de eye_, o projector foi treinado.

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (0): 

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (0): 

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (1): Generates a text response.          Args:             messages: Message list

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (1): Generates semantic embedding (for vector search).

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (0): 

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (0): 

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (0): 

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (0): 

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (1): Saves a skill (markdown + metadata).

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (1): Returns the Markdown content of a skill.

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (1): Returns the metadata of a skill.

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (1): Returns the complete skill bundle including all folders (reference, template, as

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (1): Returns the local workspace path, if applicable.         Returns None for pure

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): Lists all skills in the store.

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (1): Removes a skill from the store.

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (1): Returns the skill graph.

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (1): Updates the skill graph.

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (1): Saves the quantized vector (TurboQuant) of a skill.

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): Factory that returns the correct adapter based on URI.          Examples:

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (0): 

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (0): 

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): FastAPI

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): Uvicorn

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): Pydantic

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): MCP

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (1): SkillCrafter UI

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (1): EvoSkills Paper

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (1): Co-Evolution

## Knowledge Gaps
- **141 isolated node(s):** `Removes reasoning tags (<think>...</think> or <thought>...</thought>)     leavin`, `Classifies the task into Category and Subcategory for Task-Aware Retrieval.`, `Main MemCollab pipeline: dual trajectory → contrastive analysis → Skill.md`, `List all generated skills`, `Task-Aware Retrieval: Classifies the question and searches for compatible Skills` (+136 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 16`** (2 nodes): `api skills endpoint`, `loadSkillList function`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 17`** (2 nodes): `api craft endpoint`, `craftSkill function`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 18`** (2 nodes): `TurboQuant ref`, `TurboQuant Paper`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 19`** (2 nodes): `MemCollab ref`, `MemCollab Paper`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (2 nodes): `Strong Agent`, `Weak Agent`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (2 nodes): `S-PATH-RAG ref`, `S-PATH-RAG Paper`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (1 nodes): `Code.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (1 nodes): `True if gates diverged from zero.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (1 nodes): `Carrega projector treinado do disco.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `Heurística: se os pesos divergiram de eye_, o projector foi treinado.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (1 nodes): `anthropic.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `Generates a text response.          Args:             messages: Message list`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `Generates semantic embedding (for vector search).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `openai.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `Saves a skill (markdown + metadata).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `Returns the Markdown content of a skill.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `Returns the metadata of a skill.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `Returns the complete skill bundle including all folders (reference, template, as`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `Returns the local workspace path, if applicable.         Returns None for pure`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `Lists all skills in the store.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `Removes a skill from the store.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `Returns the skill graph.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `Updates the skill graph.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `Saves the quantized vector (TurboQuant) of a skill.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `Factory that returns the correct adapter based on URI.          Examples:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `FastAPI`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `Uvicorn`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `Pydantic`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `MCP`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `SkillCrafter UI`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `EvoSkills Paper`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `Co-Evolution`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `LLMMessage` connect `Base Abstractions` to `App & Bootstrap Layer`, `Cross-Attention Injection`, `Craft & Evolve APIs`, `ABC Interfaces`, `Retrieval System`?**
  _High betweenness centrality (0.116) - this node is a cross-community bridge._
- **Why does `openskill CLI — The Swiss Army Knife for Geometric Skill Memory ===============` connect `Craft & Evolve APIs` to `Base Abstractions`, `App & Bootstrap Layer`?**
  _High betweenness centrality (0.105) - this node is a cross-community bridge._
- **Why does `list()` connect `Cross-Attention Injection` to `Skill Legacy`, `App & Bootstrap Layer`, `Client & Core API`, `Skill System`?**
  _High betweenness centrality (0.104) - this node is a cross-community bridge._
- **Are the 89 inferred relationships involving `LLMMessage` (e.g. with `OpenSkillClient` and `OpenSkillClient — Unified High-Level API ======================================`) actually correct?**
  _`LLMMessage` has 89 INFERRED edges - model-reasoned connections that need verification._
- **Are the 77 inferred relationships involving `BaseLLMProvider` (e.g. with `OpenSkill Core — Pure Business Logic, Zero I/O ================================` and `OpenSkillClient`) actually correct?**
  _`BaseLLMProvider` has 77 INFERRED edges - model-reasoned connections that need verification._
- **Are the 60 inferred relationships involving `OpenRouterProvider` (e.g. with `bootstrap_data.py — Synthetic Dataset Generator for Path Scorer ===============` and `Auto-detects embedding dimension from available skills.     Prioritizes 1536d (`) actually correct?**
  _`OpenRouterProvider` has 60 INFERRED edges - model-reasoned connections that need verification._
- **Are the 45 inferred relationships involving `LocalDiskStore` (e.g. with `bootstrap_data.py — Synthetic Dataset Generator for Path Scorer ===============` and `Auto-detects embedding dimension from available skills.     Prioritizes 1536d (`) actually correct?**
  _`LocalDiskStore` has 45 INFERRED edges - model-reasoned connections that need verification._