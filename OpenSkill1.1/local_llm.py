"""
local_llm.py — Local Inference & Soft Latent Injection Engine
=============================================================
Uses local HuggingFace models when available.
Falls back to OpenRouter (Mistral) when HF models are unavailable.
Implements the S-Path-RAG Soft Latent Injection.
"""
import numpy as np
import torch
import torch.nn as nn
import os
import httpx
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

from openskill.utils.config import get_openrouter_key

# ── Model Configuration ──────────────────────────────────────────────────────
LLM_MODEL_ID = "Qwen/Qwen3.5-0.8B"  # Chat/Instruct Model
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# OpenRouter fallback configuration
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = get_openrouter_key()
FALLBACK_MODEL = "mistralai/mistral-7b-instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── HF Model Loading with Fallback ──────────────────────────────────────────
HF_AVAILABLE = False

print(f"Loading Embedding Model: {EMBED_MODEL_ID}...")
embedder = SentenceTransformer(EMBED_MODEL_ID, device=device)
EMBED_DIM = embedder.get_sentence_embedding_dimension()

try:
    print(f"Loading LLM: {LLM_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_ID,
        trust_remote_code=True
    )
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    LLM_HIDDEN_SIZE = llm_model.config.hidden_size
    HF_AVAILABLE = True
    print(f"✓ HF LLM loaded successfully. Hidden size: {LLM_HIDDEN_SIZE}")
except Exception as e:
    print(f"✗ Failed to load HF LLM ({LLM_MODEL_ID}): {e}")
    print(f"→ Falling back to OpenRouter with {FALLBACK_MODEL}")
    HF_AVAILABLE = False
    LLM_HIDDEN_SIZE = 4096  # Mistral default hidden size
    tokenizer = None
    llm_model = None


# ── Projection Layer (S-Path-RAG Alignment) ────────────────────────────────
# The S-Path-RAG paper projects the Skill vector to the internal dimension of the LLM.
# Ideally, this layer should be trained (Alignment Loss). For zero-shot inference,
# we initialize a linear layer.
class SkillProjector(nn.Module):
    def __init__(self, embed_dim, llm_dim, dtype=torch.float16):
        super().__init__()
        self.proj = nn.Linear(embed_dim, llm_dim, dtype=dtype)

    def forward(self, x):
        return self.proj(x)


# Dynamic Projector Pattern:
# Instead of static initialization with fixed EMBED_DIM, the projector is created
# dynamically based on actual skill vector dimensions. This solves tensor shape
# mismatches when skill vectors come from different embedding models.
#
# Benefits:
# - Backward compatible with existing 256D quantized skills
# - Forward compatible when embedding models change (256D → 384D → 512D)
# - No need to regenerate skill vectors when embedding model changes
# - Eliminates conflicts between hardcoded EMBED_DIM constants in different modules
projector = None


# ── Core Functions ─────────────────────────────────────────────────────────────

def get_embedding(text: str) -> np.ndarray:
    """Generates a real embedding using SentenceTransformer."""
    with torch.no_grad():
        vec = embedder.encode(text, normalize_embeddings=True)
    return vec


def _openrouter_generate(prompt: str, max_tokens: int = 1500) -> str:
    """Fallback to OpenRouter API when HF models unavailable."""
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "HF models unavailable and OPENROUTER_API_KEY not set. "
            "Set OPENROUTER_API_KEY in .env.local for fallback."
        )

    
    messages = [{"role": "user", "content": prompt}]
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://openskill.local",
        "X-Title": "OpenSkill",
    }
    payload = {
        "model": FALLBACK_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    
    try:
        response = httpx.post(OPENROUTER_URL, headers=headers, json=payload, timeout=180.0)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"OpenRouter API error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        raise RuntimeError(f"OpenRouter fallback failed: {e}")


def generate_text(prompt: str, max_tokens: int = 1500) -> str:
    """Default text generation (used for Trace2Skill and MemCollab)."""
    if not HF_AVAILABLE:
        return _openrouter_generate(prompt, max_tokens)
    
    messages = [{"role": "user", "content": prompt}]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text_input, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True
        )

    # Decodes only the newly generated tokens
    generated = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def generate_with_soft_latents(prompt: str, skill_vectors: list[np.ndarray], max_tokens: int = 1500) -> str:
    """
    S-Path-RAG Soft Latent Injection:
    Attaches de-quantized quantized vectors DIRECTLY to the LLM tensors.
    
    When HF models unavailable, falls back to OpenRouter (soft latents not supported).
    """
    # Fallback to OpenRouter if HF unavailable
    if not HF_AVAILABLE:
        if skill_vectors:
            # Inject skill context into prompt when using OpenRouter fallback
            skill_context = f"\n\n[Relevant skills context: {len(skill_vectors)} skill vectors available]"
            prompt_with_context = prompt + skill_context
        else:
            prompt_with_context = prompt
        return _openrouter_generate(prompt_with_context, max_tokens)
    
    messages = [{"role": "user", "content": prompt}]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text_input, return_tensors="pt").to(device)

    # 1. Get normal token embeddings from the user prompt
    with torch.no_grad():
        prompt_embeds = llm_model.get_input_embeddings()(inputs.input_ids)

    # 2. Prepare Skill Soft Latents
    if skill_vectors:
        # Convert list of numpy arrays to tensor [Num_Skills, Embed_Dim]
        # FIX: Avoid creating tensor from list of numpy arrays (PyTorch anti-pattern)
        # First convert to single numpy array, then to tensor with model's dtype
        skill_array = np.array(skill_vectors, dtype=np.float32)
        skills_tensor = torch.from_numpy(skill_array).to(device=device, dtype=llm_model.dtype)

        # 3. Dynamic Projector Creation: Auto-detect input dimension
        actual_embed_dim = skills_tensor.shape[-1]  # Get actual dimension from skill vectors

        # Validation: Check for consistent dimensions across all skill vectors
        if len(skills_tensor.shape) != 2:
            raise ValueError(f"Expected skill vectors as 2D tensor [num_skills, embed_dim], got shape {skills_tensor.shape}")

        # Create projector with correct input dimension and dtype
        global projector
        if projector is None or projector.proj.in_features != actual_embed_dim:
            print(f"✓ Creating dynamic projector: {actual_embed_dim}D → {LLM_HIDDEN_SIZE}D")
            projector = SkillProjector(actual_embed_dim, LLM_HIDDEN_SIZE, dtype=skills_tensor.dtype).to(device)

        # 4. Project from actual embedding dimension to LLM dimension
        with torch.no_grad():
            projected_skills = projector(skills_tensor).unsqueeze(0)  # [1, Num_Skills, LLM_Dim]

        # 5. Vector Injection (Concatenation at the beginning)
        combined_embeds = torch.cat([projected_skills, prompt_embeds], dim=1)

        # 6. Adjust attention mask for the new size
        batch_size = inputs.attention_mask.shape[0]
        num_skills = projected_skills.shape[1]
        skill_mask = torch.ones((batch_size, num_skills), dtype=inputs.attention_mask.dtype, device=device)
        combined_mask = torch.cat([skill_mask, inputs.attention_mask], dim=1)
    else:
        combined_embeds = prompt_embeds
        combined_mask = inputs.attention_mask

    # 6. Generate the geometrically conditioned response
    with torch.no_grad():
        outputs = llm_model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)