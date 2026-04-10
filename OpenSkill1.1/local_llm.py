"""
local_llm.py — Local Inference & Soft Latent Injection Engine
=============================================================
Replaces OpenRouter with local HuggingFace models.
Implements the S-Path-RAG Soft Latent Injection.
"""
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# ── Model Configuration ──────────────────────────────────────────────────────
# We use the Chat model (without -Base) to support apply_chat_template
LLM_MODEL_ID = "Qwen/Qwen3.5-0.8B"  # Chat/Instruct Model
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading Embedding Model: {EMBED_MODEL_ID}...")
embedder = SentenceTransformer(EMBED_MODEL_ID, device=device)
EMBED_DIM = embedder.get_sentence_embedding_dimension()

print(f"Loading LLM: {LLM_MODEL_ID}...")
# For very recent Qwen models, trust_remote_code=True is essential
tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL_ID,
    trust_remote_code=True
)

llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True # Added here!
)
LLM_HIDDEN_SIZE = llm_model.config.hidden_size


# ── Projection Layer (S-Path-RAG Alignment) ────────────────────────────────
# The S-Path-RAG paper projects the Skill vector to the internal dimension of the LLM.
# Ideally, this layer should be trained (Alignment Loss). For zero-shot inference,
# we initialize a linear layer.
class SkillProjector(nn.Module):
    def __init__(self, embed_dim, llm_dim):
        super().__init__()
        self.proj = nn.Linear(embed_dim, llm_dim, dtype=torch.float16)

    def forward(self, x):
        return self.proj(x)


projector = SkillProjector(EMBED_DIM, LLM_HIDDEN_SIZE).to(device)


# ── Core Functions ─────────────────────────────────────────────────────────────

def get_embedding(text: str) -> np.ndarray:
    """Generates a real embedding using SentenceTransformer."""
    with torch.no_grad():
        vec = embedder.encode(text, normalize_embeddings=True)
    return vec


def generate_text(prompt: str, max_tokens: int = 1500) -> str:
    """Default text generation (used for Trace2Skill and MemCollab)."""
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
    """
    messages = [{"role": "user", "content": prompt}]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text_input, return_tensors="pt").to(device)

    # 1. Get normal token embeddings from the user prompt
    with torch.no_grad():
        prompt_embeds = llm_model.get_input_embeddings()(inputs.input_ids)

    # 2. Prepare Skill Soft Latents
    if skill_vectors:
        # Convert list of numpy arrays to tensor [Num_Skills, Embed_Dim]
        skills_tensor = torch.tensor(skill_vectors, dtype=torch.float16, device=device)

        # 3. Project from Embedding dimension (e.g., 384) to LLM dimension (e.g., 4096)
        with torch.no_grad():
            projected_skills = projector(skills_tensor).unsqueeze(0)  # [1, Num_Skills, LLM_Dim]

        # 4. Vector Injection (Concatenation at the beginning)
        combined_embeds = torch.cat([projected_skills, prompt_embeds], dim=1)

        # 5. Adjust attention mask for the new size
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