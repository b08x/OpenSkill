"""
GNN Encoder — S-Path-RAG Neural Encodings
=========================================
"""
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
# We use safetensors as it is faster, more secure, and an industry standard (SaaS)
from safetensors.torch import save_file, load_file
import structlog

log = structlog.get_logger()


class SkillGNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.conv1 = gnn.GCNConv(input_dim, hidden_dim)
        self.conv2 = gnn.GCNConv(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

        # Safe Identity Initialization
        with torch.no_grad():
            # Accesses the weight of the internal GCN linear layer
            if hasattr(self.conv1, 'lin'):
                nn.init.eye_(self.conv1.lin.weight)
                nn.init.eye_(self.conv2.lin.weight)
            else:
                # Fallback for versions where the weight is at the root
                nn.init.eye_(self.conv1.weight)
                nn.init.eye_(self.conv2.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        identity = x  # Saves the original

        x = self.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        # S-Path-RAG Residual Connection: Maintains the original semantic base
        return x + identity

def encode_graph_embeddings(
    metas: list[Any],
    edges: list[dict],
    embed_dim: int,
    save_dir: Path = None  # Receives the root folder to save/load the brain
) -> dict[str, np.ndarray]:

    if not metas: return {}

    # Ultra-robust helper for getting properties without type errors
    def _val(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # Helper to extract the correct embedding for the required dimension
    def get_emb(m):
        emb = _val(m, 'embedding')
        if emb and len(emb) == embed_dim:
            return emb

        # Tries to find in multi-view profiles
        vectors = _val(m, 'vectors') or {}
        if isinstance(vectors, dict):
            for p in vectors.values():
                p_dim = _val(p, 'dimension', 0)
                p_emb = _val(p, 'embedding')
                if p_dim == embed_dim and p_emb:
                    return p_emb

        return [0.0] * embed_dim

    # 1. Map IDs to indices (Now safe against AttributeError)
    node_list = sorted([_val(m, 'id') for m in metas if _val(m, 'id')])
    id_map = {sid: i for i, sid in enumerate(node_list)}

    # 2. Create attribute matrix (x) - USING THE HELPER
    x = torch.tensor([get_emb(m) for m in metas if _val(m, 'id')], dtype=torch.float32)

    # 3. Create edges (edge_index)
    edge_idx = []
    for e in edges:
        if e['from'] in id_map and e['to'] in id_map:
            edge_idx.append([id_map[e['from']], id_map[e['to']]])

    if edge_idx:
        edge_index = torch.tensor(edge_idx, dtype=torch.long).t().contiguous()
    else:
        # Creates a valid empty tensor if there are no edges
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # 4. Initialize GNN
    model = SkillGNN(input_dim=embed_dim, hidden_dim=embed_dim)

    # ─── GNN PERSISTENCE LOGIC (SAAS READY) ───
    if save_dir:
        model_path = save_dir / "gnn_brain.safetensors"
        if model_path.exists():
            try:
                # Loads the brain that has already evolved
                weights = load_file(model_path)
                model.load_state_dict(weights)
                log.info("gnn.loaded_weights", path=str(model_path))
            except Exception as e:
                log.error("gnn.load_error", error=str(e))
        else:
            # First run: saves initial weights for future training
            save_file(model.state_dict(), model_path)
            log.info("gnn.initialized_weights", path=str(model_path))

    # 5. Inference
    model.eval()
    with torch.no_grad():
        updated_embeddings = model(x, edge_index)

    return {node_list[i]: updated_embeddings[i].numpy() for i in range(len(node_list))}
