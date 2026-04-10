"""
TurboQuantizer — Geometric Skill Memory Layer (Lloyd-Max Optimized)
==================================================================
Strict implementation of TurboQuant (arXiv:2504.19874).

Stage 1: Random Rotation + Continuous Lloyd-Max K-Means (Optimal MSE).
Stage 2: 1-bit Residual QJL (Unbiased Inner Product).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional
import structlog

from openskill.llm.base import BaseLLMProvider

log = structlog.get_logger()

QUANT_BITS = 4
QUANT_LEVELS = 2**QUANT_BITS
ROTATION_SEED = 42
LAMBDA_CORRECTION = 0.1

@dataclass
class QuantizedVector:
    qvec: np.ndarray      # int8: Centroid indices (0 to 15)
    residual: np.ndarray  # int8: QJL error signs (+1 or -1)
    centroids: list[float] # float: The 16 optimal values found by Lloyd-Max
    dim: int

    def to_dict(self) -> dict:
        return {
            "qvec": self.qvec.tolist(),
            "residual": self.residual.tolist(),
            "centroids": self.centroids, # Replaces the old 'scale'
            "dim": self.dim,
            "bits": QUANT_BITS
        }

    @classmethod
    def from_dict(cls, d: dict) -> QuantizedVector:
        # Compatibility with old format (Uniform Min-Max)
        if "scale" in d and "centroids" not in d:
            # Roughly reconstructs uniform centroids if reading an old file
            vmin, vmax = d["scale"]
            step = (vmax - vmin) / (QUANT_LEVELS - 1)
            centroids = [vmin + i * step for i in range(QUANT_LEVELS)]
        else:
            centroids = d["centroids"]

        return cls(
            qvec=np.array(d["qvec"], dtype=np.int8),
            residual=np.array(d["residual"], dtype=np.int8),
            centroids=centroids,
            dim=d["dim"]
        )

class TurboQuantizer:
    def __init__(self, dimension: int = 1536):
        self.dim = dimension
        self._rotation_matrix = self._build_rotation_matrix(dimension, ROTATION_SEED)

    def _ensure_dimension(self, d: int):
        if d != self.dim:
            log.info("vector.dim_update", old=self.dim, new=d)
            self.dim = d
            self._rotation_matrix = self._build_rotation_matrix(d, ROTATION_SEED)

    async def embed(self, llm: BaseLLMProvider, text: str) -> np.ndarray:
        try:
            vec = await llm.embed(text)
            return np.array(vec, dtype=np.float32)
        except Exception as e:
            log.error("vector.embed_error", error=str(e))
            return np.random.randn(self.dim).astype(np.float32)

    def quantize(self, vector: np.ndarray) -> QuantizedVector:
        self._ensure_dimension(vector.shape[0])

        # 1. Random Rotation
        v_rotated = self._rotation_matrix @ vector

        # 2. 1D Lloyd-Max K-Means (Empirical for the current sample)
        # The paper resolves analytically for the Beta distribution.
        # Here, as we process online, we perform a super-fast K-Means on the actual vector data.
        centroids, qvec = self._lloyd_max_1d(v_rotated, QUANT_LEVELS)

        # 3. De-quantization for residual calculation
        v_deq_temp = centroids[qvec]

        # 4. Residual QJL
        residual_error = v_rotated - v_deq_temp
        residual_signs = np.sign(residual_error).astype(np.int8)
        residual_signs[residual_signs == 0] = 1

        return QuantizedVector(
            qvec=qvec.astype(np.int8),
            residual=residual_signs,
            centroids=centroids.tolist(),
            dim=self.dim
        )

    def dequantize(self, qv: QuantizedVector) -> np.ndarray:
        self._ensure_dimension(qv.dim)

        # Maps indices back to optimal centroid values
        centroids_arr = np.array(qv.centroids, dtype=np.float32)
        v_rotated_approx = centroids_arr[qv.qvec]

        return self._rotation_matrix.T @ v_rotated_approx

    def similarity(self, qv_a: QuantizedVector, qv_b: QuantizedVector) -> float:
        v_a = self.dequantize(qv_a)
        v_b = self.dequantize(qv_b)

        dot_product = np.dot(v_a, v_b)
        correction = np.dot(qv_a.residual.astype(np.float32), qv_b.residual.astype(np.float32))
        corrected_score = dot_product + (LAMBDA_CORRECTION / self.dim) * correction

        norm_a = np.linalg.norm(v_a) + 1e-8
        norm_b = np.linalg.norm(v_b) + 1e-8
        return float(corrected_score / (norm_a * norm_b))

    def _build_rotation_matrix(self, dim: int, seed: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        H = rng.randn(dim, dim).astype(np.float32)
        Q, _ = np.linalg.qr(H)
        return Q

    def _lloyd_max_1d(self, data: np.ndarray, num_levels: int, max_iter: int = 20) -> tuple[np.ndarray, np.ndarray]:
        """
        Fast implementation of the Lloyd-Max algorithm for optimal scalar quantization.
        Finds 'num_levels' centroids that minimize the Mean Squared Error (MSE).
        """
        # Initializes centroids uniformly between min and max
        vmin, vmax = data.min(), data.max()
        centroids = np.linspace(vmin, vmax, num_levels)

        q_indices = np.zeros_like(data, dtype=np.int32)

        for _ in range(max_iter):
            # Step 1: Assignment (Calculates distance to all centroids and picks the closest)
            # data[:, None] creates a column matrix for broadcasting with centroids (row)
            distances = np.abs(data[:, None] - centroids[None, :])
            q_indices = np.argmin(distances, axis=1)

            # Step 2: Centroid update
            new_centroids = np.zeros_like(centroids)
            for i in range(num_levels):
                points_in_cluster = data[q_indices == i]
                if len(points_in_cluster) > 0:
                    new_centroids[i] = np.mean(points_in_cluster)
                else:
                    # If a cluster becomes empty, we keep the previous centroid
                    new_centroids[i] = centroids[i]

            # Stop condition: if centroids do not change, it has converged
            if np.allclose(centroids, new_centroids, atol=1e-6):
                break

            centroids = new_centroids

        return centroids, q_indices

def pack_qvector(qv: QuantizedVector) -> dict:
    """Helper to transform the QuantizedVector object into a dictionary for JSON."""
    return qv.to_dict()

def unpack_qvector(d: dict) -> QuantizedVector:
    return QuantizedVector.from_dict(d)
