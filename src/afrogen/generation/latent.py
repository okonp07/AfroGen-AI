from __future__ import annotations

import hashlib

import numpy as np


def build_latent_matrix(prompt: str, seed: int, shape: tuple[int, int]) -> np.ndarray:
    digest = hashlib.sha256(f"{prompt}|{seed}".encode("utf-8")).digest()
    buffer = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
    needed = shape[0] * shape[1]
    tiled = np.resize(buffer, needed)
    matrix = tiled.reshape(shape)
    return (matrix / 127.5) - 1.0


def apply_delta(latent: np.ndarray, delta: np.ndarray | None) -> np.ndarray:
    if delta is None:
        return latent
    return np.clip(latent + delta, -1.0, 1.0)
