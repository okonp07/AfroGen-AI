from __future__ import annotations

import numpy as np


def ensure_delta_state(session_state: object, shape: tuple[int, int]) -> np.ndarray:
    if "latent_delta" not in session_state:
        session_state.latent_delta = np.zeros(shape, dtype=np.float32)
    return session_state.latent_delta
