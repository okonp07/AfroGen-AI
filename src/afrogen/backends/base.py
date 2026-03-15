from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from afrogen.generation.pipeline import GenerationResult


@dataclass(frozen=True)
class BackendInfo:
    name: str
    description: str
    editable_latent: bool
    ready_for_training: bool


class AfroGenBackend(Protocol):
    info: BackendInfo

    def generate(self, prompt: str, seed: int = 7, delta: np.ndarray | None = None) -> GenerationResult:
        """Generate a portrait result."""
