from __future__ import annotations

import numpy as np

from afrogen.generation.pipeline import AfroGenPipeline, GenerationResult

from .base import BackendInfo


class SyntheticAfroGenBackend:
    def __init__(self, image_size: int, latent_shape: tuple[int, int]) -> None:
        self.pipeline = AfroGenPipeline(image_size=image_size, latent_shape=latent_shape)
        self.info = BackendInfo(
            name="synthetic",
            description="Deterministic illustration backend used for product scaffolding.",
            editable_latent=True,
            ready_for_training=False,
            load_state="ready",
        )

    def generate(self, prompt: str, seed: int = 7, delta: np.ndarray | None = None) -> GenerationResult:
        return self.pipeline.generate(prompt=prompt, seed=seed, delta=delta)
