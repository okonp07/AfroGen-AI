from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .latent import apply_delta, build_latent_matrix
from .prompting import PromptProfile, parse_prompt
from .render import render_portrait


@dataclass
class GenerationResult:
    profile: PromptProfile
    base_latent: np.ndarray
    edited_latent: np.ndarray
    image: object


class AfroGenPipeline:
    def __init__(self, image_size: int = 512, latent_shape: tuple[int, int] = (4, 4)) -> None:
        self.image_size = image_size
        self.latent_shape = latent_shape

    def generate(self, prompt: str, seed: int = 7, delta: np.ndarray | None = None) -> GenerationResult:
        profile = parse_prompt(prompt)
        base_latent = build_latent_matrix(prompt=prompt, seed=seed, shape=self.latent_shape)
        edited_latent = apply_delta(base_latent, delta)
        image = render_portrait(profile=profile, latent=edited_latent, size=self.image_size)
        return GenerationResult(
            profile=profile,
            base_latent=base_latent,
            edited_latent=edited_latent,
            image=image,
        )
