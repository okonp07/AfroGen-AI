from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

from afrogen.generation.latent import apply_delta, build_latent_matrix
from afrogen.generation.pipeline import GenerationResult
from afrogen.generation.prompting import parse_prompt

from .base import BackendInfo


class TrainedAfroGenBackend:
    def __init__(self, name: str, image_size: int, latent_shape: tuple[int, int]) -> None:
        self.name = name
        self.image_size = image_size
        self.latent_shape = latent_shape
        self.artifact_metadata = self._load_artifact_metadata()
        self.info = BackendInfo(
            name=name,
            description="Placeholder backend for the future trained afrocentric face model.",
            editable_latent=True,
            ready_for_training=True,
        )

    def generate(self, prompt: str, seed: int = 7, delta: np.ndarray | None = None) -> GenerationResult:
        profile = parse_prompt(prompt)
        base_latent = build_latent_matrix(prompt=prompt, seed=seed, shape=self.latent_shape)
        edited_latent = apply_delta(base_latent, delta)
        image = self._render_placeholder(prompt=prompt)
        return GenerationResult(
            profile=profile,
            base_latent=base_latent,
            edited_latent=edited_latent,
            image=image,
            backend_name=self.name,
            backend_message=self._backend_message(),
        )

    def _load_artifact_metadata(self) -> dict:
        artifact_path = Path(__file__).resolve().parents[3] / "models" / "trained_backend_stub.json"
        if artifact_path.exists():
            return json.loads(artifact_path.read_text(encoding="utf-8"))
        return {}

    def _backend_message(self) -> str:
        if not self.artifact_metadata:
            return "Trained backend placeholder. Run scripts/build_training_stub.py to create the first backend artifact stub."
        return self.artifact_metadata.get("message", "Trained backend stub active.")

    def _render_placeholder(self, prompt: str) -> Image.Image:
        image = Image.new("RGB", (self.image_size, self.image_size), (247, 236, 221))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle([32, 32, self.image_size - 32, self.image_size - 32], radius=24, outline=(129, 93, 65), width=4)
        text = [
            f"Backend: {self.name}",
            "Status: placeholder",
            "Next step:",
            "Load a trained model",
            "and keep the same",
            "generate(prompt, seed, delta)",
            "interface.",
            "",
            f"Prompt: {prompt[:48]}",
        ]
        draw.multiline_text((56, 64), "\n".join(text), fill=(65, 48, 38), spacing=8)
        return image
