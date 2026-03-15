from __future__ import annotations

import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

from afrogen.generation.latent import apply_delta, build_latent_matrix
from afrogen.generation.pipeline import GenerationResult
from afrogen.generation.prompting import parse_prompt

from .artifacts import BackendArtifact, load_backend_artifact, validate_backend_artifact
from .base import BackendInfo
from .resolve import resolve_artifact_reference


class TrainedAfroGenBackend:
    def __init__(
        self,
        name: str,
        image_size: int,
        latent_shape: tuple[int, int],
        artifact_path: str | None = None,
    ) -> None:
        self.name = name
        self.image_size = image_size
        self.latent_shape = latent_shape
        self.project_root = Path(__file__).resolve().parents[3]
        self.artifact_reference = artifact_path or "models/trained_backend_stub.json"
        self.artifact_path = resolve_artifact_reference(self.artifact_reference, self.project_root)
        self.artifact_metadata = self._load_artifact_metadata()
        self.rollout_state, self.rollout_message = validate_backend_artifact(self.artifact_metadata, self.project_root)
        self.info = BackendInfo(
            name=name,
            description="Placeholder backend for the future trained afrocentric face model.",
            editable_latent=True,
            ready_for_training=True,
            load_state=self._load_state(),
            rollout_state=self.rollout_state,
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

    def _load_artifact_metadata(self) -> BackendArtifact | None:
        return load_backend_artifact(self.artifact_reference, self.project_root)

    def _load_state(self) -> str:
        if not self.artifact_metadata:
            return "missing"
        return self.artifact_metadata.status

    def _backend_message(self) -> str:
        if not self.artifact_metadata:
            return "Trained backend placeholder. Run scripts/build_training_stub.py to create the first backend artifact stub."
        prefix = f"Artifact status: {self.artifact_metadata.status}."
        return f"{prefix} {self.artifact_metadata.message} Rollout state: {self.rollout_state}. {self.rollout_message}"

    def summary(self) -> dict:
        return {
            "backend_name": self.name,
            "load_state": self.info.load_state,
            "rollout_state": self.info.rollout_state,
            "artifact_path": str(self.artifact_path),
            "checkpoint_path": self.artifact_metadata.checkpoint_path if self.artifact_metadata else "",
            "latent_editor_path": self.artifact_metadata.latent_editor_path if self.artifact_metadata else "",
        }

    def _render_placeholder(self, prompt: str) -> Image.Image:
        image = Image.new("RGB", (self.image_size, self.image_size), (247, 236, 221))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle([32, 32, self.image_size - 32, self.image_size - 32], radius=24, outline=(129, 93, 65), width=4)
        text = [
            f"Backend: {self.name}",
            f"Status: {self.info.load_state}",
            f"Rollout: {self.info.rollout_state}",
            "Next step:",
            "Load a trained model",
            "and keep the same",
            "generate(prompt, seed, delta)",
            "interface.",
            "",
            f"Checkpoint: {(self.artifact_metadata.checkpoint_path if self.artifact_metadata else 'not set')[:32]}",
            "",
            f"Prompt: {prompt[:48]}",
        ]
        draw.multiline_text((56, 64), "\n".join(text), fill=(65, 48, 38), spacing=8)
        return image
