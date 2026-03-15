from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageEnhance

from afrogen.generation.latent import apply_delta, build_latent_matrix
from afrogen.generation.pipeline import GenerationResult
from afrogen.generation.prompting import parse_prompt
from afrogen.generation.render import render_portrait

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
        inference_client: object | None = None,
    ) -> None:
        self.name = name
        self.image_size = image_size
        self.latent_shape = latent_shape
        self.project_root = Path(__file__).resolve().parents[3]
        self.artifact_reference = artifact_path or "models/trained_backend_stub.json"
        self._inference_client = inference_client
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
        backend_name = self.name
        backend_message = self._backend_message()

        if self.rollout_state not in {"ready_for_inference", "ready_for_hosted_inference"}:
            image = self._render_placeholder(prompt=prompt)
        elif not self.artifact_metadata or not self.artifact_metadata.hosted_model_id.strip():
            image = self._render_placeholder(prompt=prompt)
            backend_message = (
                "Artifact validated for inference, but no hosted_model_id is declared yet. "
                "Publish hosted model metadata to enable real Hugging Face inference."
            )
        else:
            try:
                image = self._generate_hosted_image(prompt=prompt, profile=profile, edited_latent=edited_latent, seed=seed)
                backend_message = (
                    f"Hosted hybrid inference active via {self.artifact_metadata.hosted_model_id}. "
                    f"Rollout state: {self.rollout_state}."
                )
            except Exception as exc:
                image = render_portrait(profile=profile, latent=edited_latent, size=self.image_size)
                backend_name = "synthetic"
                backend_message = (
                    "Hosted hybrid inference failed and synthetic fallback was used. "
                    f"Reason: {exc}"
                )

        return GenerationResult(
            profile=profile,
            base_latent=base_latent,
            edited_latent=edited_latent,
            image=image,
            backend_name=backend_name,
            backend_message=backend_message,
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
            "hosted_model_id": self.artifact_metadata.hosted_model_id if self.artifact_metadata else "",
        }

    def _build_client(self):
        if self._inference_client is not None:
            return self._inference_client
        if not self.artifact_metadata:
            raise RuntimeError("No backend artifact metadata is available.")
        try:
            from huggingface_hub import InferenceClient
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "huggingface_hub is required for hosted hybrid inference. Install it from requirements.txt."
            ) from exc
        self._inference_client = InferenceClient(
            model=self.artifact_metadata.hosted_model_id,
            token=os.getenv("HF_TOKEN"),
            timeout=self.artifact_metadata.inference_timeout_seconds,
        )
        return self._inference_client

    def _compose_prompt(self, prompt: str, profile) -> str:
        if not self.artifact_metadata:
            return prompt
        descriptors = [
            f"age group: {profile.age_group}",
            f"skin tone: {profile.skin_tone}",
            f"hairstyle: {profile.hairstyle}",
            f"expression: {profile.expression}",
        ]
        if profile.accessory != "none":
            descriptors.append(f"accessory: {profile.accessory}")
        return f"{self.artifact_metadata.prompt_prefix}. {prompt}. " + ", ".join(descriptors)

    def _generate_hosted_image(self, prompt: str, profile, edited_latent: np.ndarray, seed: int) -> Image.Image:
        if not self.artifact_metadata:
            raise RuntimeError("No backend artifact metadata is available.")
        client = self._build_client()
        image = client.text_to_image(
            prompt=self._compose_prompt(prompt, profile),
            negative_prompt=self.artifact_metadata.negative_prompt or None,
            guidance_scale=self.artifact_metadata.guidance_scale,
            num_inference_steps=self.artifact_metadata.num_inference_steps,
            width=self.artifact_metadata.output_width or self.image_size,
            height=self.artifact_metadata.output_height or self.image_size,
            seed=seed,
        )
        return self._apply_latent_edit_postprocess(image.convert("RGB"), edited_latent)

    def _apply_latent_edit_postprocess(self, image: Image.Image, edited_latent: np.ndarray) -> Image.Image:
        normalized = (edited_latent + 1.0) / 2.0
        brightness = 0.9 + float(np.mean(normalized[0])) * 0.25
        contrast = 0.9 + float(np.mean(normalized[1])) * 0.3
        saturation = 0.85 + float(np.mean(normalized[2])) * 0.35
        sharpness = 0.9 + float(np.mean(normalized[3])) * 0.4

        hybrid = ImageEnhance.Brightness(image).enhance(brightness)
        hybrid = ImageEnhance.Contrast(hybrid).enhance(contrast)
        hybrid = ImageEnhance.Color(hybrid).enhance(saturation)
        hybrid = ImageEnhance.Sharpness(hybrid).enhance(sharpness)
        return hybrid

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
            f"Hosted model: {(self.artifact_metadata.hosted_model_id if self.artifact_metadata else 'not set')[:32]}",
            "",
            f"Prompt: {prompt[:48]}",
        ]
        draw.multiline_text((56, 64), "\n".join(text), fill=(65, 48, 38), spacing=8)
        return image
