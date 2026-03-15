from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from afrogen.backends import create_backend
from afrogen.backends.artifacts import BackendArtifact, save_backend_artifact


class DummyInferenceClient:
    def __init__(self, image: Image.Image | None = None, error: Exception | None = None) -> None:
        self.image = image or Image.new("RGB", (256, 256), (120, 80, 60))
        self.error = error

    def text_to_image(self, **_: object) -> Image.Image:
        if self.error is not None:
            raise self.error
        return self.image


class BackendTests(unittest.TestCase):
    def test_factory_returns_synthetic_backend(self) -> None:
        backend = create_backend("synthetic", image_size=256, latent_shape=(4, 4))
        result = backend.generate("A smiling Black man with locs", seed=3, delta=np.zeros((4, 4)))
        self.assertEqual(result.backend_name, "synthetic")
        self.assertEqual(result.edited_latent.shape, (4, 4))

    def test_factory_returns_trained_placeholder_backend(self) -> None:
        backend = create_backend("hybrid", image_size=256, latent_shape=(4, 4))
        result = backend.generate("A calm Black woman with braids", seed=5)
        self.assertEqual(result.backend_name, "hybrid")
        self.assertTrue(result.backend_message)

    def test_trained_backend_loads_ready_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "trained_backend_stub.json"
            checkpoint_path = Path(temp_dir) / "phase5.ckpt"
            checkpoint_path.write_text("weights", encoding="utf-8")
            save_backend_artifact(
                artifact_path,
                BackendArtifact(
                    backend_name="hybrid",
                    model_strategy="latent-diffusion-plus-editor",
                    baseline_model_family="sdxl-lora-plus-latent-editor",
                    status="ready",
                    message="Ready to load checkpoint metadata.",
                    checkpoint_path=str(checkpoint_path),
                    scheduler_name="EulerDiscreteScheduler",
                    supports_prompt_generation=True,
                    hosted_model_id="black-forest-labs/FLUX.1-schnell",
                ),
            )
            backend = create_backend("hybrid", image_size=256, latent_shape=(4, 4), artifact_path=artifact_path)
            self.assertEqual(backend.info.load_state, "ready")
            self.assertEqual(backend.info.rollout_state, "ready_for_hosted_inference")
            self.assertEqual(backend.summary()["checkpoint_path"], str(checkpoint_path))
            self.assertEqual(backend.summary()["hosted_model_id"], "black-forest-labs/FLUX.1-schnell")

    def test_trained_backend_uses_hosted_inference_when_model_id_is_present(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "trained_backend_stub.json"
            save_backend_artifact(
                artifact_path,
                BackendArtifact(
                    backend_name="hybrid",
                    model_strategy="latent-diffusion-plus-editor",
                    baseline_model_family="sdxl-lora-plus-latent-editor",
                    status="ready",
                    message="Hosted inference metadata is ready.",
                    checkpoint_path="missing/model.ckpt",
                    scheduler_name="EulerDiscreteScheduler",
                    supports_prompt_generation=True,
                    hosted_model_id="black-forest-labs/FLUX.1-schnell",
                ),
            )
            backend = create_backend(
                "hybrid",
                image_size=256,
                latent_shape=(4, 4),
                artifact_path=artifact_path,
                inference_client=DummyInferenceClient(),
            )
            result = backend.generate("A calm Black woman with braids", seed=5)
            self.assertEqual(result.backend_name, "hybrid")
            self.assertIn("Hosted hybrid inference active", result.backend_message)

    def test_trained_backend_falls_back_to_synthetic_if_hosted_inference_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "trained_backend_stub.json"
            save_backend_artifact(
                artifact_path,
                BackendArtifact(
                    backend_name="hybrid",
                    model_strategy="latent-diffusion-plus-editor",
                    baseline_model_family="sdxl-lora-plus-latent-editor",
                    status="ready",
                    message="Hosted inference metadata is ready.",
                    checkpoint_path="missing/model.ckpt",
                    scheduler_name="EulerDiscreteScheduler",
                    supports_prompt_generation=True,
                    hosted_model_id="black-forest-labs/FLUX.1-schnell",
                ),
            )
            backend = create_backend(
                "hybrid",
                image_size=256,
                latent_shape=(4, 4),
                artifact_path=artifact_path,
                inference_client=DummyInferenceClient(error=RuntimeError("upstream unavailable")),
            )
            result = backend.generate("A calm Black woman with braids", seed=5)
            self.assertEqual(result.backend_name, "synthetic")
            self.assertIn("synthetic fallback", result.backend_message)

    def test_trained_backend_marks_hosted_model_as_ready_without_local_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "trained_backend_stub.json"
            save_backend_artifact(
                artifact_path,
                BackendArtifact(
                    backend_name="hybrid",
                    model_strategy="latent-diffusion-plus-editor",
                    baseline_model_family="sdxl-lora-plus-latent-editor",
                    status="ready",
                    message="Hosted inference metadata is ready.",
                    checkpoint_path="missing/model.ckpt",
                    scheduler_name="EulerDiscreteScheduler",
                    supports_prompt_generation=True,
                    hosted_model_id="black-forest-labs/FLUX.1-schnell",
                ),
            )
            backend = create_backend("hybrid", image_size=256, latent_shape=(4, 4), artifact_path=artifact_path)
            self.assertEqual(backend.info.rollout_state, "ready_for_hosted_inference")

    def test_trained_backend_reports_checkpoint_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "trained_backend_stub.json"
            save_backend_artifact(
                artifact_path,
                BackendArtifact(
                    backend_name="hybrid",
                    model_strategy="latent-diffusion-plus-editor",
                    baseline_model_family="sdxl-lora-plus-latent-editor",
                    status="ready",
                    message="Metadata uploaded but checkpoint not present.",
                    checkpoint_path="missing/model.ckpt",
                    scheduler_name="EulerDiscreteScheduler",
                    supports_prompt_generation=True,
                ),
            )
            backend = create_backend("hybrid", image_size=256, latent_shape=(4, 4), artifact_path=artifact_path)
            self.assertEqual(backend.info.rollout_state, "checkpoint_missing")

    def test_trained_backend_reports_incomplete_inference_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "trained_backend_stub.json"
            checkpoint_path = Path(temp_dir) / "phase5.ckpt"
            checkpoint_path.write_text("weights", encoding="utf-8")
            save_backend_artifact(
                artifact_path,
                BackendArtifact(
                    backend_name="hybrid",
                    model_strategy="latent-diffusion-plus-editor",
                    baseline_model_family="sdxl-lora-plus-latent-editor",
                    status="ready",
                    message="Checkpoint uploaded but metadata is incomplete.",
                    checkpoint_path=str(checkpoint_path),
                    scheduler_name="",
                    supports_prompt_generation=False,
                ),
            )
            backend = create_backend("hybrid", image_size=256, latent_shape=(4, 4), artifact_path=artifact_path)
            self.assertEqual(backend.info.rollout_state, "incomplete_inference_contract")

    def test_trained_backend_reports_missing_latent_editor_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "trained_backend_stub.json"
            checkpoint_path = Path(temp_dir) / "phase5.ckpt"
            checkpoint_path.write_text("weights", encoding="utf-8")
            save_backend_artifact(
                artifact_path,
                BackendArtifact(
                    backend_name="hybrid",
                    model_strategy="latent-diffusion-plus-editor",
                    baseline_model_family="sdxl-lora-plus-latent-editor",
                    status="ready",
                    message="Checkpoint uploaded and latent editing declared.",
                    checkpoint_path=str(checkpoint_path),
                    latent_editor_path="missing/editor.pt",
                    scheduler_name="EulerDiscreteScheduler",
                    supports_prompt_generation=True,
                    supports_latent_editing=True,
                ),
            )
            backend = create_backend("hybrid", image_size=256, latent_shape=(4, 4), artifact_path=artifact_path)
            self.assertEqual(backend.info.rollout_state, "latent_editor_missing")


if __name__ == "__main__":
    unittest.main()
