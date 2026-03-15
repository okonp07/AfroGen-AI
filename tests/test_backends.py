from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from afrogen.backends import create_backend
from afrogen.backends.artifacts import BackendArtifact, save_backend_artifact


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
            save_backend_artifact(
                artifact_path,
                BackendArtifact(
                    backend_name="hybrid",
                    model_strategy="latent-diffusion-plus-editor",
                    baseline_model_family="sdxl-lora-plus-latent-editor",
                    status="ready",
                    message="Ready to load checkpoint metadata.",
                    checkpoint_path="models/checkpoints/phase5.ckpt",
                ),
            )
            backend = create_backend("hybrid", image_size=256, latent_shape=(4, 4), artifact_path=artifact_path)
            self.assertEqual(backend.info.load_state, "ready")
            self.assertEqual(backend.summary()["checkpoint_path"], "models/checkpoints/phase5.ckpt")


if __name__ == "__main__":
    unittest.main()
