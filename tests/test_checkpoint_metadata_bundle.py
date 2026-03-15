from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from afrogen.training import export_checkpoint_metadata_bundle


class CheckpointMetadataBundleTests(unittest.TestCase):
    def test_export_checkpoint_metadata_bundle_creates_ready_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "models").mkdir()
            (root / "outputs").mkdir()
            (root / "models" / "hybrid_backend_artifact_template.json").write_text(
                """{
  "backend_name": "hybrid",
  "model_strategy": "latent-diffusion-plus-editor",
  "baseline_model_family": "sdxl-lora-plus-latent-editor",
  "status": "ready",
  "message": "ready",
  "checkpoint_path": "checkpoints/model.safetensors",
  "scheduler_name": "EulerDiscreteScheduler",
  "supports_prompt_generation": true,
  "supports_latent_editing": false,
  "device": "cpu"
}""",
                encoding="utf-8",
            )
            (root / "models" / "model_repo_README.md").write_text("# readme", encoding="utf-8")

            bundle = export_checkpoint_metadata_bundle(
                project_root=root,
                template_path=Path("models/hybrid_backend_artifact_template.json"),
                model_repo_readme_path=Path("models/model_repo_README.md"),
                output_dir=Path("outputs/checkpoint_metadata_bundle"),
            )
            self.assertTrue(Path(bundle.artifact_path).exists())
            self.assertTrue(Path(bundle.readme_path).exists())


if __name__ == "__main__":
    unittest.main()
