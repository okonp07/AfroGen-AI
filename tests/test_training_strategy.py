from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from afrogen.data import build_manifest
from afrogen.training import PHASE3_DATASET_SLICE, PHASE3_MODEL_STRATEGY, build_training_plan


class TrainingStrategyTests(unittest.TestCase):
    def test_phase3_constants_are_set(self) -> None:
        self.assertEqual(PHASE3_DATASET_SLICE.name, "phase3_research_v1")
        self.assertEqual(PHASE3_MODEL_STRATEGY.target_backend, "hybrid")
        self.assertIn("diffusion", PHASE3_MODEL_STRATEGY.name)

    def test_training_plan_carries_phase3_strategy_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "data" / "raw"
            raw_dir.mkdir(parents=True)
            (raw_dir / "sample_train.png").write_bytes(b"fake")

            manifest_path, _ = build_manifest(
                project_root=root,
                raw_dir=Path("data/raw"),
                processed_manifest=Path("data/processed/manifest.jsonl"),
                allowed_extensions=(".png",),
            )
            plan = build_training_plan(manifest_path)
            self.assertEqual(plan.dataset_slice_name, "phase3_research_v1")
            self.assertEqual(plan.recommended_backend, "hybrid")
            self.assertEqual(plan.baseline_model_family, "sdxl-lora-plus-latent-editor")


if __name__ == "__main__":
    unittest.main()
