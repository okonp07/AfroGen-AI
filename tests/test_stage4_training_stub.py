from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from afrogen.data import SliceBatch, build_manifest, upsert_slice_batch
from afrogen.training import build_training_stub, save_training_stub


class Stage4Tests(unittest.TestCase):
    def test_slice_registry_upsert(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = Path(temp_dir) / "slice_registry.json"
            batches = upsert_slice_batch(
                registry_path,
                SliceBatch(
                    slice_name="phase3_research_v1",
                    batch_name="fairface_batch_001",
                    source_dataset="FairFace",
                    relative_dir="data/raw/batches/fairface_batch_001",
                ),
            )
            self.assertEqual(len(batches), 1)
            self.assertTrue(registry_path.exists())

    def test_training_stub_writes_plan_and_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "data" / "raw" / "batches" / "fairface_batch_001"
            raw_dir.mkdir(parents=True)
            (raw_dir / "sample_train.png").write_bytes(b"fake")

            manifest_path, _ = build_manifest(
                project_root=root,
                raw_dir=Path("data/raw"),
                processed_manifest=Path("data/processed/manifest.jsonl"),
                allowed_extensions=(".png",),
            )

            artifact_path = root / "models" / "trained_backend_stub.json"
            run_plan = build_training_stub(manifest_path, artifact_path)
            run_plan_path = root / "models" / "training_run_plan.json"
            save_training_stub(run_plan_path, artifact_path, run_plan)

            self.assertTrue(run_plan_path.exists())
            self.assertTrue(artifact_path.exists())
            self.assertEqual(run_plan.backend, "hybrid")
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(artifact["status"], "stub")


if __name__ == "__main__":
    unittest.main()
