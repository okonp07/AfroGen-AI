from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from afrogen.data import build_manifest, load_manifest
from afrogen.training import build_training_plan


class DatasetManifestTests(unittest.TestCase):
    def test_build_manifest_with_metadata_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "data" / "raw"
            raw_dir.mkdir(parents=True)
            (raw_dir / "portrait_train_001.png").write_bytes(b"fake")
            (raw_dir / "metadata.csv").write_text(
                "\n".join(
                    [
                        "file_name,gender_presentation,age_group,skin_tone,hairstyle,expression,accessory,split,tags",
                        "portrait_train_001.png,woman,adult,deep,braids,smile,earrings,train,afrocentric|studio",
                    ]
                ),
                encoding="utf-8",
            )

            manifest_path, records = build_manifest(
                project_root=root,
                raw_dir=Path("data/raw"),
                processed_manifest=Path("data/processed/manifest.jsonl"),
                allowed_extensions=(".png",),
            )

            self.assertTrue(manifest_path.exists())
            self.assertEqual(len(records), 1)
            self.assertIn("braids", records[0].prompt)
            self.assertEqual(records[0].tags, ["afrocentric", "studio"])

    def test_training_plan_reports_missing_validation(self) -> None:
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
            records = load_manifest(manifest_path)
            self.assertEqual(len(records), 1)

            plan = build_training_plan(manifest_path)
            self.assertEqual(plan.train_records, 1)
            self.assertIn("warning", plan.status)


if __name__ == "__main__":
    unittest.main()
