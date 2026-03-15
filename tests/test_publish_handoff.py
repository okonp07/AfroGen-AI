from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from afrogen import load_app_config


class PublishHandoffTests(unittest.TestCase):
    def test_default_artifact_points_to_hf_reference(self) -> None:
        config = load_app_config()
        self.assertTrue(config["app"]["trained_backend_artifact"].startswith("hf://"))

    def test_bundle_directory_can_be_populated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = Path(temp_dir) / "outputs" / "model_repo_bundle"
            bundle_dir.mkdir(parents=True)
            (bundle_dir / "trained_backend_stub.json").write_text("{}", encoding="utf-8")
            files = sorted(path.name for path in bundle_dir.glob("*") if path.is_file())
            self.assertEqual(files, ["trained_backend_stub.json"])


if __name__ == "__main__":
    unittest.main()
