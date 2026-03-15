from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from publish_hf_bundle import _resolve_bundle_dir


class PublishHfBundleTests(unittest.TestCase):
    def test_resolve_bundle_dir_prefers_checkpoint_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_bundle = root / "outputs" / "checkpoint_metadata_bundle"
            model_bundle = root / "outputs" / "model_repo_bundle"
            checkpoint_bundle.mkdir(parents=True)
            model_bundle.mkdir(parents=True)
            self.assertEqual(_resolve_bundle_dir(root, None), checkpoint_bundle)

    def test_resolve_bundle_dir_falls_back_to_model_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_bundle = root / "outputs" / "model_repo_bundle"
            model_bundle.mkdir(parents=True)
            self.assertEqual(_resolve_bundle_dir(root, None), model_bundle)

    def test_resolve_bundle_dir_uses_explicit_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            explicit = root / "custom" / "bundle"
            explicit.mkdir(parents=True)
            self.assertEqual(_resolve_bundle_dir(root, "custom/bundle"), explicit)


if __name__ == "__main__":
    unittest.main()
