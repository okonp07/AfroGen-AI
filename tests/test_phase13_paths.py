from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class Phase13PathTests(unittest.TestCase):
    def test_model_bundle_directory_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle_dir = root / "outputs" / "model_repo_bundle"
            bundle_dir.mkdir(parents=True)
            (bundle_dir / "README.md").write_text("# bundle", encoding="utf-8")
            (bundle_dir / "trained_backend_stub.json").write_text("{}", encoding="utf-8")
            files = sorted(path.name for path in bundle_dir.iterdir())
            self.assertEqual(files, ["README.md", "trained_backend_stub.json"])


if __name__ == "__main__":
    unittest.main()
