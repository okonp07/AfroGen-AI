from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from afrogen.training import export_model_repo_bundle


class ModelRepoBundleTests(unittest.TestCase):
    def test_export_model_repo_bundle_copies_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "models").mkdir()
            (root / "outputs").mkdir()
            artifact = root / "models" / "trained_backend_stub.json"
            run_plan = root / "models" / "training_run_plan.json"
            readme = root / "models" / "model_repo_README.md"
            artifact.write_text("{}", encoding="utf-8")
            run_plan.write_text("{}", encoding="utf-8")
            readme.write_text("# test", encoding="utf-8")

            bundle = export_model_repo_bundle(
                project_root=root,
                artifact_path=Path("models/trained_backend_stub.json"),
                run_plan_path=Path("models/training_run_plan.json"),
                model_repo_readme_path=Path("models/model_repo_README.md"),
                output_dir=Path("outputs/model_repo_bundle"),
            )
            self.assertTrue(Path(bundle.artifact_path).exists())
            self.assertTrue(Path(bundle.run_plan_path).exists())
            self.assertTrue(Path(bundle.readme_path).exists())


if __name__ == "__main__":
    unittest.main()
