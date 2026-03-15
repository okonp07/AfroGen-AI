from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from afrogen.backends.resolve import resolve_artifact_reference


class ArtifactResolutionTests(unittest.TestCase):
    def test_resolves_local_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "models" / "artifact.json"
            target.parent.mkdir(parents=True)
            target.write_text("{}", encoding="utf-8")
            resolved = resolve_artifact_reference("models/artifact.json", root)
            self.assertEqual(resolved, target)

    def test_resolves_hf_reference_via_hub_download(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            downloaded = Path(temp_dir) / "downloaded.json"
            downloaded.write_text("{}", encoding="utf-8")
            with patch("afrogen.backends.resolve._download_from_hub", return_value=str(downloaded)) as mocked:
                resolved = resolve_artifact_reference(
                    "hf://okonp007/afrogen-models/trained_backend_stub.json",
                    Path(temp_dir),
                )
                self.assertEqual(resolved, downloaded)
                mocked.assert_called_once()


if __name__ == "__main__":
    unittest.main()
