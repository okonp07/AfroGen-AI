from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from afrogen.config import apply_env_overrides


class ConfigOverrideTests(unittest.TestCase):
    def test_env_overrides_replace_backend_and_artifact(self) -> None:
        config = {
            "app": {
                "backend": "synthetic",
                "trained_backend_artifact": "models/local.json",
                "default_prompt": "base prompt",
            },
            "inference": {"default_device": "cpu"},
        }
        with patch.dict(
            os.environ,
            {
                "AFROGEN_BACKEND": "hybrid",
                "AFROGEN_ARTIFACT_PATH": "hf://okonp007/afrogen-models/trained_backend_stub.json",
                "AFROGEN_DEFAULT_PROMPT": "cloud prompt",
                "AFROGEN_DEVICE": "cuda",
            },
            clear=False,
        ):
            updated = apply_env_overrides(config)
        self.assertEqual(updated["app"]["backend"], "hybrid")
        self.assertEqual(
            updated["app"]["trained_backend_artifact"],
            "hf://okonp007/afrogen-models/trained_backend_stub.json",
        )
        self.assertEqual(updated["app"]["default_prompt"], "cloud prompt")
        self.assertEqual(updated["inference"]["default_device"], "cuda")


if __name__ == "__main__":
    unittest.main()
