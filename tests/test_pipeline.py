from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from afrogen.generation import AfroGenPipeline
from afrogen.generation.prompting import parse_prompt


class PipelineTests(unittest.TestCase):
    def test_prompt_parser_extracts_expected_features(self) -> None:
        profile = parse_prompt("A smiling older Black man with locs and glasses")
        self.assertEqual(profile.age_group, "senior")
        self.assertEqual(profile.hairstyle, "locs")
        self.assertEqual(profile.expression, "smile")
        self.assertEqual(profile.accessory, "glasses")

    def test_pipeline_is_deterministic_without_delta(self) -> None:
        pipeline = AfroGenPipeline()
        first = pipeline.generate("A Black woman with braids", seed=9)
        second = pipeline.generate("A Black woman with braids", seed=9)
        self.assertTrue(np.allclose(first.base_latent, second.base_latent))

    def test_pipeline_applies_delta(self) -> None:
        pipeline = AfroGenPipeline()
        delta = np.ones((4, 4), dtype=np.float32) * 0.25
        result = pipeline.generate("A calm Black child with curly hair", seed=4, delta=delta)
        self.assertEqual(result.edited_latent.shape, (4, 4))
        self.assertTrue(np.all(result.edited_latent >= -1.0))
        self.assertTrue(np.all(result.edited_latent <= 1.0))


if __name__ == "__main__":
    unittest.main()
