from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from afrogen.training import export_checkpoint_metadata_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a metadata-complete hybrid backend artifact bundle.")
    parser.add_argument("--hosted-model-id", default="", help="Hosted Hugging Face model or endpoint id.")
    parser.add_argument("--prompt-prefix", default="", help="Prompt prefix to prepend during hosted inference.")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt to send during hosted inference.")
    parser.add_argument("--guidance-scale", type=float, default=None, help="Hosted inference guidance scale override.")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Hosted inference step count override.",
    )
    parser.add_argument(
        "--inference-timeout-seconds",
        type=float,
        default=None,
        help="Hosted inference timeout override in seconds.",
    )
    parser.add_argument("--output-width", type=int, default=None, help="Hosted output image width.")
    parser.add_argument("--output-height", type=int, default=None, help="Hosted output image height.")
    args = parser.parse_args()

    bundle = export_checkpoint_metadata_bundle(
        project_root=PROJECT_ROOT,
        template_path=Path("models/hybrid_backend_artifact_template.json"),
        model_repo_readme_path=Path("models/model_repo_README.md"),
        output_dir=Path("outputs/checkpoint_metadata_bundle"),
        hosted_model_id=args.hosted_model_id,
        prompt_prefix=args.prompt_prefix,
        negative_prompt=args.negative_prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        inference_timeout_seconds=args.inference_timeout_seconds,
        output_width=args.output_width,
        output_height=args.output_height,
    )
    print(bundle)


if __name__ == "__main__":
    main()
