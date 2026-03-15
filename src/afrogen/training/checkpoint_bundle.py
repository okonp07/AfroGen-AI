from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from afrogen.backends import BackendArtifact, save_backend_artifact


@dataclass(frozen=True)
class CheckpointMetadataBundle:
    output_dir: str
    artifact_path: str
    readme_path: str


def export_checkpoint_metadata_bundle(
    project_root: Path,
    template_path: Path,
    model_repo_readme_path: Path,
    output_dir: Path,
    hosted_model_id: str = "",
    prompt_prefix: str = "",
    negative_prompt: str = "",
    guidance_scale: float | None = None,
    num_inference_steps: int | None = None,
    inference_timeout_seconds: float | None = None,
    output_width: int | None = None,
    output_height: int | None = None,
) -> CheckpointMetadataBundle:
    full_output_dir = project_root / output_dir
    full_output_dir.mkdir(parents=True, exist_ok=True)

    template = json.loads((project_root / template_path).read_text(encoding="utf-8"))
    if hosted_model_id:
        template["hosted_model_id"] = hosted_model_id
    if prompt_prefix:
        template["prompt_prefix"] = prompt_prefix
    if negative_prompt:
        template["negative_prompt"] = negative_prompt
    if guidance_scale is not None:
        template["guidance_scale"] = guidance_scale
    if num_inference_steps is not None:
        template["num_inference_steps"] = num_inference_steps
    if inference_timeout_seconds is not None:
        template["inference_timeout_seconds"] = inference_timeout_seconds
    if output_width is not None:
        template["output_width"] = output_width
    if output_height is not None:
        template["output_height"] = output_height
    artifact = BackendArtifact(**template)

    artifact_path = full_output_dir / "trained_backend_stub.json"
    readme_path = full_output_dir / "README.md"

    save_backend_artifact(artifact_path, artifact)
    shutil.copy2(project_root / model_repo_readme_path, readme_path)

    return CheckpointMetadataBundle(
        output_dir=str(full_output_dir),
        artifact_path=str(artifact_path),
        readme_path=str(readme_path),
    )
