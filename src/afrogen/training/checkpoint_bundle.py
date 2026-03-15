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
) -> CheckpointMetadataBundle:
    full_output_dir = project_root / output_dir
    full_output_dir.mkdir(parents=True, exist_ok=True)

    template = json.loads((project_root / template_path).read_text(encoding="utf-8"))
    artifact = BackendArtifact(**template)

    artifact_path = full_output_dir / "trained_backend_ready.json"
    readme_path = full_output_dir / "README.md"

    save_backend_artifact(artifact_path, artifact)
    shutil.copy2(project_root / model_repo_readme_path, readme_path)

    return CheckpointMetadataBundle(
        output_dir=str(full_output_dir),
        artifact_path=str(artifact_path),
        readme_path=str(readme_path),
    )
