from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .resolve import resolve_artifact_reference


@dataclass(frozen=True)
class BackendArtifact:
    backend_name: str
    model_strategy: str
    baseline_model_family: str
    status: str
    message: str
    manifest_path: str = ""
    training_run_plan_path: str = ""
    checkpoint_path: str = ""
    latent_editor_path: str = ""
    device: str = "cpu"

    def to_dict(self) -> dict:
        return asdict(self)


def load_backend_artifact(artifact_path: str | Path, project_root: Path | None = None) -> BackendArtifact | None:
    resolved = resolve_artifact_reference(artifact_path, project_root or Path.cwd())
    if not resolved.exists():
        return None
    data = json.loads(resolved.read_text(encoding="utf-8"))
    return BackendArtifact(**data)


def save_backend_artifact(artifact_path: Path, artifact: BackendArtifact) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact.to_dict(), indent=2), encoding="utf-8")


def validate_backend_artifact(artifact: BackendArtifact | None, project_root: Path) -> tuple[str, str]:
    if artifact is None:
        return "missing", "No backend artifact could be resolved."

    if artifact.status == "stub":
        return "stub", artifact.message

    checkpoint = artifact.checkpoint_path.strip()
    if not checkpoint:
        return "metadata_only", "Artifact metadata exists, but no checkpoint path has been declared yet."

    checkpoint_path = resolve_artifact_reference(checkpoint, project_root)
    if not checkpoint_path.exists():
        return "checkpoint_missing", f"Artifact metadata points to a checkpoint that is not available: {checkpoint}"

    return "ready_for_inference", "Artifact metadata and checkpoint path are available for inference integration."
