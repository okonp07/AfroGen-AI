from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


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


def load_backend_artifact(artifact_path: Path) -> BackendArtifact | None:
    if not artifact_path.exists():
        return None
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    return BackendArtifact(**data)


def save_backend_artifact(artifact_path: Path, artifact: BackendArtifact) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact.to_dict(), indent=2), encoding="utf-8")
