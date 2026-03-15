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
    prompt_encoder_path: str = ""
    scheduler_name: str = ""
    supports_prompt_generation: bool = False
    supports_latent_editing: bool = False
    device: str = "cpu"
    hosted_model_id: str = ""
    inference_provider: str = "huggingface_hub"
    prompt_prefix: str = "afrocentric portrait, studio lighting, high detail"
    negative_prompt: str = ""
    guidance_scale: float = 4.0
    num_inference_steps: int = 4
    inference_timeout_seconds: float = 45.0
    output_width: int = 512
    output_height: int = 512

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

    if not artifact.supports_prompt_generation:
        return "incomplete_inference_contract", "Artifact metadata does not declare prompt-generation support yet."

    if not artifact.scheduler_name.strip():
        return "incomplete_inference_contract", "Artifact metadata exists, but no scheduler name has been declared."

    if artifact.hosted_model_id.strip():
        return "ready_for_hosted_inference", "Hosted model metadata is available for hybrid inference."

    checkpoint = artifact.checkpoint_path.strip()
    if not checkpoint:
        return "metadata_only", "Artifact metadata exists, but no checkpoint path or hosted model has been declared yet."

    checkpoint_path = resolve_artifact_reference(checkpoint, project_root)
    if not checkpoint_path.exists():
        return "checkpoint_missing", f"Artifact metadata points to a checkpoint that is not available: {checkpoint}"

    if artifact.supports_latent_editing and artifact.latent_editor_path.strip():
        latent_editor_path = resolve_artifact_reference(artifact.latent_editor_path, project_root)
        if not latent_editor_path.exists():
            return "latent_editor_missing", (
                "Artifact declares latent editing support, but the referenced latent editor asset is not available: "
                f"{artifact.latent_editor_path}"
            )

    return "ready_for_inference", "Artifact metadata and checkpoint path are available for inference integration."
