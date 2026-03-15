from __future__ import annotations

import os
from pathlib import Path


def _download_from_hub(repo_id: str, filename: str, token: str | None) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "huggingface_hub is required for hf:// artifact references. Install it from requirements.txt."
        ) from exc
    return hf_hub_download(repo_id=repo_id, filename=filename, token=token)


def resolve_artifact_reference(reference: str | Path, project_root: Path, token_env: str = "HF_TOKEN") -> Path:
    ref = str(reference)
    if ref.startswith("hf://"):
        repo_and_path = ref.removeprefix("hf://")
        parts = repo_and_path.split("/", 2)
        if len(parts) < 3:
            raise ValueError("Hugging Face artifact references must look like hf://repo-id/path/to/file.json")
        repo_id = "/".join(parts[:2])
        filename = parts[2]
        token = os.getenv(token_env)
        download_path = _download_from_hub(repo_id=repo_id, filename=filename, token=token)
        return Path(download_path)

    path = Path(ref)
    if path.is_absolute():
        return path
    return project_root / path
