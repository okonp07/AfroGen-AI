from __future__ import annotations

from pathlib import Path

from .synthetic import SyntheticAfroGenBackend
from .trained import TrainedAfroGenBackend


def create_backend(name: str, image_size: int, latent_shape: tuple[int, int], artifact_path: str | Path | None = None):
    normalized = name.strip().lower()
    if normalized == "synthetic":
        return SyntheticAfroGenBackend(image_size=image_size, latent_shape=latent_shape)
    if normalized in {"trained", "diffusion", "hybrid"}:
        return TrainedAfroGenBackend(
            name=normalized,
            image_size=image_size,
            latent_shape=latent_shape,
            artifact_path=Path(artifact_path) if artifact_path else None,
        )
    raise ValueError(f"Unsupported backend '{name}'")
