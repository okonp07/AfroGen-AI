"""Backend factory and interfaces for AfroGen-AI."""

from .artifacts import BackendArtifact, load_backend_artifact, save_backend_artifact
from .factory import create_backend

__all__ = [
    "BackendArtifact",
    "create_backend",
    "load_backend_artifact",
    "save_backend_artifact",
]
