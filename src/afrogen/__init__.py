"""AfroGen-AI package."""

from .config import load_app_config
from .generation.pipeline import GenerationResult

__all__ = ["GenerationResult", "load_app_config"]
