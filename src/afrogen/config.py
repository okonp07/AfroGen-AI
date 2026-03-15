from __future__ import annotations

import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "app.json"


def load_app_config(config_path: Path | None = None) -> dict:
    path = config_path or DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return apply_env_overrides(config)


def apply_env_overrides(config: dict) -> dict:
    app = config.setdefault("app", {})
    inference = config.setdefault("inference", {})

    if os.getenv("AFROGEN_BACKEND"):
        app["backend"] = os.environ["AFROGEN_BACKEND"]
    if os.getenv("AFROGEN_ARTIFACT_PATH"):
        app["trained_backend_artifact"] = os.environ["AFROGEN_ARTIFACT_PATH"]
    if os.getenv("AFROGEN_DEFAULT_PROMPT"):
        app["default_prompt"] = os.environ["AFROGEN_DEFAULT_PROMPT"]
    if os.getenv("AFROGEN_DEVICE"):
        inference["default_device"] = os.environ["AFROGEN_DEVICE"]

    return config
