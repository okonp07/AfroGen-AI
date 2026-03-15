from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import urllib.request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from afrogen import load_app_config


def main() -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required to inspect the live Space.")

    config = load_app_config()
    artifact_reference = config["app"]["trained_backend_artifact"]
    runtime = {}
    normalized_variables = {}
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        space_runtime = api.get_space_runtime("okonp007/AfroGen-AI")
        hardware = space_runtime.hardware
        if isinstance(hardware, str):
            requested_hardware = hardware
        else:
            requested_hardware = hardware.requested if hardware else None
        runtime = {
            "stage": space_runtime.stage,
            "hardware": {
                "requested": requested_hardware,
            },
        }
        variables = api.get_space_variables("okonp007/AfroGen-AI")
        normalized_variables = {key: value.value for key, value in variables.items()}
    except ModuleNotFoundError:
        headers = {"Authorization": f"Bearer {token}"}
        req = urllib.request.Request(
            "https://huggingface.co/api/spaces/okonp007/AfroGen-AI",
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
        runtime = data.get("runtime", {})
        variables = data.get("variables", [])
        for item in variables:
            key = item.get("key")
            value = item.get("value")
            if key:
                normalized_variables[key] = value

    summary = {
        "space_repo": "okonp007/AfroGen-AI",
        "stage": runtime.get("stage"),
        "requested_hardware": (runtime.get("hardware") or {}).get("requested"),
        "artifact_reference": artifact_reference,
        "variables": normalized_variables,
    }
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
