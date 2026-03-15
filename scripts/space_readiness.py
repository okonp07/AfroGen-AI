from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from afrogen import load_app_config


def main() -> None:
    config = load_app_config()
    app_config = config["app"]
    inference_config = config["inference"]

    print("AfroGen Spaces readiness")
    print(f"Backend: {app_config['backend']}")
    print(f"Artifact reference: {app_config['trained_backend_artifact']}")
    print(f"Default prompt: {app_config['default_prompt']}")
    print(f"Device: {inference_config['default_device']}")
    print("Suggested Space repo: okonp07/AfroGen-AI")
    print("Suggested model repo: okonp07/afrogen-models")
    print("Recommended env vars:")
    print("  AFROGEN_BACKEND=synthetic")
    print("  AFROGEN_ARTIFACT_PATH=hf://okonp07/afrogen-models/trained_backend_stub.json")
    print("Optional secret:")
    print("  HF_TOKEN=<private-model-repo-token>")


if __name__ == "__main__":
    main()
