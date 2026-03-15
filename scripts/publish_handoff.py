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

    bundle_dir = PROJECT_ROOT / "outputs" / "model_repo_bundle"
    bundle_files = sorted(path.name for path in bundle_dir.glob("*") if path.is_file())

    print("AfroGen publish handoff")
    print()
    print("Space repo")
    print("  okonp07/AfroGen-AI")
    print("Model repo")
    print("  okonp07/afrogen-models")
    print()
    print("Space settings")
    print("  SDK: Gradio")
    print("  App file: app.py")
    print()
    print("Recommended first env vars")
    print("  AFROGEN_BACKEND=synthetic")
    print(f"  AFROGEN_ARTIFACT_PATH={app_config['trained_backend_artifact']}")
    print()
    print("Model repo upload bundle")
    print(f"  Directory: {bundle_dir}")
    if bundle_files:
        for file_name in bundle_files:
            print(f"  - {file_name}")
    else:
        print("  - Bundle not found yet. Run: python3 scripts/export_model_repo_bundle.py")
    checkpoint_bundle_dir = PROJECT_ROOT / "outputs" / "checkpoint_metadata_bundle"
    checkpoint_files = sorted(path.name for path in checkpoint_bundle_dir.glob("*") if path.is_file())
    print()
    print("Hybrid rollout bundle")
    print(f"  Directory: {checkpoint_bundle_dir}")
    if checkpoint_files:
        for file_name in checkpoint_files:
            print(f"  - {file_name}")
    else:
        print(
            "  - Bundle not found yet. Run: python3 scripts/export_checkpoint_metadata.py "
            "--hosted-model-id <your-model-id>"
        )
    print()
    print("Optional secret")
    print("  HF_TOKEN=<private-model-repo-token>")


if __name__ == "__main__":
    main()
