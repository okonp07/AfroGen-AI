from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


MODEL_REPO_ID = "okonp007/afrogen-models"


def main() -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required to publish the model bundle.")

    bundle_dir = PROJECT_ROOT / "outputs" / "model_repo_bundle"
    if not bundle_dir.exists():
        raise SystemExit("Model repo bundle not found. Run scripts/export_model_repo_bundle.py first.")

    with tempfile.TemporaryDirectory() as temp_dir:
        clone_dir = Path(temp_dir) / "hf-model-repo"
        remote_url = f"https://okonp007:{token}@huggingface.co/{MODEL_REPO_ID}"

        subprocess.run(["git", "clone", remote_url, str(clone_dir)], check=True)

        for path in clone_dir.iterdir():
            if path.name == ".git":
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

        for item in bundle_dir.iterdir():
            target = clone_dir / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)

        subprocess.run(["git", "-C", str(clone_dir), "add", "-A"], check=True)
        diff = subprocess.run(
            ["git", "-C", str(clone_dir), "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        if not diff:
            print("No model bundle changes to publish.")
            return

        subprocess.run(
            ["git", "-C", str(clone_dir), "commit", "-m", "Update AfroGen model bundle"],
            check=True,
        )
        subprocess.run(["git", "-C", str(clone_dir), "push", "origin", "main"], check=True)
        print(f"Published bundle to {MODEL_REPO_ID}")


if __name__ == "__main__":
    main()
