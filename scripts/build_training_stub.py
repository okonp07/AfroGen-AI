from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from afrogen import load_app_config
from afrogen.training import build_training_stub, save_training_stub


def main() -> None:
    config = load_app_config()
    training_config = config["training"]
    manifest_path = PROJECT_ROOT / training_config["default_manifest_path"]
    run_plan_path = PROJECT_ROOT / training_config["run_plan_path"]
    artifact_path = PROJECT_ROOT / training_config["artifact_stub_path"]

    run_plan = build_training_stub(manifest_path=manifest_path, artifact_path=artifact_path)
    save_training_stub(run_plan_path=run_plan_path, artifact_path=artifact_path, run_plan=run_plan)

    print(f"Training run plan written to: {run_plan_path}")
    print(f"Backend artifact stub written to: {artifact_path}")
    print(run_plan)


if __name__ == "__main__":
    main()
