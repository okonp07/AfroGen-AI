from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from afrogen import load_app_config
from afrogen.training import export_model_repo_bundle


def main() -> None:
    config = load_app_config()
    training_config = config["training"]
    bundle = export_model_repo_bundle(
        project_root=PROJECT_ROOT,
        artifact_path=Path(training_config["artifact_stub_path"]),
        run_plan_path=Path(training_config["run_plan_path"]),
        model_repo_readme_path=Path("models/model_repo_README.md"),
        output_dir=Path("outputs/model_repo_bundle"),
    )
    print(bundle)


if __name__ == "__main__":
    main()
