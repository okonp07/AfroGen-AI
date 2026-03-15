from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from afrogen import load_app_config
from afrogen.training import build_training_plan


def main() -> None:
    config = load_app_config()
    manifest_path = PROJECT_ROOT / config["training"]["default_manifest_path"]
    plan = build_training_plan(manifest_path)
    print(plan)


if __name__ == "__main__":
    main()
