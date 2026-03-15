from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from afrogen import load_app_config
from afrogen.data import load_manifest


def main() -> None:
    config = load_app_config()
    manifest_path = PROJECT_ROOT / config["dataset"]["manifest_path"]
    records = load_manifest(manifest_path)
    print(f"Loaded {len(records)} manifest rows from {manifest_path}")
    for record in records[:5]:
        print(record.to_dict())


if __name__ == "__main__":
    main()
