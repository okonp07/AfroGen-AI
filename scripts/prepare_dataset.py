from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from afrogen import load_app_config
from afrogen.data import build_manifest


def main() -> None:
    config = load_app_config()
    dataset_config = config["dataset"]
    manifest_path, records = build_manifest(
        project_root=PROJECT_ROOT,
        raw_dir=Path(dataset_config["raw_dir"]),
        processed_manifest=Path(dataset_config["manifest_path"]),
        allowed_extensions=tuple(dataset_config["image_extensions"]),
    )
    print(f"Manifest written to: {manifest_path}")
    print(f"Images indexed: {len(records)}")
    split_counts: dict[str, int] = {}
    for record in records:
        split_counts[record.split] = split_counts.get(record.split, 0) + 1
    print(f"Splits: {split_counts}")


if __name__ == "__main__":
    main()
