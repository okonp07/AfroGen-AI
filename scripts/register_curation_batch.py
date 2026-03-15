from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from afrogen import load_app_config
from afrogen.data import SliceBatch, upsert_slice_batch


def main() -> None:
    config = load_app_config()
    dataset_config = config["dataset"]
    training_config = config["training"]

    batch = SliceBatch(
        slice_name=training_config["dataset_slice_name"],
        batch_name="fairface_batch_001",
        source_dataset="FairFace",
        relative_dir=str(Path(dataset_config["curation_batch_dir"]) / "fairface_batch_001"),
        notes="Starter research batch registration for phase 4.",
    )
    registry_path = PROJECT_ROOT / dataset_config["slice_registry_path"]
    batches = upsert_slice_batch(registry_path, batch)
    print(f"Registry updated: {registry_path}")
    print(f"Registered batches: {len(batches)}")
    for item in batches:
        print(item)


if __name__ == "__main__":
    main()
