from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class SliceBatch:
    slice_name: str
    batch_name: str
    source_dataset: str
    relative_dir: str
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def load_slice_registry(registry_path: Path) -> list[SliceBatch]:
    if not registry_path.exists():
        return []
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    return [SliceBatch(**item) for item in data]


def save_slice_registry(registry_path: Path, batches: list[SliceBatch]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps([batch.to_dict() for batch in batches], indent=2),
        encoding="utf-8",
    )


def upsert_slice_batch(registry_path: Path, batch: SliceBatch) -> list[SliceBatch]:
    batches = load_slice_registry(registry_path)
    updated = False
    for index, existing in enumerate(batches):
        if existing.batch_name == batch.batch_name and existing.slice_name == batch.slice_name:
            batches[index] = batch
            updated = True
            break
    if not updated:
        batches.append(batch)
    save_slice_registry(registry_path, batches)
    return batches
