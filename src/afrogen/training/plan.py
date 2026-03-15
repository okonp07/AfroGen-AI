from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from afrogen.data import load_manifest


@dataclass(frozen=True)
class TrainingPlan:
    manifest_path: str
    train_records: int
    validation_records: int
    test_records: int
    recommended_backend: str
    status: str


def build_training_plan(manifest_path: Path) -> TrainingPlan:
    records = load_manifest(manifest_path)
    counts = {"train": 0, "validation": 0, "test": 0}
    for record in records:
        counts[record.split] = counts.get(record.split, 0) + 1

    if counts["train"] == 0:
        status = "blocked: add curated training images and rebuild the manifest"
    elif counts["validation"] == 0:
        status = "warning: add validation data before full training"
    else:
        status = "ready for a first training baseline"

    return TrainingPlan(
        manifest_path=str(manifest_path),
        train_records=counts["train"],
        validation_records=counts["validation"],
        test_records=counts["test"],
        recommended_backend="hybrid",
        status=status,
    )
