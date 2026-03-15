from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from afrogen.data import load_manifest
from afrogen.training.strategy import PHASE3_DATASET_SLICE, PHASE3_MODEL_STRATEGY


@dataclass(frozen=True)
class TrainingPlan:
    manifest_path: str
    train_records: int
    validation_records: int
    test_records: int
    dataset_slice_name: str
    model_strategy: str
    baseline_model_family: str
    recommended_backend: str
    status: str


def build_training_plan(manifest_path: Path) -> TrainingPlan:
    records = load_manifest(manifest_path)
    counts = {"train": 0, "validation": 0, "test": 0}
    for record in records:
        counts[record.split] = counts.get(record.split, 0) + 1

    if counts["train"] == 0:
        status = "blocked: add curated training images and rebuild the manifest"
    elif counts["train"] < PHASE3_DATASET_SLICE.minimum_train_records:
        status = "warning: below phase 3 minimum train size for the first serious baseline"
    elif counts["validation"] < PHASE3_DATASET_SLICE.minimum_validation_records:
        status = "warning: add validation data before full training"
    else:
        status = "ready for a first training baseline"

    return TrainingPlan(
        manifest_path=str(manifest_path),
        train_records=counts["train"],
        validation_records=counts["validation"],
        test_records=counts["test"],
        dataset_slice_name=PHASE3_DATASET_SLICE.name,
        model_strategy=PHASE3_MODEL_STRATEGY.name,
        baseline_model_family=PHASE3_MODEL_STRATEGY.baseline_model_family,
        recommended_backend=PHASE3_MODEL_STRATEGY.target_backend,
        status=status,
    )
