from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from afrogen.data import load_manifest
from afrogen.training.plan import build_training_plan
from afrogen.training.strategy import PHASE3_MODEL_STRATEGY


@dataclass(frozen=True)
class TrainingRunPlan:
    dataset_slice_name: str
    manifest_path: str
    train_records: int
    validation_records: int
    test_records: int
    backend: str
    model_strategy: str
    baseline_model_family: str
    output_artifact_path: str
    status: str


def build_training_stub(manifest_path: Path, artifact_path: Path) -> TrainingRunPlan:
    plan = build_training_plan(manifest_path)
    run_plan = TrainingRunPlan(
        dataset_slice_name=plan.dataset_slice_name,
        manifest_path=str(manifest_path),
        train_records=plan.train_records,
        validation_records=plan.validation_records,
        test_records=plan.test_records,
        backend=plan.recommended_backend,
        model_strategy=plan.model_strategy,
        baseline_model_family=plan.baseline_model_family,
        output_artifact_path=str(artifact_path),
        status=plan.status,
    )
    return run_plan


def save_training_stub(run_plan_path: Path, artifact_path: Path, run_plan: TrainingRunPlan) -> None:
    run_plan_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    run_plan_path.write_text(json.dumps(asdict(run_plan), indent=2), encoding="utf-8")
    artifact_path.write_text(
        json.dumps(
            {
                "backend_name": PHASE3_MODEL_STRATEGY.target_backend,
                "model_strategy": PHASE3_MODEL_STRATEGY.name,
                "baseline_model_family": PHASE3_MODEL_STRATEGY.baseline_model_family,
                "status": "stub",
                "message": "Replace this file with real checkpoint metadata once model training is implemented.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
