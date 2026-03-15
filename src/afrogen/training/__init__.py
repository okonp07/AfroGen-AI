"""Training scaffolding for future model work."""

from .bundle import ModelRepoBundle, export_model_repo_bundle
from .plan import TrainingPlan, build_training_plan
from .run_stub import TrainingRunPlan, build_training_stub, save_training_stub
from .strategy import (
    PHASE3_DATASET_SLICE,
    PHASE3_MODEL_STRATEGY,
    DatasetSliceSpec,
    ModelStrategy,
)

__all__ = [
    "DatasetSliceSpec",
    "ModelStrategy",
    "ModelRepoBundle",
    "PHASE3_DATASET_SLICE",
    "PHASE3_MODEL_STRATEGY",
    "TrainingPlan",
    "TrainingRunPlan",
    "export_model_repo_bundle",
    "build_training_stub",
    "build_training_plan",
    "save_training_stub",
]
