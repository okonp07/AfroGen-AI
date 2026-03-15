"""Training scaffolding for future model work."""

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
    "PHASE3_DATASET_SLICE",
    "PHASE3_MODEL_STRATEGY",
    "TrainingPlan",
    "TrainingRunPlan",
    "build_training_stub",
    "build_training_plan",
    "save_training_stub",
]
