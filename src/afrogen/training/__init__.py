"""Training scaffolding for future model work."""

from .plan import TrainingPlan, build_training_plan
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
    "build_training_plan",
]
