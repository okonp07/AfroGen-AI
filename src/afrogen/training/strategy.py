from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSliceSpec:
    name: str
    minimum_train_records: int
    minimum_validation_records: int
    minimum_test_records: int
    priority_sources: tuple[str, ...]
    target_attributes: tuple[str, ...]


@dataclass(frozen=True)
class ModelStrategy:
    name: str
    target_backend: str
    baseline_model_family: str
    rationale: str
    training_stages: tuple[str, ...]


PHASE3_DATASET_SLICE = DatasetSliceSpec(
    name="phase3_research_v1",
    minimum_train_records=5000,
    minimum_validation_records=500,
    minimum_test_records=250,
    priority_sources=("FairFace", "FFHQ", "BUPT-Balancedface"),
    target_attributes=(
        "age_group",
        "gender_presentation",
        "skin_tone",
        "hairstyle",
        "expression",
        "accessory",
        "lighting",
        "background",
    ),
)


PHASE3_MODEL_STRATEGY = ModelStrategy(
    name="latent-diffusion-plus-editor",
    target_backend="hybrid",
    baseline_model_family="sdxl-lora-plus-latent-editor",
    rationale=(
        "Use diffusion for prompt fidelity and a low-dimensional editable latent controller "
        "for the matrix-based real-time editing experience."
    ),
    training_stages=(
        "Domain adaptation on afrocentric portrait prompts",
        "Latent editor learning for controllable matrix steering",
        "Inference optimization for interactive use",
    ),
)
