from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelRepoBundle:
    output_dir: str
    artifact_path: str
    run_plan_path: str
    readme_path: str


def export_model_repo_bundle(
    project_root: Path,
    artifact_path: Path,
    run_plan_path: Path,
    model_repo_readme_path: Path,
    output_dir: Path,
) -> ModelRepoBundle:
    full_output_dir = project_root / output_dir
    full_output_dir.mkdir(parents=True, exist_ok=True)

    copied_artifact = full_output_dir / artifact_path.name
    copied_run_plan = full_output_dir / run_plan_path.name
    copied_readme = full_output_dir / "README.md"

    shutil.copy2(project_root / artifact_path, copied_artifact)
    shutil.copy2(project_root / run_plan_path, copied_run_plan)
    shutil.copy2(project_root / model_repo_readme_path, copied_readme)

    return ModelRepoBundle(
        output_dir=str(full_output_dir),
        artifact_path=str(copied_artifact),
        run_plan_path=str(copied_run_plan),
        readme_path=str(copied_readme),
    )
