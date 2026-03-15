from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from afrogen.training import export_checkpoint_metadata_bundle


def main() -> None:
    bundle = export_checkpoint_metadata_bundle(
        project_root=PROJECT_ROOT,
        template_path=Path("models/hybrid_backend_artifact_template.json"),
        model_repo_readme_path=Path("models/model_repo_README.md"),
        output_dir=Path("outputs/checkpoint_metadata_bundle"),
    )
    print(bundle)


if __name__ == "__main__":
    main()
