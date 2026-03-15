from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from afrogen import load_app_config
from afrogen.backends import load_backend_artifact
from afrogen.backends.factory import create_backend


def main() -> None:
    config = load_app_config()
    app_config = config["app"]
    artifact_path = app_config["trained_backend_artifact"]
    try:
        artifact = load_backend_artifact(artifact_path, PROJECT_ROOT)
        backend = create_backend(
            name="hybrid",
            image_size=app_config["image_size"],
            latent_shape=tuple(app_config["latent_shape"]),
            artifact_path=artifact_path,
        )
    except ModuleNotFoundError as exc:
        print(f"Artifact path: {artifact_path}")
        print(f"Artifact metadata: unavailable ({exc})")
        print("Backend summary: unavailable until required deployment dependencies are installed.")
        return
    print(f"Artifact path: {artifact_path}")
    print(f"Artifact metadata: {artifact}")
    print(f"Backend summary: {backend.summary()}")


if __name__ == "__main__":
    main()
