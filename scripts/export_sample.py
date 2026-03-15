from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from afrogen.generation import AfroGenPipeline


def main() -> None:
    pipeline = AfroGenPipeline()
    result = pipeline.generate(
        prompt="A smiling Black man with an afro hairstyle and glasses in a studio portrait",
        seed=11,
    )
    output_path = PROJECT_ROOT / "outputs" / "sample_portrait.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
