from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from afrogen.training import PHASE3_DATASET_SLICE, PHASE3_MODEL_STRATEGY


def main() -> None:
    print("Phase 3 dataset slice:")
    print(PHASE3_DATASET_SLICE)
    print()
    print("Phase 3 model strategy:")
    print(PHASE3_MODEL_STRATEGY)


if __name__ == "__main__":
    main()
