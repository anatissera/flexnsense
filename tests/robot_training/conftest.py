"""robot_training scripts use flat imports (they are run from their own folder)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "robot_training"
for p in (ROOT, ROOT / "inference"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
