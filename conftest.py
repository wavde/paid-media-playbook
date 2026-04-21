"""Make `src/` importable as top-level packages under pytest, matching sibling repos."""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(_ROOT / "src"))
