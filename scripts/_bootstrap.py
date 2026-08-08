"""Ensure repo ``src/`` is on ``sys.path`` when running scripts directly.

Usage (at top of each script, after ``from __future__`` if present):

    import _bootstrap  # noqa: F401
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
