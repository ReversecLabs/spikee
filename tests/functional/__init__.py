"""Functional test package."""

from pathlib import Path
import sys

_project_root = Path(__file__).resolve().parents[2]
_project_root_str = str(_project_root)
if _project_root_str not in sys.path:
    sys.path.insert(0, _project_root_str)
