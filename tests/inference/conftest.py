import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

pytest_plugins = ["tests.functional.conftest"]
