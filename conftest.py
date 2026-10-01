"""Make the project root importable so tests can ``from src.rag... import ...``.

pytest prepends the directory of a collected conftest to ``sys.path``, so this
file at the project root is what lets the test suite run from any working
directory rather than only from the repo root.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
