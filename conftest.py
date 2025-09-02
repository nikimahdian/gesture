# conftest.py
import sys
from pathlib import Path

# Ensure the repository root (which contains `src/`) is on sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
