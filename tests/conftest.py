# Lets tests import `src` when pytest is run from the repo root.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
