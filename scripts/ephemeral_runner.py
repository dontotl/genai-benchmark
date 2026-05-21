#!/usr/bin/env python3

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from genai_benchmark.ephemeral_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
