"""Submission validation helper (thin re-export).

The canonical implementation lives in ``scripts/validate_submission.py``.
This shim allows ``python -m agentic_forecaster validate-submission``.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    script = Path(__file__).resolve().parent.parent / "scripts" / "validate_submission.py"
    sys.argv = [str(script)] + sys.argv[1:]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
