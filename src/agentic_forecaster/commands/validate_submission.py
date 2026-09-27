from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = argv or []
    repo_root = args[0] if args else None

    script_path = Path(__file__).resolve().parents[3] / "scripts" / "validate_submission.py"
    cmd = [sys.executable, str(script_path)]
    if repo_root:
        cmd.extend(["--repo-root", repo_root])

    result = subprocess.run(cmd, check=False)
    return result.returncode
