#!/usr/bin/env python3
"""Export final submission-worthy artefacts into the Git repository.

Usage:
    python scripts/package_submission.py

This command does NOT train models.  It copies final metrics, figures,
representative reports and manifests from the external runtime directories
(Category B) into the repository (Category A).  It never copies the raw
dataset, secrets, caches, environments or temporary files, and prints
exactly what was copied.
"""

from __future__ import annotations

from agentic_forecaster.commands.package_submission import main

if __name__ == "__main__":
    raise SystemExit(main())
