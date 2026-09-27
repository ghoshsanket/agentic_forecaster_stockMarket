"""CLI entry point for ``package-submission``.

Exports final submission-worthy artefacts from the external runtime
directories (Category B) into the Git repository (Category A).  Never copies
raw data, secrets, caches, environments or temporary files.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("agentic_forecaster.commands.package_submission")


def main(argv: list[str] | None = None) -> int:
    from agentic_forecaster.config import get_env_roots
    from agentic_forecaster.packaging import export_final_artifacts

    roots = get_env_roots()
    copied = export_final_artifacts(roots)
    print(f"package-submission: exported {len(copied)} item(s) into the repository.")
    for item in copied:
        print(f"  + {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
