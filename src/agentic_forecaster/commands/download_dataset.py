"""CLI entry point for dataset download."""

from __future__ import annotations

import logging

logger = logging.getLogger("agentic_forecaster.commands.download_dataset")


def main(argv: list[str] | None = None) -> int:
    from agentic_forecaster.config import get_env_roots
    from agentic_forecaster.data.downloader import download_dataset

    roots = get_env_roots()
    raw_root = argv[0] if argv else roots["AGENTIC_RAW_DATA_ROOT"]
    download_dataset(raw_root=raw_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
