"""Paper walk-forward evaluation (PAPER-DEFINED folds).

FOLD 0:
    train      2016-01-01 -> 2020-12-31
    validation 2021-01-01 -> 2021-12-31
    test       2022-01-01 -> 2022-12-31

FOLD 1:
    train      2016-01-01 -> 2021-12-31
    validation 2022-01-01 -> 2022-12-31
    test       2023-01-01 -> 2023-12-31

Retrains EVERY ticker between folds.  Uses real data only.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

from agentic_forecaster.agents.model_agent import ModelAgent
from agentic_forecaster.data.agent import DataAgent
from agentic_forecaster.models.checkpoint import save_model_bundle
from agentic_forecaster.utils import atomic_json_dump, ensure_dir, seed_everything

logger = logging.getLogger("agentic_forecaster.orchestration.walk_forward")

PAPER_FOLDS = [
    {
        "fold": "fold_0",
        "train_start": "2016-01-01", "train_end": "2020-12-31",
        "val_start": "2021-01-01", "val_end": "2021-12-31",
        "test_start": "2022-01-01", "test_end": "2022-12-31",
    },
    {
        "fold": "fold_1",
        "train_start": "2016-01-01", "train_end": "2021-12-31",
        "val_start": "2022-01-01", "val_end": "2022-12-31",
        "test_start": "2023-01-01", "test_end": "2023-12-31",
    },
]


def run_walk_forward(config: dict, device: str | None = None) -> dict:
    """Execute the paper walk-forward: retrain every ticker for each fold."""
    seed_everything(int(config.get("experiment", {}).get("seed", 42)))
    data_agent = DataAgent(config)
    model_agent = ModelAgent(config)

    from agentic_forecaster.config import get_env_roots
    roots = get_env_roots()
    model_root = ensure_dir(Path(roots["AGENTIC_MODEL_ROOT"]))

    ticker_files = data_agent._discover()
    tickers = sorted(ticker_files.keys())

    results = {}
    for fold_info in PAPER_FOLDS:
        fold_name = fold_info["fold"]
        logger.info("=== %s ===", fold_name)

        fold_config = copy.deepcopy(config)
        fold_config["data"].update({
            "train_start": fold_info["train_start"],
            "train_end": fold_info["train_end"],
            "val_start": fold_info["val_start"],
            "val_end": fold_info["val_end"],
            "test_start": fold_info["test_start"],
            "test_end": fold_info["test_end"],
        })

        fold_agent = DataAgent(fold_config)
        fold_results = {}

        for idx, ticker in enumerate(tickers, 1):
            logger.info("[%d/%d] %s", idx, len(tickers), ticker)
            try:
                dataset = fold_agent.run(ticker=ticker)
                fitted = model_agent.train_ticker(ticker, dataset, fold=fold_name, device=device)
                bundle_dir = save_model_bundle(
                    fitted, model_root / "trained" / ticker / fold_name
                )
                fold_results[ticker] = {
                    "status": "trained",
                    "metrics": fitted.metrics,
                    "bundle": str(bundle_dir),
                }
            except Exception as exc:
                fold_results[ticker] = {"status": "unavailable", "reason": str(exc)}
                logger.warning("  %s unavailable: %s", ticker, exc)

        results[fold_name] = fold_results

    atomic_json_dump(results, model_root / "walk_forward_results.json")
    logger.info("Walk-forward complete. Results: %s", model_root / "walk_forward_results.json")
    return results
