#!/usr/bin/env python3
"""Train all 50 models (one per ticker) using the paper configuration.

Usage:
    python scripts/train_all.py --config configs/paper.yaml
    python scripts/train_all.py --config configs/paper.yaml --tickers RELIANCE,TCS

Each ticker gets its own independent model, scaler, and calibration.
Checkpoints are written under ``$AGENTIC_MODEL_ROOT/trained/<ticker>/fold_0/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agentic_forecaster.agents.model_agent import ModelAgent
from agentic_forecaster.config import get_env_roots, load_config
from agentic_forecaster.data.agent import DataAgent
from agentic_forecaster.models.checkpoint import save_model_bundle
from agentic_forecaster.utils import atomic_json_dump, ensure_dir, seed_everything, setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--tickers", default=None, help="Comma-separated ticker subset")
    parser.add_argument("--device", default=None, help="Device: auto, cpu, cuda, cuda:0")
    args = parser.parse_args()
    setup_logging()

    config = load_config(args.config)
    seed_everything(int(config.get("experiment", {}).get("seed", 42)))
    roots = get_env_roots()

    data_agent = DataAgent(config)
    model_agent = ModelAgent(config)
    model_root = ensure_dir(Path(roots["AGENTIC_MODEL_ROOT"]))

    ticker_files = data_agent._discover()
    if args.tickers:
        requested = [t.strip().upper() for t in args.tickers.split(",")]
        tickers = [t for t in requested if t in ticker_files]
        unavailable = [t for t in requested if t not in ticker_files]
        for t in unavailable:
            print(f"  {t}: unavailable (not found in raw data)")
    else:
        tickers = sorted(ticker_files.keys())

    total = len(tickers)
    summary = {}
    for idx, ticker in enumerate(tickers, 1):
        print(f"[{idx}/{total}] {ticker}")
        try:
            dataset = data_agent.run(ticker=ticker)
            fitted = model_agent.train_ticker(ticker, dataset, device=args.device)
            bundle_dir = save_model_bundle(
                fitted, model_root / "trained" / ticker / "fold_0"
            )
            summary[ticker] = {
                "status": "trained",
                "metrics": fitted.metrics,
                "bundle": str(bundle_dir),
            }
            print(f"  -> {bundle_dir}")
        except Exception as exc:
            summary[ticker] = {"status": "unavailable", "reason": str(exc)}
            print(f"  unavailable: {exc}")

    atomic_json_dump(summary, model_root / "train_all_summary.json")
    trained = sum(1 for v in summary.values() if v["status"] == "trained")
    print(f"\nTrained {trained}/{total} tickers. Summary: {model_root / 'train_all_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
