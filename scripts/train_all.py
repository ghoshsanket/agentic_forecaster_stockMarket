#!/usr/bin/env python3
"""Train every selected NIFTY-50 stock independently (AUTHOR-CONFIRMED design).

Each ticker gets its own model, its own StandardScaler and its own
temperature calibration.  There is no pooled model.

Usage:
    python scripts/train_all.py --config configs/paper.yaml
    python scripts/train_all.py --config configs/paper.yaml --tickers RELIANCE,TCS
    python scripts/train_all.py --config configs/paper.yaml --baselines

Bundles are written to ``$AGENTIC_MODEL_ROOT/trained/<ticker>/<fold>/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agentic_forecaster.agents.model_agent import ModelAgent
from agentic_forecaster.config import get_env_roots, load_config
from agentic_forecaster.data.agent import DataAgent
from agentic_forecaster.models.checkpoint import save_model_bundle
from agentic_forecaster.utils import atomic_json_dump, seed_everything, setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--tickers", default=None, help="Comma-separated ticker subset")
    parser.add_argument("--device", default=None, help="Device: auto, cpu, cuda, cuda:0")
    parser.add_argument("--fold", default="fold_0")
    parser.add_argument("--baselines", action="store_true", help="Also train baselines")
    args = parser.parse_args()
    setup_logging()

    config = load_config(args.config)
    seed_everything(int(config.get("experiment", {}).get("seed", 42)))
    roots = get_env_roots()

    data_agent = DataAgent(config)
    model_agent = ModelAgent(config)
    universe = data_agent.universe()

    if args.tickers:
        requested = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        requested = list(universe.requested)

    total = len(requested)
    summary: dict = {}
    trained = 0

    for i, symbol in enumerate(requested, 1):
        dataset_symbol = universe.available.get(symbol)
        if dataset_symbol is None:
            reason = universe.unavailable.get(symbol, "not found in dataset")
            summary[symbol] = {"status": "unavailable", "reason": reason}
            print(f"[{i}/{total}] {symbol}: unavailable — {reason}")
            continue
        try:
            ds = data_agent.run(ticker=dataset_symbol)
            fitted = model_agent.train_ticker(
                symbol, ds, fold=args.fold, device=args.device
            )
            if args.baselines:
                model_agent.train_baselines(
                    symbol, ds, fold=args.fold, device=args.device
                )
            bundle = save_model_bundle(
                fitted,
                Path(roots["AGENTIC_MODEL_ROOT"]) / "trained" / symbol / args.fold,
            )
            trained += 1
            summary[symbol] = {
                "status": "trained",
                "dataset_symbol": dataset_symbol,
                "temperature": fitted.temperature,
                "metrics": fitted.metrics,
                "bundle": str(bundle),
            }
            print(f"[{i}/{total}] {symbol}: acc={fitted.metrics.get('accuracy'):.4f} "
                  f"T={fitted.temperature:.4f} -> {bundle}")
        except Exception as exc:
            summary[symbol] = {"status": "unavailable", "reason": str(exc)}
            print(f"[{i}/{total}] {symbol}: unavailable — {exc}")

    atomic_json_dump(summary,
                     Path(roots["AGENTIC_MODEL_ROOT"]) / "train_all_summary.json")
    print(f"\nTrained {trained}/{total} requested ticker(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
