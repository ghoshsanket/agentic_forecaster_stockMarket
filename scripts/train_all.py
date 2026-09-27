#!/usr/bin/env python3
"""Train all 50 models (one per ticker) using the paper configuration.

Usage:
    python scripts/train_all.py --config configs/paper.yaml

Checkpoints are written under ``$AGENTIC_MODEL_ROOT`` (Category B, outside
Git).  Use ``scripts/package_submission.py`` to export final artefacts into
the repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agentic_forecaster.config import get_env_roots, load_config
from agentic_forecaster.data import DataAgent
from agentic_forecaster.models import AttentionLSTM
from agentic_forecaster.training import Trainer
from agentic_forecaster.utils import atomic_json_dump, ensure_dir, seed_everything, setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--tickers", default=None, help="Comma-separated ticker subset")
    args = parser.parse_args()
    setup_logging()

    config = load_config(args.config)
    seed_everything(int(config.get("experiment", {}).get("seed", 42)))
    roots = get_env_roots()

    data_agent = DataAgent(config)
    dataset = data_agent.run()

    tickers = args.tickers.split(",") if args.tickers else None
    model_root = ensure_dir(Path(roots["AGENTIC_MODEL_ROOT"]))

    summary = {}
    for ticker in (tickers or ["ALL"]):
        run_dir = ensure_dir(model_root / ticker)
        cfg = config["models"]["attention_lstm"]
        model = AttentionLSTM(
            input_size=dataset.train.X.shape[-1],
            hidden_size=int(cfg.get("hidden_size", 64)),
            num_layers=int(cfg.get("num_layers", 2)),
            dropout=float(cfg.get("dropout", 0.2)),
        )
        trainer = Trainer(
            model,
            learning_rate=float(cfg.get("learning_rate", 1e-3)),
            weight_decay=float(cfg.get("weight_decay", 1e-4)),
            batch_size=int(cfg.get("batch_size", 64)),
            epochs=int(cfg.get("epochs", 100)),
            patience=int(cfg.get("patience", 15)),
            seed=int(config.get("experiment", {}).get("seed", 42)),
        )
        result = trainer.fit(
            dataset.train.X, dataset.train.y,
            dataset.val.X, dataset.val.y,
            checkpoint_path=str(run_dir / "model.pt"),
        )
        summary[ticker] = {
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "checkpoint": str(run_dir / "model.pt"),
        }
        print(f"[{ticker}] best_epoch={result.best_epoch} val_loss={result.best_val_loss:.4f}")

    atomic_json_dump(summary, model_root / "train_all_summary.json")
    print("All models trained. Summary:", model_root / "train_all_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
