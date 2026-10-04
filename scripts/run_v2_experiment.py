#!/usr/bin/env python3
"""Run ONE V2 experiment: one shared-model fit on one fold with one seed.

    uv run python scripts/run_v2_experiment.py \
        --config configs/v2/v2_c_contextual.yaml \
        --fold V2_DEV_FOLD_A --seed 42 --device auto

FIREWALLS
---------
* any target date >= 2022-01-01 raises :class:`V2TestFirewallError`;
* ``--fold V2_LOCKBOX`` requires ``V2_LOCKBOX=1`` in the environment, which only
  ``scripts/run_v2_lockbox.py`` sets.  The development path can never score 2021.

The script never imports ``paper_reference`` and never compares against a
published metric: V2 is a new experimental system, not a reconstruction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_forecaster.utils import setup_logging
from agentic_forecaster.v2.experiment import load_run_config, run_experiment
from agentic_forecaster.v2.firewall import V2_LOCKBOX_YEAR, lockbox_unlocked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="configs/v2/v2_<letter>_<name>.yaml")
    parser.add_argument("--fold", default=None,
                        help="V2_DEV_FOLD_A | V2_DEV_FOLD_B | V2_LOCKBOX")
    parser.add_argument("--variant", default=None, help="override the config variant")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto | cpu | cuda | cuda:N")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--experiment-id", default=None)
    args = parser.parse_args(argv)

    setup_logging()
    config = load_run_config(args.config, variant=args.variant, fold=args.fold,
                             seed=args.seed, device=args.device)
    if config.fold == "V2_LOCKBOX" and not lockbox_unlocked():
        parser.error(
            f"{V2_LOCKBOX_YEAR} is the V2 architecture lockbox. Use "
            "scripts/run_v2_lockbox.py (which sets V2_LOCKBOX=1). Never 2022+."
        )

    summary = run_experiment(config, out_dir=args.out_dir,
                             experiment_id=args.experiment_id)
    direction = summary["metrics"].get("direction", {})
    print(json.dumps({
        "experiment_id": summary["experiment_id"],
        "variant": summary["variant"],
        "fold": summary["fold"],
        "seed": summary["seed"],
        "out_dir": summary["out_dir"],
        "accuracy_micro": direction.get("accuracy_micro"),
        "accuracy_macro_ticker": direction.get("accuracy_macro_ticker"),
        "f1": direction.get("f1"),
        "roc_auc": direction.get("roc_auc"),
        "brier": direction.get("brier"),
        "ece": direction.get("ece"),
        "train_majority_baseline": direction.get("train_majority_baseline"),
        "precision_at_3_up": summary["metrics"]["selection"]["precision_at_3_up"],
        "precision_at_3_down": summary["metrics"]["selection"]["precision_at_3_down"],
        "n_parameters": summary["complexity"]["total_parameters"],
        "training_seconds": summary["complexity"]["training_seconds"],
        "best_epoch": summary["complexity"]["best_epoch"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())