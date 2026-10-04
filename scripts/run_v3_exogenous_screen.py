#!/usr/bin/env python3
"""Run the V3 cheap screen: X0..X4 x 1D/3D/5D/10D x 2014-2018 x two fixed models.

At most 5 families x 4 horizons x 5 folds x 2 models = 200 cheap fits.  No
hyper-parameter search, no neural network.

For every run the NATURAL and COMMON sample views are both scored, and
``incremental_auc_Xk = AUC(Xk) - AUC(X0)`` is recorded on the common sample so an
apparent gain cannot be an artefact of a different observation subset.

Usage::

    uv run python scripts/run_v3_exogenous_screen.py --model LOGISTIC --reset-ledger
    uv run python scripts/run_v3_exogenous_screen.py --model HIST_GRADIENT_BOOSTING
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_forecaster.utils import setup_logging
from agentic_forecaster.v2.horizons import load_track_config
from agentic_forecaster.v3.experiment import assemble_inputs, run_screen_family

CONFIG = REPO_ROOT / "configs" / "v3" / "precovid_exogenous_base.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--model", choices=("LOGISTIC", "HIST_GRADIENT_BOOSTING"),
                        required=True)
    parser.add_argument("--horizons", type=int, nargs="*", default=None)
    parser.add_argument("--folds", nargs="*", default=None)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--reset-ledger", action="store_true")
    parser.add_argument("--pca-components", type=int, default=None,
                        help="documented TRAIN-only PCA alternative for the composite")
    args = parser.parse_args(argv)
    setup_logging()

    payload = load_track_config(args.config)
    horizons = args.horizons or [int(h) for h in payload["horizons"]]
    folds = args.folds or list(payload["development_folds"])
    families = args.families or list(payload["feature_families"])

    inputs = assemble_inputs(args.config)
    manifest = run_screen_family(inputs, model_name=args.model, horizons=horizons,
                                 folds=folds, families=tuple(families),
                                 with_importance=True, reset_ledger=args.reset_ledger,
                                 pca_components=args.pca_components)
    print(json.dumps({k: manifest[k] for k in (
        "model", "horizons", "families", "n_runs", "transforms_fit_on",
        "hyperparameter_search", "2019_lockbox_evaluated", "post_2019_evaluated")},
        indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())