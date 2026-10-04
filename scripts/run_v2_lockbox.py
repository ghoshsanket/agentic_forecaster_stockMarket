#!/usr/bin/env python3
"""The ONE permitted V2 lockbox evaluation: 2021.

2021 is the V2 architecture lockbox.  It exists so that the V2-A .. V2-F
comparison cannot be tuned against it: the winner is selected on V2_DEV_FOLD_A
and V2_DEV_FOLD_B only, frozen in ``results/v2/v2_dev_selection.json``, and only
then scored here.

HARD RULES ENFORCED HERE
-------------------------
* ``V2_LOCKBOX=1`` must be set (this script sets it itself only after checking
  that a frozen selection exists).
* The lockbox fold trains through 2020-12-31 and validates 2021-01-01..12-31.
* 2022 and 2023 are NEVER scored.  The V2 test firewall is active for the whole
  programme, including this script, and the feature store does not even contain a
  row beyond 2021-12-31.
* Exactly ONE architecture, seed 42.  The frozen selection is not editable after
  this run, and no ensemble is introduced.

Usage::

    V2_LOCKBOX=1 uv run python scripts/run_v2_lockbox.py \
        --config configs/v2/v2_c_contextual.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_forecaster.utils import atomic_json_dump, ensure_dir, setup_logging
from agentic_forecaster.v2.experiment import load_run_config, run_experiment
from agentic_forecaster.v2.firewall import (
    V2_LOCKBOX_YEAR,
    V2_PAPER_TEST_FIREWALL_START,
    assert_no_paper_test_targets,
    lockbox_unlocked,
)

SELECTION_PATH = REPO_ROOT / "results" / "v2" / "v2_dev_selection.json"
LOCKBOX_SEED = 42


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="the config of the ALREADY SELECTED variant")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--seed", type=int, default=LOCKBOX_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--selection", type=Path, default=SELECTION_PATH)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    setup_logging()

    if not lockbox_unlocked():
        parser.error(
            f"{V2_LOCKBOX_YEAR} is the V2 lockbox: set V2_LOCKBOX=1 explicitly. This "
            "is the only script allowed to do so."
        )
    if not args.selection.is_file():
        parser.error(
            f"refusing to score the lockbox without a frozen selection: {args.selection} "
            "is missing. Select a winner on V2_DEV_FOLD_A/B first "
            "(scripts/summarize_v2_dev.py --freeze-selection V2-X)."
        )
    selection = json.loads(args.selection.read_text())
    if selection.get("test_2022_2023_evaluated"):
        parser.error("the frozen selection claims a 2022/2023 evaluation; refusing")
    if args.variant and args.variant.upper() != selection.get("selected_variant"):
        parser.error(
            f"config variant {args.variant} does not match the frozen selection "
            f"{selection.get('selected_variant')}. The lockbox evaluates exactly ONE "
            "already-selected architecture."
        )
    if args.seed != LOCKBOX_SEED:
        parser.error(f"the single lockbox run uses seed {LOCKBOX_SEED}, not {args.seed}")

    # 2022/2023 can never be scored here, whatever the config says.
    assert_no_paper_test_targets(["2022-01-01", "2023-12-31"], where="v2 lockbox script")

    config = load_run_config(args.config, variant=args.variant, fold="V2_LOCKBOX",
                             seed=args.seed, device=args.device)
    summary = run_experiment(config, out_dir=args.out_dir)

    direction = summary["metrics"].get("direction", {})
    report = {
        "classification": "NEW_EXPERIMENTAL_ARCHITECTURE",
        "is_original_paper_model": False,
        "lockbox_year": V2_LOCKBOX_YEAR,
        "paper_test_firewall_start": str(V2_PAPER_TEST_FIREWALL_START.date()),
        "selected_variant": selection.get("selected_variant"),
        "selection_sha256": selection.get("selection_sha256"),
        "seed": args.seed,
        "experiment_id": summary["experiment_id"],
        "experiment_dir": summary["out_dir"],
        "direction": direction,
        "selection_metrics": summary["metrics"].get("selection", {}),
        "selective_accuracy": summary["metrics"].get("selective_accuracy", {}),
        "complexity": summary["complexity"],
        "test_2022_2023_evaluated": False,
        "note": ("one architecture, one seed, one run; the winner was selected on "
                 "V2_DEV_FOLD_A/B only and this configuration was not modified "
                 "after seeing this result"),
    }
    out_dir = ensure_dir(Path(summary["out_dir"]).parent)
    atomic_json_dump(report, out_dir / "v2_lockbox_report.json")
    print(json.dumps({
        "variant": report["selected_variant"],
        "accuracy_micro": direction.get("accuracy_micro"),
        "accuracy_macro_ticker": direction.get("accuracy_macro_ticker"),
        "f1": direction.get("f1"),
        "roc_auc": direction.get("roc_auc"),
        "brier": direction.get("brier"),
        "ece": direction.get("ece"),
        "train_majority_baseline": direction.get("train_majority_baseline"),
        "delta_vs_train_majority_macro": direction.get("delta_vs_train_majority_macro"),
        "precision_at_3_up": report["selection_metrics"].get("precision_at_3_up"),
        "precision_at_3_down": report["selection_metrics"].get("precision_at_3_down"),
        "selective_accuracy": report["selective_accuracy"],
        "report": str(out_dir / "v2_lockbox_report.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())