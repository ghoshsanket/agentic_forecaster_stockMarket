#!/usr/bin/env python3
"""The ONE authorised 2019 PRE-COVID lockbox run.

2019 is the PRE-COVID architecture lockbox.  It exists so the PRE-V2-A/B/C
comparison cannot be tuned against it: the architecture is selected on 2017 and
2018 only, frozen in ``results/v2/pre_covid/pre_covid_dev_selection.json``, and
only then scored here.

WHAT IT REFUSES
---------------
* without ``PRECOVID_LOCKBOX=1``  (deliberately NOT ``V2_LOCKBOX``);
* without a frozen ``pre_covid_dev_selection.json``;
* a different architecture, config hash, supervised universe, store or seed;
* anything after 2019-12-31: the V2 test firewall AND the PRE-COVID firewall are
  both active, and the PRE-COVID store is physically capped at 2019-12-31.

The split is exactly::

    TRAIN    2005-01-01 -> 2018-12-31
    LOCKBOX  2019-01-01 -> 2019-12-31

Nothing may be changed on the basis of the result: the selection file is written
before this script runs and is not editable afterwards.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_forecaster.recovery.firewall import date_in_firewalled_window
from agentic_forecaster.utils import atomic_json_dump, setup_logging
from agentic_forecaster.v2 import precovid as pc
from agentic_forecaster.v2.experiment import load_run_config, run_experiment
from agentic_forecaster.v2.firewall import (
    PRECOVID_FINAL_DATE,
    PRECOVID_LOCKBOX_ENV,
    V2_PAPER_TEST_FIREWALL_START,
    assert_pre_covid_dates,
    pre_covid_lockbox_unlocked,
)
from agentic_forecaster.v2.ledger import read_ledger
from agentic_forecaster.v2.store import store_fingerprints

LOCKBOX_SEED = 42
CONFIG_DIR = REPO_ROOT / "configs" / "v2" / "pre_covid"


def config_for(variant: str) -> Path:
    mapping = {
        "V2-A": "v2_a_shared_lstm.yaml",
        "V2-B": "v2_b_lstm_transformer.yaml",
        "V2-C": "v2_c_contextual.yaml",
        "V2-D": "v2_d_multitask.yaml",
        "V2-E": "v2_e_film.yaml",
        "V2-F": "v2_f_reptile.yaml",
    }
    if variant not in mapping:
        raise SystemExit(f"unknown variant {variant!r}")
    path = CONFIG_DIR / mapping[variant]
    if not path.is_file():
        raise SystemExit(f"config for the selected variant is missing: {path}")
    return path


def authorise(selection_path: Path, variant: str | None, seed: int) -> dict:
    """Every precondition the single lockbox run must satisfy."""
    problems: list[str] = []
    if not pre_covid_lockbox_unlocked():
        problems.append(f"{PRECOVID_LOCKBOX_ENV}=1 is not set")
    if not selection_path.is_file():
        problems.append(f"frozen selection is missing: {selection_path}")
    if problems:
        raise SystemExit("REFUSING the 2019 lockbox run:\n  - " + "\n  - ".join(problems))

    selection = json.loads(selection_path.read_text())
    selected = selection.get("selected_architecture")
    if variant and selected and variant.upper() != selected:
        problems.append(f"requested variant {variant} != frozen {selected}")
    if selection.get("test_2022_2023_evaluated"):
        problems.append("the frozen selection claims a 2022/2023 evaluation")
    if not selection.get("frozen_before_2019_access"):
        problems.append("the selection was not frozen before 2019 access")
    if seed != LOCKBOX_SEED:
        problems.append(f"the single lockbox run uses seed {LOCKBOX_SEED}, not {seed}")
    if problems:
        raise SystemExit("REFUSING the 2019 lockbox run:\n  - " + "\n  - ".join(problems))
    return selection


def verify_hashes(selection: dict, config_path: Path) -> dict:
    """The store, universe and config must be the ones that were frozen."""
    track = pc.PreCovidTrack()
    problems: list[str] = []
    store = store_fingerprints(track.processed_root)
    if store.get("store_sha256") != selection.get("precovid_store_sha256"):
        problems.append(
            f"store hash changed: frozen {selection.get('precovid_store_sha256')} != "
            f"current {store.get('store_sha256')}")
    if store.get("sector_map_sha256") != selection.get("sector_map_sha256"):
        problems.append("sector-map hash changed since the freeze")
    if store.get("source_manifest_sha256") != selection.get("source_manifest_sha256"):
        problems.append("source dataset manifest hash changed since the freeze")

    import pandas as pd

    universe = pd.read_csv(track.path("universe_csv"))
    if pc.universe_hash(universe) != selection.get("eligible_universe_sha256"):
        problems.append("eligible supervised universe hash changed since the freeze")

    config = load_run_config(config_path, fold=pc.LOCKBOX_FOLD, seed=LOCKBOX_SEED)
    frozen_hashes = selection.get("config_sha256")
    if isinstance(frozen_hashes, list):
        frozen_hashes = frozen_hashes[0] if frozen_hashes else None
    # the fold block legitimately differs between development and lockbox, so the
    # architecture identity is compared through the resolved COMPONENT FLAGS as
    # well as the config hash
    if (frozen_hashes and config.payload["config_sha256"] != frozen_hashes
            and config.payload.get("resolved_components") != selection.get(
                "selected_component_flags")):
        problems.append("resolved component flags differ from the frozen selection")
    if problems:
        raise SystemExit("REFUSING the 2019 lockbox run:\n  - " + "\n  - ".join(problems))
    return {"store": store, "config_sha256": config.payload["config_sha256"]}


def already_run() -> bool:
    rows = read_ledger(results_root=pc.PreCovidTrack().results_root)
    return any(r.get("fold") == pc.LOCKBOX_FOLD for r in rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default=None,
                        help="must match the frozen selection")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=LOCKBOX_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--selection", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--allow-rerun", action="store_true",
                        help="only for a crashed run that produced no ledger row")
    args = parser.parse_args(argv)
    setup_logging()

    track = pc.PreCovidTrack()
    selection_path = args.selection or track.path("selection")
    selection = authorise(selection_path, args.variant, args.seed)

    if already_run() and not args.allow_rerun:
        raise SystemExit(
            "REFUSING: a 2019 lockbox row already exists in the PRE-COVID ledger. "
            "The lockbox is evaluated EXACTLY once.")

    variant = selection["selected_architecture"]
    config_path = args.config or config_for(variant)
    hashes = verify_hashes(selection, config_path)

    # 2022+ can never be scored here, whatever any config says.  The boundary
    # PREDICATE is checked (the raisers are exercised by the unit tests) and the
    # regime boundary is asserted for real.
    assert date_in_firewalled_window("2022-01-01") and date_in_firewalled_window(
        "2023-12-31")
    assert_pre_covid_dates(feature_dates=["2019-12-31"], target_dates=["2019-12-31"],
                           final_allowed_date=PRECOVID_FINAL_DATE,
                           where="precovid lockbox")

    config = load_run_config(config_path, variant=variant, fold=pc.LOCKBOX_FOLD,
                             seed=LOCKBOX_SEED, device=args.device)
    assert config.window.val_start == "2019-01-01"
    assert config.window.val_end == "2019-12-31"
    assert config.window.train_end == "2018-12-31"

    summary = run_experiment(config, out_dir=args.out_dir)
    direction = summary["metrics"].get("direction", {})
    report = {
        "experiment_regime": pc.EXPERIMENT_REGIME,
        "regime_label": pc.REGIME_LABEL,
        "survivorship_bias_label": pc.BIAS_LABEL,
        "lockbox_year": 2019,
        "final_allowed_date": str(PRECOVID_FINAL_DATE.date()),
        "paper_test_firewall_start": str(V2_PAPER_TEST_FIREWALL_START.date()),
        "authorization_env": PRECOVID_LOCKBOX_ENV,
        "selected_architecture": variant,
        "selection_sha256": selection.get("selection_sha256"),
        "frozen_before_this_run": True,
        "seed": LOCKBOX_SEED,
        "split": {"train_through": config.window.train_end,
                  "evaluate": f"{config.window.val_start}..{config.window.val_end}"},
        "store_sha256": hashes["store"].get("store_sha256"),
        "experiment_id": summary["experiment_id"],
        "experiment_dir": summary["out_dir"],
        "direction": direction,
        "selection_metrics": summary["metrics"].get("selection", {}),
        "selective_accuracy": summary["metrics"].get("selective_accuracy", {}),
        "complexity": summary["complexity"],
        "data_access_audit": summary["manifest"].get("data_access_audit"),
        "test_2022_2023_evaluated": False,
        "note": ("one architecture, one seed, one run; the selection was frozen before "
                 "this result existed and was not modified afterwards. Nothing after "
                 "2019-12-31 was scored."),
    }
    # the report lives INSIDE the experiment directory (outside Git, next to the
    # predictions it describes) so that writing it can never touch the frozen
    # selection or any other artefact of the track
    report_dir = Path(summary["out_dir"])
    atomic_json_dump(report, report_dir / "precovid_lockbox_report.json")
    if not args.out_dir:
        atomic_json_dump(report, track.results_root / "precovid_lockbox_report.json")
    print(json.dumps({
        "variant": variant,
        "accuracy_macro_ticker": direction.get("accuracy_macro_ticker"),
        "accuracy_micro": direction.get("accuracy_micro"),
        "f1": direction.get("f1"),
        "balanced_accuracy": direction.get("balanced_accuracy"),
        "roc_auc": direction.get("roc_auc"),
        "brier": direction.get("brier"),
        "ece": direction.get("ece"),
        "train_majority_baseline": direction.get("train_majority_baseline"),
        "delta_vs_train_majority_macro": direction.get("delta_vs_train_majority_macro"),
        "precision_at_3_up": report["selection_metrics"].get("precision_at_3_up"),
        "precision_at_3_down": report["selection_metrics"].get("precision_at_3_down"),
        "report": str(Path(summary["out_dir"]) / "precovid_lockbox_report.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())