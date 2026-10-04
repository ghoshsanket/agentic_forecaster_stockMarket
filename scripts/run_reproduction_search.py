#!/usr/bin/env python3
"""Run a performance-recovery search experiment on PRE-2022 data only.

Two modes:

``--stage 0``  sanity gate
    Tiny-overfit and shuffled-label control on one ticker. If the tiny overfit
    FAILS the run stops immediately and no later stage may proceed.

(default)     a real search experiment
    Trains one Attention-LSTM per ticker on the search fold's TRAIN window,
    restores the best checkpoint, scores the VALIDATION window with the
    firewall-guarded scorer, and appends one row to the experiment ledger.

The test-set firewall is enabled for the whole process. A search fold has no
test window, the DataAgent produces an EMPTY test split, and this script
asserts that emptiness plus pre-2022 date bounds before returning.

Examples
--------
STAGE 0::

    uv run python scripts/run_reproduction_search.py \\
        --config configs/reproduction_search/paper_snapshot_2025_unadjusted_perf.yaml \\
        --stage 0 --search-fold SEARCH_FOLD_C --tickers RELIANCE --device auto

A real search point::

    uv run python scripts/run_reproduction_search.py \\
        --config configs/reproduction_search/paper_snapshot_2025_unadjusted_perf.yaml \\
        --search-fold SEARCH_FOLD_C --tickers RELIANCE,TCS,INFY \\
        --training-length T30 --feature-family F2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

CANONICAL = ["Date", "Open", "High", "Low", "Close", "Volume"]


# ---------------------------------------------------------------- utilities

def runtime_dir() -> Path:
    from agentic_forecaster.recovery.ledger import runtime_dir as _rd
    return _rd()


def repo_dir() -> Path:
    return REPO_ROOT


def assert_no_test_data(dataset, experiment: str) -> dict:
    """Item 17: a search run must never produce test data.

    Asserts the test split is empty and that no train/validation origin or
    target date reaches the firewalled years. Stops immediately on violation.
    """
    import pandas as pd

    from agentic_forecaster.recovery.firewall import TEST_FIREWALL_START

    if len(dataset.test.y):
        raise AssertionError(
            f"{experiment}: dataset.test is NOT empty ({len(dataset.test.y)} rows). "
            "A search run must never materialise test data.")
    bounds: dict[str, str | None] = {}
    for name, split in (("train", dataset.train), ("validation", dataset.val)):
        if not len(split.y):
            continue
        for axis, arr in (("origin", split.dates), ("target", split.target_dates)):
            mx = pd.to_datetime(pd.Series(arr)).max()
            if mx >= TEST_FIREWALL_START:
                raise AssertionError(
                    f"{experiment}: max {name} {axis} date {mx.date()} is at/after "
                    f"{TEST_FIREWALL_START.date()}.")
            bounds[f"max_{name}_{axis}"] = str(mx.date())
        bounds[f"n_{name}"] = len(split.y)
    bounds["test_rows"] = len(dataset.test.y)
    return bounds


def resolve_config(args, base_cfg: dict) -> dict:
    """Apply the search fold and every requested variant to a config copy."""
    import copy

    from agentic_forecaster.recovery import folds, variants

    cfg = folds.fold_config_for(args.search_fold, base_cfg)
    # search_mode makes the DataAgent enforce the firewall on real data.
    cfg["data"]["search_mode"] = True

    tl = args.training_length or variants.DEFAULT_TRAINING_LENGTH
    train = variants.TRAINING_LENGTHS[tl]
    attn = cfg.setdefault("models", {}).setdefault("attention_lstm", {})
    attn["max_epochs"] = train["max_epochs"]
    attn["patience"] = train["patience"]
    attn["restore_best_checkpoint"] = train["restore_best_checkpoint"]

    fam = args.feature_family or variants.DEFAULT_FEATURE_FAMILY
    rsi = args.rsi_method or variants.DEFAULT_RSI_METHOD
    indicators = list(variants.build_feature_indicators(fam, rsi))
    feat = cfg.setdefault("features", {})
    feat["indicators"] = indicators
    feat["use_ohlcv"] = variants.FEATURE_FAMILIES[fam]["use_ohlcv"]

    if args.lookback is not None:
        cfg["data"]["sequence_length"] = int(args.lookback)
    cfg["data"]["scaler"] = args.scaler or variants.DEFAULT_SCALER
    cfg["data"]["volume_mode"] = args.volume_mode or variants.DEFAULT_VOLUME_MODE
    if args.architecture:
        arch = variants.ARCHITECTURES[args.architecture]
        attn["hidden_size"] = arch["hidden_size"]
        attn["num_layers"] = arch["num_layers"]
    if args.dropout is not None:
        attn["dropout"] = float(args.dropout)
    if args.weight_decay is not None:
        attn["weight_decay"] = float(args.weight_decay)
    attn["class_weighting"] = args.class_weighting or variants.DEFAULT_CLASS_WEIGHTING
    # Stages 0/A/B/C score the RAW p(up) by default. Comparing calibration
    # methods needs a calibrated score, so leaving a calibrator on by default
    # would confound every A-C comparison with a calibration effect. Stage D is
    # the stage that exists to study calibration, so it keeps its own default.
    _default_calibration = (
        "temperature" if str(args.stage).upper() in {"D", "4"}
        else "none")
    cfg.setdefault("calibration", {})["method"] = (
        args.calibration or _default_calibration)
    cfg.setdefault("experiment", {})["seed"] = (
        args.seed if args.seed is not None else variants.DEFAULT_SEED)
    cfg["experiment"]["name"] = f"recovery_{args.stage}_{args.search_fold}"
    return copy.deepcopy(cfg)


def describe_resolved(cfg: dict, args) -> dict:
    from agentic_forecaster.recovery import variants
    attn = cfg["models"]["attention_lstm"]
    return {
        "search_fold": args.search_fold,
        "stage": args.stage,
        "dataset_variant": cfg["data"].get("variant"),
        "feature_family": args.feature_family or variants.DEFAULT_FEATURE_FAMILY,
        "n_indicators": len(cfg["features"]["indicators"]),
        "indicators": list(cfg["features"]["indicators"]),
        "rsi_method": args.rsi_method or variants.DEFAULT_RSI_METHOD,
        "lookback": cfg["data"]["sequence_length"],
        "scaler": cfg["data"]["scaler"],
        "volume_mode": cfg["data"]["volume_mode"],
        "hidden_size": attn["hidden_size"],
        "num_layers": attn["num_layers"],
        "dropout": attn["dropout"],
        "weight_decay": attn["weight_decay"],
        "class_weighting": attn["class_weighting"],
        "max_epochs": attn["max_epochs"],
        "patience": attn["patience"],
        "restore_best_checkpoint": attn["restore_best_checkpoint"],
        "calibration": cfg["calibration"]["method"],
        "seed": cfg["experiment"]["seed"],
        "tickers": args.tickers or "<all>",
        "data_windows": {k: cfg["data"].get(k) for k in
                         ("train_start", "train_end", "val_start", "val_end")},
        "test_window": "REMOVED (search folds have no test period)",
        "search_mode": True,
    }


def dataset_manifest_hash(cfg: dict) -> str:
    """Hash the ACTUAL dataset manifest the run consumed."""
    from agentic_forecaster.recovery.ledger import hash_file
    root = Path(cfg["data"]["raw_root"])
    for up in [root, *root.parents]:
        for m in up.glob("manifests/*manifest*.json"):
            return hash_file(m)
    return "unavailable"


# ----------------------------------------------------------------- STAGE 0

def run_stage_zero(args, cfg: dict, resolved: dict) -> int:
    """Tiny overfit + shuffled-label control. A failed overfit STOPS the run."""

    import yaml as _yaml

    from agentic_forecaster.data.agent import DataAgent
    from agentic_forecaster.recovery.firewall import firewall_guard
    from agentic_forecaster.recovery.harness import (
        run_real_label_reference,
        run_shuffled_label_control,
        run_tiny_overfit,
        save_stage0,
    )
    from agentic_forecaster.recovery.ledger import append_experiment, hash_file

    out_dir = runtime_dir() / "stage0"
    out_dir.mkdir(parents=True, exist_ok=True)
    ticker = (args.tickers.split(",")[0].strip().upper()
              if args.tickers else "RELIANCE")

    agent = DataAgent(cfg)
    dataset = agent.run(ticker)
    bounds = assert_no_test_data(dataset, "STAGE_0")
    print(f"[STAGE 0] {ticker}: train={bounds.get('n_train')} "
          f"validation={bounds.get('n_validation')} test={bounds['test_rows']}")

    Xtr, ytr = dataset.train.X, dataset.train.y
    Xva, yva = dataset.val.X, dataset.val.Y if hasattr(dataset.val, "Y") else dataset.val.y
    # STAGE 0 only uses TRAIN/VALIDATION. The tiny subset is carved from TRAIN.
    #
    # Capacity and budget matter here. A first run with hidden=8 / 1 layer /
    # 300 epochs reported 0.72 train accuracy and the gate fired, but a direct
    # probe showed the SAME pipeline reaches 0.99 train accuracy with the
    # paper's own architecture and optimizer. An under-provisioned gate produces
    # false alarms, so STAGE 0 uses the paper's REFERENCE architecture
    # (2 layers x 64 units), the paper's learning rate (1e-3) and a budget
    # large enough to memorise. A failure now genuinely means the model,
    # optimizer, gradients or target alignment are broken.
    with firewall_guard(True):
        overfit = run_tiny_overfit(
            Xtr, ytr, Xva, yva, val_dates=dataset.val.dates,
            seed=cfg["experiment"]["seed"], hidden_size=64, num_layers=2,
            dropout=0.0, learning_rate=1e-3, batch_size=32,
            epochs=1000, patience=1000, max_train_rows=256,
            name="tiny_overfit")
    print(f"[STAGE 0] tiny_overfit: passed={overfit.passed} "
          f"best_train_accuracy={overfit.best_train_accuracy:.4f} "
          f"best_train_loss={overfit.best_train_loss:.4f}")

    if not overfit.passed:
        # STOP: the pipeline cannot fit its own training data.
        save_stage0([overfit], out_dir)
        print("\nSTOP: tiny overfit FAILED. Investigate model / optimizer / target "
              "alignment / gradients / scaling before any later stage. "
              "No shuffled-label control was run and no search may proceed.")
        return 1

    # Only meaningful once the overfit passed.
    with firewall_guard(True):
        control = run_shuffled_label_control(
            Xtr, ytr, Xva, yva, val_dates=dataset.val.dates,
            seed=cfg["experiment"]["seed"], hidden_size=64, num_layers=2,
            dropout=0.0, learning_rate=1e-3, batch_size=32,
            epochs=30, patience=10, name="shuffled_label_control")
    print(f"[STAGE 0] shuffled_label_control: passed={control.passed} "
          f"validation_accuracy={control.validation_accuracy:.4f} "
          f"(majority={control.majority_validation_accuracy:.4f})")

    # A fair comparison needs a NORMAL real-label model, not the deliberately
    # overfit probe, as the control's opponent.
    with firewall_guard(True):
        reference = run_real_label_reference(
            Xtr, ytr, Xva, yva, val_dates=dataset.val.dates,
            seed=cfg["experiment"]["seed"], hidden_size=64, num_layers=2,
            dropout=0.0, learning_rate=1e-3, batch_size=32,
            epochs=100, patience=10, name="real_label_reference")
    print(f"[STAGE 0] real_label_reference: validation_accuracy="
          f"{reference.validation_accuracy:.4f} "
          f"(majority={reference.majority_validation_accuracy:.4f})")

    payload = save_stage0([overfit, control, reference], out_dir)
    # Hash the actual written config file so the ledger hash is reproducible
    # from the artifact with `sha256sum`, exactly as for search experiments.
    stage0_cfg = out_dir / "resolved_config.yaml"
    stage0_cfg.write_text(_yaml.safe_dump(cfg, sort_keys=False))
    cfg_hash = hash_file(stage0_cfg)
    # Each Stage-0 experiment gets its OWN ledger metadata. A single shared
    # dict previously recorded max_epochs=1000/patience=1000 and the overfit's
    # best_epoch for all three rows, which misrepresented the ledger.
    def _base() -> dict:
        return {
            "dataset_variant": cfg["data"].get("variant"),
            "dataset_hash": dataset_manifest_hash(cfg),
            "universe_id": cfg["data"].get("universe_id"),
            "config_hash": cfg_hash,
            "search_fold": args.search_fold,
            "ticker_subset": ticker,
            "feature_set": resolved["feature_family"],
            "rsi_method": resolved["rsi_method"],
            "lookback": resolved["lookback"],
            "scaler": resolved["scaler"],
            "hidden_size": 64, "layers": 2, "dropout": 0.0,
            "weight_decay": 0.0, "class_weighting": "none",
            "calibration_method": "none", "seed": cfg["experiment"]["seed"],
            "test_evaluated": "false",
        }

    append_experiment({**_base(),
                       "max_epochs": 1000, "patience": 1000,
                       "best_epoch": overfit.best_epoch,
                       "epochs_run": overfit.epochs_run,
                       "notes": (f"STAGE_0 tiny_overfit: requested_epochs=1000 "
                                 f"patience=1000 best_epoch={overfit.best_epoch} "
                                 f"epochs_run={overfit.epochs_run} "
                                 f"best_train_acc={overfit.best_train_accuracy:.4f} "
                                 f"best_train_loss={overfit.best_train_loss:.4f} "
                                 f"passed={overfit.passed}"),
                       "validation_accuracy": overfit.validation_accuracy,
                       "validation_brier": overfit.validation_brier,
                       "train_loss": overfit.best_train_loss}, root=repo_dir())

    append_experiment({**_base(),
                       "max_epochs": 30, "patience": 10,
                       "best_epoch": control.best_epoch,
                       "epochs_run": control.epochs_run,
                       "notes": (f"STAGE_0 shuffled_label_control: requested_epochs=30 "
                                 f"patience=10 best_epoch={control.best_epoch} "
                                 f"epochs_run={control.epochs_run} "
                                 f"val_acc={control.validation_accuracy:.4f} "
                                 f"majority={control.majority_validation_accuracy:.4f} "
                                 f"passed={control.passed}"),
                       "validation_accuracy": control.validation_accuracy,
                       "validation_brier": control.validation_brier,
                       "train_loss": control.train_loss}, root=repo_dir())

    append_experiment({**_base(),
                       "max_epochs": 100, "patience": 10,
                       "best_epoch": reference.best_epoch,
                       "epochs_run": reference.epochs_run,
                       "notes": (f"STAGE_0 real_label_reference: requested_epochs=100 "
                                 f"patience=10 best_epoch={reference.best_epoch} "
                                 f"epochs_run={reference.epochs_run} "
                                 f"val_acc={reference.validation_accuracy:.4f} "
                                 f"majority={reference.majority_validation_accuracy:.4f}"),
                       "validation_accuracy": reference.validation_accuracy,
                       "validation_brier": reference.validation_brier,
                       "train_loss": reference.train_loss}, root=repo_dir())

    print(f"\n[STAGE 0] artifacts -> {out_dir}")
    print(f"[STAGE 0] all_passed={payload['all_passed']}")
    print(f"[STAGE 0] GATE 1 (model can overfit TRAIN)            : "
          f"{'PASS' if overfit.passed else 'FAIL'} "
          f"(best_train_acc={overfit.best_train_accuracy:.4f})")
    print(f"[STAGE 0] GATE 2 (shuffled labels behave like chance) : "
          f"{'PASS' if control.passed else 'FAIL'} "
          f"(val_acc={control.validation_accuracy:.4f} vs "
          f"majority={control.majority_validation_accuracy:.4f})")
    real_beats_shuffled = (reference.validation_accuracy is not None
                           and control.validation_accuracy is not None
                           and reference.validation_accuracy > control.validation_accuracy)
    print(f"[STAGE 0] NOT A GATE - single-ticker signal check    : "
          f"real={reference.validation_accuracy:.4f} vs "
          f"shuffled={control.validation_accuracy:.4f} -> "
          f"{'signal seen' if real_beats_shuffled else 'NOT DEMONSTRATED'}")
    print("  One ticker and one pre-test window is not sufficient evidence of "
          "predictive signal, and its absence is not a reason to stop. Whether "
          "signal appears consistently across stocks and windows is what the "
          "STAGE A PILOT determines.")
    print("\nSTAGE 0 complete. Stage A pilot may now be run.")
    return 0 if payload["all_passed"] else 1


# ------------------------------------------------------------- search point

def run_search_experiment(args, cfg: dict, resolved: dict) -> int:
    """Train one Attention-LSTM per ticker; score VALIDATION only."""
    import pandas as pd
    import yaml as _yaml

    from agentic_forecaster.agents.model_agent import ModelAgent
    from agentic_forecaster.data.agent import DataAgent
    from agentic_forecaster.recovery.firewall import firewall_guard
    from agentic_forecaster.recovery.ledger import (
        append_experiment,
        hash_file,
        new_experiment_id,
    )
    from agentic_forecaster.recovery.scoring import (
        aggregate,
        cross_sectional_metrics,
        score_validation_with_targets,
    )

    tickers = ([t.strip().upper() for t in args.tickers.split(",") if t.strip()]
               if args.tickers else list(DataAgent(cfg).universe().requested))
    experiment_id = new_experiment_id()
    out_dir = runtime_dir() / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Hash the ACTUAL written config file, not a re-serialisation of a subset.
    # Anyone can recompute it with sha256sum on the artifact, and it changes if
    # and only if the effective configuration changes.
    cfg_path = out_dir / "resolved_config.yaml"
    cfg_path.write_text(_yaml.safe_dump(cfg, sort_keys=False))
    cfg_hash = hash_file(cfg_path)
    data_agent = DataAgent(cfg)
    model_agent = ModelAgent(cfg)

    per_ticker: list[dict] = []
    pred_rows: list[dict] = []
    history: dict = {}
    bounds_all: dict = {}

    for symbol in tickers:
        try:
            dataset = data_agent.run(symbol)
        except (FileNotFoundError, KeyError) as exc:
            print(f"  {symbol:12s} unavailable: {exc}")
            continue
        b = assert_no_test_data(dataset, f"{experiment_id}/{symbol}")
        bounds_all[symbol] = b
        fitted = model_agent.train_ticker(symbol, dataset, fold=args.search_fold,
                                          device=args.device)
        p = fitted.predict_proba(dataset.val.X)
        with firewall_guard(True):
            m = score_validation_with_targets(
                dataset.val.y, p, dataset.val.dates, dataset.val.target_dates,
                where=f"validation/{symbol}")
        best_epoch = fitted.train_config.get("best_epoch")
        per_ticker.append({"ticker": symbol, **m, "best_epoch": best_epoch,
                           "pos_weight": fitted.train_config.get("pos_weight"),
                           "calibration": fitted.calibration_method})
        history[symbol] = fitted.history
        for d, td, yv, pv in zip(dataset.val.dates, dataset.val.target_dates,
                                 dataset.val.y, p):
            pred_rows.append({"date": str(pd.Timestamp(d).date()),
                              "target_date": str(pd.Timestamp(td).date()),
                              "ticker": symbol, "y": float(yv), "p_up": float(pv)})
        print(f"  {symbol:12s} acc={m['accuracy']:.4f} f1={m['f1']:.4f} "
              f"brier={m['brier']:.4f} ece={m['ece']:.4f} best_epoch={best_epoch}")

    if not per_ticker:
        print("no tickers produced results; nothing to record")
        return 1

    agg = aggregate(per_ticker)
    preds = pd.DataFrame(pred_rows)
    if preds["ticker"].nunique() >= 3:
        with firewall_guard(True):
            agg.update(cross_sectional_metrics(preds, k=3, where="validation"))

    pd.DataFrame(per_ticker).to_csv(out_dir / "ticker_metrics.csv", index=False)
    preds.to_csv(out_dir / "validation_predictions.csv", index=False)
    (out_dir / "training_history.json").write_text(
        json.dumps(history, indent=2, default=str))
    (out_dir / "aggregate_validation_metrics.json").write_text(
        json.dumps({"experiment_id": experiment_id, "resolved": resolved,
                    "split_bounds": bounds_all, "aggregate": agg}, indent=2))
    (out_dir / "manifest.json").write_text(json.dumps({
        "experiment_id": experiment_id,
        "search_fold": args.search_fold,
        "tickers": [t["ticker"] for t in per_ticker],
        "test_evaluated": False,
        "test_rows": 0,
        "test_split_empty": True,
        "resolved": resolved,
    }, indent=2))

    row = append_experiment({
        "dataset_variant": cfg["data"].get("variant"),
        "dataset_hash": dataset_manifest_hash(cfg),
        "universe_id": cfg["data"].get("universe_id"),
        "config_hash": cfg_hash,
        "search_fold": args.search_fold,
        "ticker_subset": ",".join(t["ticker"] for t in per_ticker),
        "feature_set": resolved["feature_family"],
        "rsi_method": resolved["rsi_method"],
        "lookback": resolved["lookback"],
        "scaler": resolved["scaler"],
        "hidden_size": resolved["hidden_size"],
        "layers": resolved["num_layers"],
        "dropout": resolved["dropout"],
        "max_epochs": resolved["max_epochs"],
        "best_epoch": max((t["best_epoch"] or 0) for t in per_ticker),
        "patience": resolved["patience"],
        "weight_decay": resolved["weight_decay"],
        "class_weighting": resolved["class_weighting"],
        "calibration_method": resolved["calibration"],
        "seed": resolved["seed"],
        "train_loss": agg.get("train_loss"),
        "validation_accuracy": agg.get("accuracy"),
        "validation_f1": agg.get("f1"),
        "validation_brier": agg.get("brier"),
        "validation_ece": agg.get("ece"),
        "test_evaluated": "false",
        "notes": (f"{experiment_id} stage={args.stage} "
                  f"p@3up={agg.get('precision_at_3_up')} "
                  f"p@3down={agg.get('precision_at_3_down')} "
                  f"volume_mode={resolved['volume_mode']}"),
    }, root=repo_dir())
    print(f"\nexperiment_id : {row['experiment_id']}")
    print(f"artifacts     : {out_dir}")
    print(f"test_evaluated: {row['test_evaluated']}")
    return 0


# --------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--search-fold", default="SEARCH_FOLD_C")
    parser.add_argument("--tickers", default=None, help="comma-separated subset")
    parser.add_argument("--stage", default="A")
    parser.add_argument("--training-length", default=None, help="T10_AUTHOR_CONFIRMED (primary) / T3_RECONSTRUCTION_SHORTCUT / T30 / T50 / T100_DIAGNOSTIC")
    parser.add_argument("--feature-family", default=None, help="F0/F1/F2/F3")
    parser.add_argument("--rsi-method", default=None, help="R0_rolling/R1_wilder/R2_ema")
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--scaler", default=None, help="standard/minmax/robust")
    parser.add_argument("--volume-mode", default=None, help="raw/log1p")
    parser.add_argument("--architecture", default=None, help="A0/A1/A2/A3")
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--class-weighting", default=None, help="none/pos_weight")
    parser.add_argument("--calibration", default=None,
                        help="none/temperature/platt/isotonic")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve + firewall-check only; no training")
    args = parser.parse_args()

    from agentic_forecaster.config import load_config
    from agentic_forecaster.recovery.firewall import firewall_guard

    base = load_config(args.config)
    with firewall_guard(True):
        cfg = resolve_config(args, base)
        resolved = describe_resolved(cfg, args)

    if args.dry_run:
        print(json.dumps(resolved, indent=2))
        print("\nDRY RUN: firewall active, no training performed.")
        return 0

    with firewall_guard(True):
        if str(args.stage) == "0":
            return run_stage_zero(args, cfg, resolved)
        return run_search_experiment(args, cfg, resolved)


if __name__ == "__main__":
    raise SystemExit(main())
