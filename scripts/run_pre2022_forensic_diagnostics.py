#!/usr/bin/env python3
"""Pre-2022 forensic diagnosis of the paper reconstruction.

WHY
---
The faithful reconstruction (author-confirmed 10 epochs) scores BELOW the
majority-class baseline on pre-2022 validation. Hyperparameter search would
therefore be tuning noise. This script instead probes the STRUCTURAL choices
that could explain a high historical result, and reports what it finds.

HARD RULES
----------
* No experiment may score a target date >= 2022-01-01. The firewall is
  asserted before every metric; a protected date ABORTS the entire run.
* `PAPER_REFERENCE` is never imported and no published metric is used as a
  target. These diagnostics are explanatory, not number-matching.
* The anchor is T10-F2 on unadjusted data. T100 is never substituted.
* Deliberately invalid probes (L1-L4) are labelled SCIENTIFICALLY_INVALID and
  are never presented as model results.
* The per-stock methodology is the author-confirmed design and is never
  replaced. Pooled and 85/15 variants are labelled as forensic diagnostics.

Usage:
    python scripts/run_pre2022_forensic_diagnostics.py --diagnostic all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_forecaster.agents.model_agent import ModelAgent
from agentic_forecaster.data.agent import DataAgent
from agentic_forecaster.models.attention_lstm import AttentionLSTM
from agentic_forecaster.recovery import folds as folds_mod
from agentic_forecaster.recovery import forensics as fx
from agentic_forecaster.recovery import variants
from agentic_forecaster.recovery.firewall import firewall_guard
from agentic_forecaster.recovery.scoring import (
    cross_sectional_metrics,
    score_validation_with_targets,
)
from agentic_forecaster.utils import seed_everything

TICKERS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC", "LT",
           "SUNPHARMA", "TATASTEEL"]
AUDIT_TICKERS = ["RELIANCE", "TCS", "INFY"]
PROBE_TICKERS = ["RELIANCE", "TCS", "INFY"]
SEARCH_FOLDS = ["SEARCH_FOLD_A", "SEARCH_FOLD_B", "SEARCH_FOLD_C"]

#: The legitimate forensic anchor. Author-confirmed epochs; F2 features.
ANCHOR = {
    "training_length": "T10_AUTHOR_CONFIRMED",
    "feature_family": "F2",
    "rsi_method": "R1_wilder",
    "lookback": 30,
    "scaler": "standard",
    "volume_mode": "raw",
    "architecture": "A0",
    "dropout": 0.2,
    "weight_decay": 1e-4,
    "class_weighting": "none",
    "calibration": "none",
    "seed": 42,
}


# ---------------------------------------------------------------------------
# config construction
# ---------------------------------------------------------------------------

def build_config(base_path: str, fold: str, *,
                 adjusted: bool = False) -> dict:
    """Build a search-mode config for one fold, optionally switching to
    ADJUSTED data. The adjusted selector must change the REAL data root, not a
    cosmetic flag, or diagnostic A would silently re-run unadjusted data."""
    from agentic_forecaster.config import load_config
    cfg = load_config(base_path)
    cfg = folds_mod.fold_config_for(fold, cfg)
    cfg["data"]["search_mode"] = True
    if adjusted:
        raw = Path(cfg["data"]["raw_root"])
        # swap the `unadjusted` path segment for `adjusted`
        parts = [("adjusted" if p == "unadjusted" else p) for p in raw.parts]
        new_raw = Path(*parts)
        if not new_raw.is_dir():
            raise FileNotFoundError(f"adjusted root not found: {new_raw}")
        cfg["data"]["raw_root"] = str(new_raw)
        cfg["data"]["variant"] = "adjusted"
    train = variants.TRAINING_LENGTHS[ANCHOR["training_length"]]
    attn = cfg.setdefault("models", {}).setdefault("attention_lstm", {})
    attn.update({"max_epochs": train["max_epochs"],
                 "patience": train["patience"],
                 "restore_best_checkpoint": train["restore_best_checkpoint"]})
    cfg["data"]["sequence_length"] = ANCHOR["lookback"]
    cfg["data"]["scaler"] = ANCHOR["scaler"]
    cfg["data"]["volume_mode"] = ANCHOR["volume_mode"]
    arch = variants.ARCHITECTURES[ANCHOR["architecture"]]
    attn["hidden_size"] = arch["hidden_size"]
    attn["num_layers"] = arch["num_layers"]
    attn["dropout"] = ANCHOR["dropout"]
    attn["weight_decay"] = ANCHOR["weight_decay"]
    attn["class_weighting"] = ANCHOR["class_weighting"]
    cfg.setdefault("features", {})["indicators"] = list(
        variants.build_feature_indicators(ANCHOR["feature_family"],
                                          ANCHOR["rsi_method"]))
    cfg.setdefault("calibration", {})["method"] = ANCHOR["calibration"]
    cfg.setdefault("experiment", {})["seed"] = ANCHOR["seed"]
    return cfg


def _torch_proba(model, X: np.ndarray) -> np.ndarray:
    """Raw sigmoid p(up) from a trained torch module.

    Mirrors FittedModel._torch_raw: the trainer exposes no predict_proba, and
    these diagnostics deliberately need the UNCALIBRATED score.
    """
    import torch
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=device))
        return torch.sigmoid(logits).cpu().numpy().reshape(-1)


def _frame(cfg: dict) -> pd.DataFrame:
    df = pd.DataFrame({
        "ticker": cfg["tickers"], "fold": cfg["folds"],
        "y_true": cfg["y_true"], "p_up": cfg["p_up"],
        "target_date": cfg["target_date"],
    })
    df["predicted"] = (df["p_up"] >= 0.5).astype(int)
    df["correct"] = (df["predicted"] == df["y_true"]).astype(int)
    return df


def _f1(y, p) -> float:
    return fx._f1(np.asarray(y, int), np.asarray(p, float))


def _acc(y, p) -> float:
    return fx._acc(np.asarray(y, int), np.asarray(p, float))


# ---------------------------------------------------------------------------
# single legitimate T10-F2 run over all stocks/folds
# ---------------------------------------------------------------------------

def run_anchor(base_path: str, *, adjusted: bool = False,
               device: str = "auto",
               tickers: list[str] | None = None) -> dict:
    """Train and score the legitimate anchor. Returns per-case rows + frame."""
    tickers = tickers or TICKERS
    rows, pred_rows, fold_rows, hist = [], [], [], {}
    for fold in SEARCH_FOLDS:
        cfg = build_config(base_path, fold, adjusted=adjusted)
        da, ma = DataAgent(cfg), ModelAgent(cfg)
        fold_pred = []
        for sym in tickers:
            seed_everything(ANCHOR["seed"])
            ds = da.run(sym)
            # FIREWALL: asserted before ANY metric is computed.
            fx.assert_targets_pre_test(ds.train.target_dates,
                                       ds.val.target_dates,
                                       where=f"{fold}/{sym}")
            if len(ds.test.dates):
                raise fx.ForensicFirewallAbort(
                    f"{fold}/{sym}: test split is not empty")
            fitted = ma.train_ticker(sym, ds, fold=fold, device=device)
            p = fitted.predict_proba(ds.val.X)
            with firewall_guard(True):
                m = score_validation_with_targets(
                    ds.val.y, p, ds.val.dates, ds.val.target_dates,
                    where=f"{fold}/{sym}")
            rows.append({"ticker": sym, "fold": fold,
                         "variant": "adjusted" if adjusted else "unadjusted",
                         **m,
                         "best_epoch": fitted.train_config.get("best_epoch")})
            hist[f"{fold}/{sym}"] = {
                "train_loss": fitted.history.get("train_loss", []),
                "val_loss": fitted.history.get("val_loss", [])}
            for d, td, yv, pv in zip(ds.val.dates, ds.val.target_dates,
                                     ds.val.y, p):
                # `precision_at_k` expects the standard column names.
                r = {"ticker": sym, "fold": fold,
                     "date": str(pd.Timestamp(d).date()),
                     "target_date": str(pd.Timestamp(td).date()),
                     "y": int(yv), "y_true": int(yv), "p_up": float(pv)}
                pred_rows.append(r)
                fold_pred.append(r)
            bal = fx.label_balance(ds.train.y, ds.val.y)
            fold_rows.append({"ticker": sym, "fold": fold,
                              "train_positive_rate": bal["train"]["positive_rate"],
                              "val_positive_rate": bal["validation"]["positive_rate"],
                              "train_majority_baseline_accuracy":
                                  bal["train_majority_baseline_accuracy"],
                              "validation_oracle_majority_rate":
                                  bal["validation_oracle_majority_rate"]})
        # cross-sectional P@3 needs all tickers of the fold together
        cs = cross_sectional_metrics(pd.DataFrame(fold_pred), k=3)
        for r in rows:
            if r["fold"] == fold:
                r["precision_at_3_up"] = cs.get("precision_at_3_up")
                r["precision_at_3_down"] = cs.get("precision_at_3_down")
    frame = _frame({"tickers": [r["ticker"] for r in pred_rows],
                    "folds": [r["fold"] for r in pred_rows],
                    "y_true": [r["y_true"] for r in pred_rows],
                    "p_up": [r["p_up"] for r in pred_rows],
                    "target_date": [r["target_date"] for r in pred_rows]})
    agg = aggregate_rows(rows)
    return {"rows": rows, "frame": frame, "balance": fold_rows,
            "history": hist, "aggregate": agg}


def aggregate_rows(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    maj = df["majority_accuracy"].mean()
    return {"n_cases": len(df),
            "mean_accuracy": float(df["accuracy"].mean()),
            "mean_f1": float(df["f1"].mean()),
            "mean_brier": float(df["brier"].mean()),
            "mean_ece": float(df["ece"].mean()),
            "mean_majority_accuracy": float(maj),
            "beating_majority_pct": float(
                (df["accuracy"] > df["majority_accuracy"]).mean() * 100),
            "precision_at_3_up": float(df["precision_at_3_up"].mean())
            if "precision_at_3_up" in df else None,
            "precision_at_3_down": float(df["precision_at_3_down"].mean())
            if "precision_at_3_down" in df else None}


# ---------------------------------------------------------------------------
# Diagnostic A - adjusted vs unadjusted
# ---------------------------------------------------------------------------

def diagnostic_a(base_path: str, device: str) -> dict:
    print("[A] adjusted vs unadjusted T10-F2 ...", flush=True)
    unadj = json.loads((fx.repo_forensic_dir() /
                        "unadjusted_anchor_reference.json").read_text()) \
        if (fx.repo_forensic_dir() / "unadjusted_anchor_reference.json").is_file() \
        else None
    adj = run_anchor(base_path, adjusted=True, device=device)
    fx.write_forensic_json(
        {"diagnostic": "A_ADJUSTED_VS_UNADJUSTED", "anchor": ANCHOR,
         "adjusted": adj["aggregate"], "per_ticker": _per_ticker(adj["rows"]),
         "per_fold": _per_fold(adj["rows"]),
         "label_balance": adj["balance"]},
        fx.repo_forensic_dir() / "adjusted_anchor_metrics.json")
    fx.write_forensic_csv(adj["rows"],
                          fx.repo_forensic_dir() / "adjusted_anchor_rows.csv")
    return {"adjusted": adj["aggregate"], "unadjusted_reference": unadj,
            "per_ticker": _per_ticker(adj["rows"]),
            "per_fold": _per_fold(adj["rows"])}


def _per_ticker(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    return {t: {"accuracy": float(g["accuracy"].mean()),
                "f1": float(g["f1"].mean()),
                "brier": float(g["brier"].mean()),
                "ece": float(g["ece"].mean())}
            for t, g in df.groupby("ticker")}


def _per_fold(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    return {f: {"accuracy": float(g["accuracy"].mean()),
                "f1": float(g["f1"].mean()),
                "brier": float(g["brier"].mean()),
                "ece": float(g["ece"].mean()),
                "majority_accuracy": float(g["majority_accuracy"].mean())}
            for f, g in df.groupby("fold")}


# ---------------------------------------------------------------------------
# Diagnostic B - legitimate baseline panel
# ---------------------------------------------------------------------------

def diagnostic_b(base_path: str, device: str) -> dict:
    """Majority / LogisticRegression / RandomForest / PlainLSTM / AttentionLSTM.

    Trains on TRAIN only and scores VALIDATION only. The question is whether a
    simpler model captures signal the Attention model misses; if every
    legitimate model sits near majority, that is the finding.
    """
    print("[B] baseline panel ...", flush=True)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    from agentic_forecaster.models.lstm import PlainLSTM
    from agentic_forecaster.training.trainer import Trainer

    rows = []
    for fold in SEARCH_FOLDS:
        cfg = build_config(base_path, fold)
        da = DataAgent(cfg)
        for sym in TICKERS:
            seed_everything(ANCHOR["seed"])
            ds = da.run(sym)
            fx.assert_targets_pre_test(ds.train.target_dates,
                                       ds.val.target_dates,
                                       where=f"{fold}/{sym}")
            ytr, yva = np.asarray(ds.train.y, int), np.asarray(ds.val.y, int)
            # Train-majority predictor: the class comes from TRAIN labels only,
            # so this is a legitimate deployable baseline. The
            # validation-oracle rate is recorded separately for class-balance
            # description and is NEVER used as a baseline to beat.
            tr_majority = 1 if ytr.mean() >= 0.5 else 0
            r = _row(fold, sym, "TrainMajorityBaseline",
                     _acc(yva, np.full(len(yva), float(tr_majority))),
                     _f1(yva, np.full(len(yva), float(tr_majority))),
                     _brier_const(ytr, yva, tr_majority))
            r["train_majority_baseline_accuracy"] = fx.train_majority_baseline(ytr, yva)
            r["validation_oracle_majority_rate"] = fx.validation_oracle_majority_rate(yva)
            r["val_positive_rate"] = float(np.mean(yva))
            rows.append(r)

            # flat sklearn models on the last timestep of each sequence
            Xtr = ds.train.X[:, -1, :]
            Xva = ds.val.X[:, -1, :]
            for name, clf in (("LogisticRegression",
                               LogisticRegression(max_iter=1000)),
                              ("RandomForest",
                               RandomForestClassifier(
                                   n_estimators=300, max_depth=8,
                                   random_state=ANCHOR["seed"]))):
                clf.fit(Xtr, ytr)
                p = clf.predict_proba(Xva)[:, 1]
                fx.assert_targets_pre_test([], ds.val.target_dates,
                                           where=f"{fold}/{sym}")
                rows.append(_row(fold, sym, name, _acc(yva, p), _f1(yva, p),
                                 _brier(yva, p)))

            # PlainLSTM: same lookback/features/scaler/optimizer, 10 epochs
            seed_everything(ANCHOR["seed"])
            n_feat = ds.train.X.shape[-1]
            plstm = PlainLSTM(input_size=n_feat, hidden_size=64, num_layers=2,
                              dropout=ANCHOR["dropout"])
            tr = Trainer(plstm, learning_rate=0.001, beta1=0.9, beta2=0.999,
                         weight_decay=ANCHOR["weight_decay"], batch_size=64,
                         max_epochs=10, patience=10, restore_best_checkpoint=True,
                         device=device)
            fit = tr.fit(ds.train.X, ytr, ds.val.X, yva)
            p = _torch_proba(plstm, ds.val.X)
            r = _row(fold, sym, "PlainLSTM", _acc(yva, p), _f1(yva, p),
                     _brier(yva, p))
            r["best_epoch"] = fit.best_epoch
            rows.append(r)

            # AttentionLSTM T10-F2, trained through the standard agent
            ma = ModelAgent(cfg)
            fitted = ma.train_ticker(sym, ds, fold=fold, device=device)
            p = fitted.predict_proba(ds.val.X)
            rows.append(_row(fold, sym, "AttentionLSTM_T10F2", _acc(yva, p),
                             _f1(yva, p), _brier(yva, p)))
    df = pd.DataFrame(rows)
    fx.write_forensic_csv(rows, fx.repo_forensic_dir() / "baseline_panel.csv")
    agg = {n: {"accuracy": float(g["accuracy"].mean()),
               "f1": float(g["f1"].mean()),
               "brier": float(g["brier"].mean()),
               "n_cases": len(g)}
           for n, g in df.groupby("model")}
    maj = agg.get("TrainMajorityBaseline", {}).get("accuracy")
    for a in agg.values():
        a["delta_vs_train_majority"] = None if maj is None else a["accuracy"] - maj
    fx.write_forensic_json({"diagnostic": "B_BASELINE_PANEL",
                            "anchor": ANCHOR, "aggregate": agg,
                            "per_ticker": {n: {t: float(g2["accuracy"].mean())
                                               for t, g2 in g.groupby("ticker")}
                                           for n, g in df.groupby("model")}},
                           fx.repo_forensic_dir() / "baseline_panel.json")
    return {"aggregate": agg, "per_ticker": {
        n: {t: float(g2["accuracy"].mean()) for t, g2 in g.groupby("ticker")}
        for n, g in df.groupby("model")}}


def _row(fold, ticker, model, acc, f1, brier) -> dict:
    return {"fold": fold, "ticker": ticker, "model": model,
            "accuracy": acc, "f1": f1, "brier": brier}


def _brier(y, p) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def _brier_const(ytr, yva, cls) -> float:
    return _brier(yva, np.full(len(yva), float(cls)))


# ---------------------------------------------------------------------------
# Diagnostic C - pooled model
# ---------------------------------------------------------------------------

def diagnostic_c(base_path: str, device: str) -> dict:
    """One Attention-LSTM trained on POOLED train sequences, validated on all.

    NOT the author-confirmed design (one model per stock). Run to test whether a
    lost implementation might have pooled implicitly.
    """
    print("[C] pooled model ...", flush=True)
    from agentic_forecaster.training.trainer import Trainer

    rows = []
    for fold in SEARCH_FOLDS:
        cfg = build_config(base_path, fold)
        da = DataAgent(cfg)
        Xtr, ytr, dtr, ttr, Xva, yva, tva = [], [], [], [], [], [], []
        for sym in TICKERS:
            ds = da.run(sym)
            Xtr.append(ds.train.X); ytr.append(ds.train.y)
            dtr.append(ds.train.dates); ttr.append(ds.train.target_dates)
            Xva.append(ds.val.X); yva.append(ds.val.y); tva.append(ds.val.target_dates)
        Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr).astype(int)
        ttr = np.concatenate(ttr); Xva = np.concatenate(Xva)
        yva = np.concatenate(yva).astype(int); tva = np.concatenate(tva)
        fx.assert_targets_pre_test(ttr, tva, where=f"pooled/{fold}")

        seed_everything(ANCHOR["seed"])
        model = AttentionLSTM(input_size=Xtr.shape[-1], hidden_size=64,
                              num_layers=2, dropout=ANCHOR["dropout"])
        tr = Trainer(model, learning_rate=0.001, beta1=0.9, beta2=0.999,
                     weight_decay=ANCHOR["weight_decay"], batch_size=64,
                     max_epochs=10, patience=10, restore_best_checkpoint=True,
                     device=device)
        fit = tr.fit(Xtr, ytr, Xva, yva)
        p = _torch_proba(model, Xva)
        rows.append({"fold": fold, "model": fx.POOLED_LABEL,
                     "accuracy": _acc(yva, p), "f1": _f1(yva, p),
                     "brier": _brier(yva, p),
                     "best_epoch": fit.best_epoch,
                     "n_train": len(ytr), "n_val": len(yva)})
    df = pd.DataFrame(rows)
    # The pooled model is NOT author-confirmed, but it is a legitimate
    # scientific experiment -- it must NOT carry the leakage-probe marker.
    pooled_marker = {"VALID_FOR_FINAL_MODEL": False,
                     "SCIENTIFICALLY_INVALID": False,
                     "LABEL": fx.POOLED_LABEL}
    fx.write_forensic_csv(rows, fx.repo_forensic_dir() / "pooled_model.csv",
                          marker=pooled_marker)
    agg = {"accuracy": float(df["accuracy"].mean()),
           "f1": float(df["f1"].mean()), "brier": float(df["brier"].mean()),
           "best_epochs": df["best_epoch"].tolist()}
    fx.write_forensic_json({"diagnostic": "C_POOLED_MODEL", "label": fx.POOLED_LABEL,
                            "aggregate": agg,
                            "per_fold": df.to_dict("records")},
                           fx.repo_forensic_dir() / "pooled_model.json",
                           marker={"LABEL": fx.POOLED_LABEL})
    return {"label": fx.POOLED_LABEL, "aggregate": agg,
            "per_fold": df.to_dict("records")}


# ---------------------------------------------------------------------------
# Diagnostic D - pre-2022 85/15 analog
# ---------------------------------------------------------------------------

def diagnostic_d(base_path: str, device: str) -> dict:
    """LONG-HISTORY pre-2022 85/15 chronological analog.

    NOT the official walk-forward protocol, and not author-confirmed.

    The previous version built its sample list from SEARCH_FOLD_C, whose
    training window starts in 2016. That silently reduced the "long history"
    test to 2016-2021 -- roughly the same span as the legitimate folds, so it
    could not test what it claimed. This version loads each stock from the
    EARLIEST available raw history, truncates strictly at 2021-12-31, and splits
    the resulting eligible supervised samples chronologically 85/15.

    The first eligible sample legitimately falls later than the raw first date,
    because the 30-day lookback and the indicator warm-up consume early rows.
    """
    print("[D] pre-2022 LONG-HISTORY 85/15 analog ...", flush=True)
    from agentic_forecaster.training.trainer import Trainer

    rows, windows = [], []
    for sym in TICKERS:
        cfg = build_config(base_path, "SEARCH_FOLD_C")
        # open the window to the earliest legitimate history, ending at the
        # firewall boundary. The sample-split logic also requires each sample's
        # TARGET date to fall inside the window, so nothing can reach 2022.
        data = cfg["data"]
        for k in ("train_start", "train_end", "val_start", "val_end"):
            data.pop(k, None)
        data["train_start"] = fx.LONG_HISTORY_EARLIEST
        data["train_end"] = "2021-12-31"
        data["val_start"] = fx.LONG_HISTORY_EARLIEST
        data["val_end"] = "2021-12-31"
        raw = _raw_frame(cfg, sym)
        raw_first = str(pd.Timestamp(raw["date"].min()).date())

        ds = DataAgent(cfg).run(sym)
        # everything lands in `train` because the val window is identical and
        # the train branch is tested first.
        y = np.asarray(ds.train.y, int)
        td = np.asarray(ds.train.target_dates)
        fx.assert_targets_pre_test(td, [], where=f"long-history/{sym}")
        if not len(y):
            windows.append({"ticker": sym, "raw_first_date": raw_first,
                            "first_eligible_date": None, "final_date": None,
                            "n_eligible": 0})
            continue

        first_elig = str(pd.Timestamp(ds.train.dates[0]).date())
        final = str(pd.Timestamp(td[-1]).date())
        cut = max(1, min(len(y) - 1, round(len(y) * 0.85)))
        X, tr_y, va_y = ds.train.X, y[:cut], y[cut:]
        tr_td, va_td = td[:cut], td[cut:]

        # hard firewall assertions on the realised split
        assert str(pd.Timestamp(tr_td.max()).date()) < fx.PRE_TEST_CUTOFF
        assert str(pd.Timestamp(va_td.max()).date()) < fx.PRE_TEST_CUTOFF

        windows.append({
            "ticker": sym,
            "raw_first_date": raw_first,
            "first_eligible_date": first_elig,
            "final_date": final,
            "n_eligible": len(y),
            "n_train": len(tr_y), "n_val": len(va_y),
            "train_first_target_date": str(pd.Timestamp(tr_td[0]).date()),
            "train_last_target_date": str(pd.Timestamp(tr_td.max()).date()),
            "val_first_target_date": str(pd.Timestamp(va_td[0]).date()),
            "val_last_target_date": str(pd.Timestamp(va_td.max()).date()),
        })
        if not len(va_y):
            continue

        # AUTHOR-CONFIRMED model settings, unchanged. Only the epoch/feature
        # configuration is fixed; no new hyperparameter is introduced.
        seed_everything(ANCHOR["seed"])
        model = AttentionLSTM(input_size=X.shape[-1], hidden_size=64,
                              num_layers=2, dropout=ANCHOR["dropout"])
        tr = Trainer(model, learning_rate=0.001, beta1=0.9, beta2=0.999,
                     weight_decay=ANCHOR["weight_decay"], batch_size=64,
                     max_epochs=10, patience=10, restore_best_checkpoint=True,
                     device=device)
        fit = tr.fit(X[:cut], tr_y, X[cut:], va_y)
        p = _torch_proba(model, X[cut:])
        rows.append({
            "ticker": sym, "label": fx.ANALOG_85_15_LABEL,
            "accuracy": _acc(va_y, p), "f1": _f1(va_y, p),
            "brier": _brier(va_y, p),
            "train_majority_baseline_accuracy": float(
                max(np.mean(tr_y), 1 - np.mean(tr_y))),
            "validation_oracle_majority_rate": float(
                max(np.mean(va_y), 1 - np.mean(va_y))),
            "train_positive_rate": float(np.mean(tr_y)),
            "val_positive_rate": float(np.mean(va_y)),
            "n_train": len(tr_y), "n_val": len(va_y),
            "first_eligible_date": first_elig,
            "best_epoch": fit.best_epoch,
        })
    df = pd.DataFrame(rows)
    marker = {"LABEL": fx.ANALOG_85_15_LABEL,
              "VALID_FOR_FINAL_MODEL": False,
              "SCIENTIFICALLY_INVALID": False,
              "NOTE": "alternate paper protocol diagnostic"}
    fx.write_forensic_csv(rows,
                          fx.repo_forensic_dir() / "pre2022_long_history_85_15.csv",
                          marker=marker)
    fx.write_forensic_csv(windows,
                          fx.repo_forensic_dir() / "pre2022_long_history_windows.csv")
    agg = {
        "label": fx.ANALOG_85_15_LABEL,
        "accuracy": float(df["accuracy"].mean()) if len(df) else None,
        "f1": float(df["f1"].mean()) if len(df) else None,
        "brier": float(df["brier"].mean()) if len(df) else None,
        "train_majority_baseline_accuracy": float(
            df["train_majority_baseline_accuracy"].mean()) if len(df) else None,
        "validation_oracle_majority_rate": float(
            df["validation_oracle_majority_rate"].mean()) if len(df) else None,
        "earliest_eligible_date": min((w["first_eligible_date"] for w in windows
                                       if w.get("first_eligible_date")), default=None),
        "latest_final_date": max((w["final_date"] for w in windows
                                  if w.get("final_date")), default=None),
        "n_tickers": len(df),
        "max_validation_target_date": max((w["val_last_target_date"] for w in windows
                                           if w.get("val_last_target_date")),
                                          default=None),
    }
    fx.write_forensic_json({"diagnostic": "D_LONG_HISTORY_85_15", **agg,
                            "windows": windows},
                           fx.repo_forensic_dir() / "pre2022_long_history_85_15.json",
                           marker=marker)
    return {"label": fx.ANALOG_85_15_LABEL, "aggregate": agg,
            "windows": windows, "per_ticker": df.to_dict("records")}


# ---------------------------------------------------------------------------
# Diagnostic E - target alignment audit
# ---------------------------------------------------------------------------

def diagnostic_e(base_path: str, device: str) -> dict:
    """Prove next-day target construction from raw OHLCV, >=100 samples/fold."""
    print("[E] target alignment audit ...", flush=True)
    all_rows, total_mismatch = [], 0
    for fold in SEARCH_FOLDS:
        cfg = build_config(base_path, fold)
        da = DataAgent(cfg)
        for sym in AUDIT_TICKERS:
            ds = da.run(sym)
            fx.assert_targets_pre_test(ds.train.target_dates,
                                       ds.val.target_dates, where=f"{fold}/{sym}")
            raw = _raw_frame(cfg, sym)
            rows, mm = fx.validate_target_alignment(
                ds, raw, fold=fold, ticker=sym, n_samples=100)
            all_rows.extend([asdict_row(r) for r in rows])
            total_mismatch += mm
    fx.write_forensic_csv(all_rows,
                          fx.repo_forensic_dir() / "target_alignment_audit.csv")
    out = {"diagnostic": "E_TARGET_ALIGNMENT",
           "n_samples": len(all_rows),
           "mismatch_count": total_mismatch,
           "tickers": AUDIT_TICKERS, "folds": SEARCH_FOLDS,
           "expected_mismatch_count": 0}
    fx.write_forensic_json(out, fx.repo_forensic_dir() / "target_alignment_audit.json")
    return out


def asdict_row(r) -> dict:
    return {k: getattr(r, k) for k in r.__dataclass_fields__}


def _raw_frame(cfg: dict, ticker: str) -> pd.DataFrame:
    """Raw UNTRANSFORMED Yahoo OHLCV, for independent target re-derivation."""
    path = Path(cfg["data"]["raw_root"]) / f"{ticker}.csv"
    df = pd.read_csv(path)
    if "date" not in df.columns and "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    return df


# ---------------------------------------------------------------------------
# Diagnostic L - invalid leakage probes
# ---------------------------------------------------------------------------

def diagnostic_l(base_path: str, device: str) -> dict:
    """Deliberately WRONG probes (L0-L5). Explain, never endorse.

    Every artifact is stamped VALID_FOR_FINAL_MODEL=false and
    SCIENTIFICALLY_INVALID=true (except L0, the correct control, and L5, which
    does not change training at all).
    """
    print("[L] invalid leakage probes ...", flush=True)
    out = {}
    for probe in ("L0", "L1", "L2", "L3", "L4", "L5"):
        rows = _run_probe(base_path, device, probe)
        invalid = probe in ("L1", "L2", "L3", "L4")
        marker = dict(fx.INVALID_MARKER) if invalid else {
            "VALID_FOR_FINAL_MODEL": True, "SCIENTIFICALLY_INVALID": False}
        if probe == "L0":
            marker["NOTE"] = "correct control"
        if probe == "L5":
            marker["NOTE"] = "reports TRAIN vs VALIDATION side by side; training unaltered"
        out[probe] = {"invalid": invalid, "rows": rows}
        fx.write_forensic_csv(rows,
                              fx.forensic_dir("invalid_leakage_probes") / f"{probe}.csv",
                              marker=marker | {"PROBE": probe})
    agg = {p: {"mean_validation_accuracy": float(
                   pd.DataFrame(v["rows"])["validation_accuracy"].mean()),
               "mean_train_accuracy": float(
                   pd.DataFrame(v["rows"])["train_accuracy"].mean()),
               "scientifically_invalid": v["invalid"]}
           for p, v in out.items()}
    fx.write_forensic_json({"diagnostic": "L_INVALID_PROBES", "probes": agg},
                           fx.repo_forensic_dir() / "invalid_leakage_probes.json",
                           marker={"NOTE": "L1-L4 are INTENTIONALLY WRONG and exist "
                                           "only to explain possible historical "
                                           "reporting artefacts. They are never valid models."})
    return out


def _run_probe(base_path: str, device: str, probe: str) -> list[dict]:
    rng = np.random.default_rng(0)
    rows: list[dict] = []
    l1_audit: list[dict] = []
    for fold in SEARCH_FOLDS:
        cfg = build_config(base_path, fold)
        da = DataAgent(cfg)
        for sym in PROBE_TICKERS:
            ds = da.run(sym)
            fx.assert_targets_pre_test(ds.train.target_dates,
                                       ds.val.target_dates, where=f"{probe}/{fold}/{sym}")
            ytr = np.asarray(ds.train.y, int)
            yva = np.asarray(ds.val.y, int)
            Xtr, Xva = ds.train.X, ds.val.X

            if probe == "L1":
                # target day GENUINELY included as the final timestep.
                # The naive left-shift left the final row as the ORIGIN, so it
                # leaked nothing; the mapping below makes the last element the
                # real target-day feature vector.
                def _l1(X, y, dates, tdates):
                    return fx.build_target_day_leak(X, y, dates, tdates)

                if probe == "L1" and len(ds.val.y):
                    tr1 = _l1(Xtr, ytr, ds.train.dates, ds.train.target_dates)
                    va1 = _l1(Xva, yva, ds.val.dates, ds.val.target_dates)
                    Xtr, ytr = tr1["X_leaked"], tr1["y"]
                    Xva, yva = va1["X_leaked"], va1["y"]
                    l1_audit.append({
                        "fold": fold, "ticker": sym,
                        "n_train_kept": tr1["n_kept"],
                        "n_val_kept": va1["n_kept"],
                        "n_dropped": tr1["n_dropped"] + va1["n_dropped"],
                        "final_timestep_equals_target_date": all(
                            d == t for d, t in
                            zip(va1["final_timestep_dates"], va1["target_dates"])),
                        "target_after_origin": all(
                            t > o for t, o in
                            zip(va1["target_dates"], va1["origin_dates"])),
                    })
            elif probe == "L2":
                # current-day direction as target: features contain t
                ytr = _current_day_target(cfg, sym, ds.train.dates)
                yva = _current_day_target(cfg, sym, ds.val.dates)
            elif probe == "L3":
                # random overlapping-sequence split
                Xtr, ytr, Xva, yva = _random_split(Xtr, ytr, Xva, yva, rng)
            elif probe == "L4":
                # Deliberate scaler leakage: refit on train+validation. This
                # REQUIRES unscaled sequences -- rescaling already-scaled data
                # would not be a real leak. DataAgent exposes the original
                # feature values for exactly this purpose.
                ds_unscaled = DataAgent(cfg).run(
                    sym, return_unscaled_sequences=True)
                fx.assert_targets_pre_test(
                    ds_unscaled.train.target_dates,
                    ds_unscaled.val.target_dates, where=f"L4/{fold}/{sym}")
                Xtr = np.asarray(ds_unscaled.train.X, dtype=np.float64)
                Xva = np.asarray(ds_unscaled.val.X, dtype=np.float64)
                ytr = np.asarray(ds_unscaled.train.y, int)
                yva = np.asarray(ds_unscaled.val.y, int)
                assert not ds_unscaled.scaled, "L4 needs unscaled sequences"
                Xtr, Xva = _leak_scaler(Xtr, Xva)

            # L4 is evaluated with a distance-sensitive model: ordinary linear
            # rescaling cannot move a tree's split points, so a RandomForest
            # would make the probe look artificially inert.
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=1000)
            clf.fit(Xtr.reshape(len(Xtr), -1), ytr)
            p_tr = clf.predict_proba(Xtr.reshape(len(Xtr), -1))[:, 1]
            p_va = clf.predict_proba(Xva.reshape(len(Xva), -1))[:, 1]
            rows.append({"probe": probe, "fold": fold, "ticker": sym,
                         "model": "LogisticRegression",
                         "train_accuracy": _acc(ytr, p_tr),
                         "validation_accuracy": _acc(yva, p_va),
                         "validation_f1": _f1(yva, p_va)})
    if probe == "L1":
        # expose the L1 alignment audit alongside the metrics
        for r in rows:
            r["n_l1_kept"] = next((a["n_val_kept"] for a in l1_audit
                                   if a["fold"] == r["fold"]
                                   and a["ticker"] == r["ticker"]), None)
            r["l1_final_timestep_equals_target_date"] = next(
                (a["final_timestep_equals_target_date"] for a in l1_audit
                 if a["fold"] == r["fold"] and a["ticker"] == r["ticker"]), None)
    return rows


# NOTE: the previous `_leak_target_day` shifted the window left and left the
# FINAL timestep as the origin row, so the target day never entered the input
# and the probe measured nothing. It was replaced by
# `fx.build_target_day_leak`, which appends the genuine target-day vector.


def _current_day_target(cfg, ticker, dates):
    """y = Close[t] > Close[t-1]: the current day's direction, already inside
    the features as a log return. INVALID."""
    raw = _raw_frame(cfg, ticker)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values("date").reset_index(drop=True)
    up = (raw["Close"].diff() > 0).astype(int)
    lookup = {d: int(up.iloc[i]) for i, d in enumerate(raw["date"])}
    return np.array([lookup.get(pd.Timestamp(d), 0) for d in dates], dtype=int)


def _random_split(Xtr, ytr, Xva, yva, rng):
    """INVALID probe L3: pool the sequences, then split RANDOMLY.

    This is exactly the time-series error being demonstrated -- sequences are
    built first, overlapping windows are scattered across the split, and
    neighbouring (near-identical) samples end up on both sides.
    """
    X = np.concatenate([Xtr, Xva])
    y = np.concatenate([ytr, yva])
    idx = np.arange(len(y))
    rng.shuffle(idx)
    half = len(idx) // 2
    return X[idx[:half]], y[idx[:half]], X[idx[half:]], y[idx[half:]]


def _leak_scaler(Xtr, Xva):
    """INVALID: fit StandardScaler on train+validation, then apply to both.

    Takes ORIGINAL (unscaled) feature values. Fitting on already-scaled data
    would produce a near-identity transform and hide the leak entirely, so the
    caller must pass sequences obtained with
    ``return_unscaled_sequences=True``.
    """
    from sklearn.preprocessing import StandardScaler
    tr2, va2 = Xtr.reshape(len(Xtr), -1), Xva.reshape(len(Xva), -1)
    sc = StandardScaler().fit(np.concatenate([tr2, va2]))
    return (sc.transform(tr2).reshape(Xtr.shape).astype(np.float32),
            sc.transform(va2).reshape(Xva.shape).astype(np.float32))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

DIAGNOSTICS = ("a", "b", "c", "d", "e", "l")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config",
                    default="configs/reproduction_search/paper_snapshot_2025_unadjusted_perf.yaml")
    ap.add_argument("--diagnostic", default="all",
                    help="a/b/c/d/e/l or all")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fx.forensic_dir()
    fx.repo_forensic_dir()
    want = DIAGNOSTICS if args.diagnostic == "all" else (args.diagnostic.lower(),)
    results: dict = {}
    try:
        if "a" in want:
            results["A_adjusted"] = diagnostic_a(args.config, args.device)
        if "b" in want:
            results["B_baseline_panel"] = diagnostic_b(args.config, args.device)
        if "c" in want:
            results["C_pooled"] = diagnostic_c(args.config, args.device)
        if "d" in want:
            results["D_85_15"] = diagnostic_d(args.config, args.device)
        if "e" in want:
            results["E_alignment"] = diagnostic_e(args.config, args.device)
        if "l" in want:
            results["L_invalid_probes"] = diagnostic_l(args.config, args.device)
    except fx.ForensicFirewallAbort as exc:
        print(f"\n*** FORENSIC RUN ABORTED: protected date encountered: {exc}",
              file=sys.stderr)
        fx.write_forensic_json({"aborted": True, "reason": str(exc)},
                               fx.repo_forensic_dir() / "ABORTED.json",
                               marker={"VALID_FOR_FINAL_MODEL": False})
        return 2
    out = Path(args.out) if args.out else fx.repo_forensic_dir() / "forensics_raw.json"
    fx.write_forensic_json(results, out)
    print(f"\nforensic artifacts -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
