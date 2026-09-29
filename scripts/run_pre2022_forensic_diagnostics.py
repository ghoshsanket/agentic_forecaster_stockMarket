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
                              "majority_accuracy": bal["validation"]["majority_accuracy"]})
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
            tr_majority = 1 if ytr.mean() >= 0.5 else 0
            # majority baseline: always predict the TRAIN majority class
            rows.append(_row(fold, sym, "Majority",
                             _acc(yva, np.full(len(yva), float(tr_majority))),
                             _f1(yva, np.full(len(yva), float(tr_majority))),
                             _brier_const(ytr, yva, tr_majority)))

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
    maj = agg.get("Majority", {}).get("accuracy")
    for a in agg.values():
        a["delta_vs_majority"] = None if maj is None else a["accuracy"] - maj
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
    fx.write_forensic_csv(rows, fx.repo_forensic_dir() / "pooled_model.csv",
                          marker=fx.INVALID_MARKER | {"NOTE": fx.POOLED_LABEL})
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
    """85/15 chronological split on data truncated at 2021-12-31.

    NOT the official walk-forward protocol. Tests whether much longer history
    or the 85:15 procedure materially changes performance.
    """
    print("[D] pre-2022 85/15 analog ...", flush=True)

    rows, windows = [], []
    for sym in TICKERS:
        # fold C is the closest pre-test analogue; use its raw source frame
        cfg = build_config(base_path, "SEARCH_FOLD_C")
        da = DataAgent(cfg)
        ds = da.run(sym)
        # 85/15 over the source origin dates, truncated at the cutoff
        tr_d, va_d, last = fx.pre2022_85_15_split(ds.val.dates)
        # the true split uses all pre-2022 origin dates, so rebuild from the
        # raw frame: reuse the largest fold window and apply 85/15 to it
        all_dates = list(ds.train.dates) + list(ds.val.dates)
        tr_d, va_d, last = fx.pre2022_85_15_split(all_dates)
        windows.append({"ticker": sym, "n_train_dates": len(tr_d),
                        "n_val_dates": len(va_d),
                        "train_first": tr_d[0] if tr_d else None,
                        "train_last": tr_d[-1] if tr_d else None,
                        "val_first": va_d[0] if va_d else None,
                        "val_last": last})
        if not va_d:
            continue
        # Re-run the DataAgent on the 85/15 windows by overriding the split.
        cfg2 = build_config(base_path, "SEARCH_FOLD_C")
        cfg2["data"]["train_start"] = tr_d[0]
        cfg2["data"]["train_end"] = tr_d[-1]
        cfg2["data"]["val_start"] = va_d[0]
        cfg2["data"]["val_end"] = va_d[-1]
        cfg2["data"]["test_start"] = None
        cfg2["data"]["test_end"] = None
        ds2 = DataAgent(cfg2).run(sym)
        fx.assert_targets_pre_test(ds2.train.target_dates,
                                   ds2.val.target_dates, where=f"8515/{sym}")
        seed_everything(ANCHOR["seed"])
        fitted = ModelAgent(cfg2).train_ticker(sym, ds2, fold="ANALOG_85_15",
                                               device=device)
        p = fitted.predict_proba(ds2.val.X)
        yva = np.asarray(ds2.val.y, int)
        rows.append({"ticker": sym, "label": fx.ANALOG_85_15_LABEL,
                     "accuracy": _acc(yva, p), "f1": _f1(yva, p),
                     "brier": _brier(yva, p),
                     "majority_accuracy": float(max(yva.mean(), 1 - yva.mean())),
                     "n_train": len(ds2.train.y),
                     "n_val": len(yva),
                     "best_epoch": fitted.train_config.get("best_epoch")})
    df = pd.DataFrame(rows)
    fx.write_forensic_csv(rows, fx.repo_forensic_dir() / "pre2022_85_15.csv",
                          marker={"LABEL": fx.ANALOG_85_15_LABEL,
                                  "VALID_FOR_FINAL_MODEL": False,
                                  "SCIENTIFICALLY_INVALID": False,
                                  "NOTE": "diagnostic analog; NOT the official walk-forward protocol"})
    fx.write_forensic_csv(windows, fx.repo_forensic_dir() / "pre2022_85_15_windows.csv")
    agg = {"accuracy": float(df["accuracy"].mean()) if len(df) else None,
           "f1": float(df["f1"].mean()) if len(df) else None,
           "brier": float(df["brier"].mean()) if len(df) else None,
           "majority_accuracy": float(df["majority_accuracy"].mean()) if len(df) else None,
           "n_tickers": len(df)}
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
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(0)
    rows = []
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
                # target day INCLUDED in the input window
                Xtr = _leak_target_day(Xtr, ds)
                Xva = _leak_target_day(Xva, ds)
            elif probe == "L2":
                # current-day direction as target: features contain t
                ytr = _current_day_target(cfg, sym, ds.train.dates)
                yva = _current_day_target(cfg, sym, ds.val.dates)
            elif probe == "L3":
                # random overlapping-sequence split
                Xtr, ytr, Xva, yva = _random_split(Xtr, ytr, Xva, yva, rng)
            elif probe == "L4":
                # scaler fit on train+validation
                Xtr, Xva = _leak_scaler(Xtr, Xva)

            clf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                         random_state=ANCHOR["seed"])
            clf.fit(Xtr.reshape(len(Xtr), -1), ytr)
            p_tr = clf.predict_proba(Xtr.reshape(len(Xtr), -1))[:, 1]
            p_va = clf.predict_proba(Xva.reshape(len(Xva), -1))[:, 1]
            rows.append({"probe": probe, "fold": fold, "ticker": sym,
                         "train_accuracy": _acc(ytr, p_tr),
                         "validation_accuracy": _acc(yva, p_va),
                         "validation_f1": _f1(yva, p_va)})
    return rows


def _leak_target_day(X, ds):
    """Append the NEXT row's scaled features to each window (the L1 off-by-one)."""
    # shift each sequence forward by one step so it includes t+1
    out = np.copy(X)
    out[:, :-1, :] = X[:, 1:, :]
    return out


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
    """Refit the scaler on train+validation: INVALID leakage."""
    from sklearn.preprocessing import StandardScaler
    both = np.concatenate([Xtr.reshape(len(Xtr), -1), Xva.reshape(len(Xva), -1)])
    sc = StandardScaler().fit(both)
    return (sc.transform(Xtr.reshape(len(Xtr), -1)).reshape(Xtr.shape),
            sc.transform(Xva.reshape(len(Xva), -1)).reshape(Xva.shape))


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
