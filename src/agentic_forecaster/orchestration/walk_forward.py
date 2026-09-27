"""Paper walk-forward reproduction (items 13-17).

Executes the paper's two folds, retraining EVERY ticker between folds, and
writes the full aggregate reproduction output set under::

    $AGENTIC_OUTPUT_ROOT/reproduction/<run_id>/

        manifest.json
        ticker_metrics.csv
        aggregate_metrics.json
        predictions.csv.gz
        calibration_metrics.csv
        precision_at_3.csv
        p3_daily_selections.csv
        baseline_metrics.csv
        ablation_metrics.csv
        paper_comparison.csv
        training_summary.json
        figures/
        reports/
"""

from __future__ import annotations

import copy
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from agentic_forecaster.config import get_env_roots
from agentic_forecaster.data.agent import DataAgent
from agentic_forecaster.evaluation.metrics import precision_at_3_cross_sectional
from agentic_forecaster.models.checkpoint import save_model_bundle
from agentic_forecaster.orchestration.pipeline import Pipeline
from agentic_forecaster.utils import atomic_json_dump, ensure_dir, seed_everything

logger = logging.getLogger("agentic_forecaster.orchestration.walk_forward")

# PAPER-DEFINED folds.
PAPER_FOLDS = [
    {
        "fold": "fold_0",
        "train_start": "2016-01-01", "train_end": "2020-12-31",
        "val_start": "2021-01-01", "val_end": "2021-12-31",
        "test_start": "2022-01-01", "test_end": "2022-12-31",
    },
    {
        "fold": "fold_1",
        "train_start": "2016-01-01", "train_end": "2021-12-31",
        "val_start": "2022-01-01", "val_end": "2022-12-31",
        "test_start": "2023-01-01", "test_end": "2023-12-31",
    },
]

# Metrics averaged (unweighted) across tickers into aggregate_metrics.json.
AGGREGATE_KEYS = (
    "accuracy", "f1", "roc_auc",
    "brier_raw", "brier_calibrated", "ece_raw", "ece_calibrated",
)


def _default_tickers(config: dict, data_agent: DataAgent) -> list[str]:
    """Ticker list to reproduce when the caller does not restrict it.

    Synthetic (demo/test) configs have no raw dataset, so the generated
    symbols are used instead of the configured NIFTY-50 universe.
    """
    if config.get("data", {}).get("synthetic"):
        from agentic_forecaster.data.dataset import make_synthetic_dataset

        frames = make_synthetic_dataset(
            n_tickers=int(config["data"].get("n_tickers", 3)),
            n_days=int(config["data"].get("n_days", 600)),
            seed=int(config.get("experiment", {}).get("seed", 42)),
        )
        return sorted(frames)
    universe = data_agent.universe()
    return sorted(universe.available.values())


def _fold_config(config: dict, fold: dict) -> dict:
    out = copy.deepcopy(config)
    out["data"].update({
        "train_start": fold["train_start"], "train_end": fold["train_end"],
        "val_start": fold["val_start"], "val_end": fold["val_end"],
        "test_start": fold["test_start"], "test_end": fold["test_end"],
    })
    return out


def run_walk_forward(
    config: dict,
    device: str | None = None,
    run_id: str | None = None,
    tickers: list[str] | None = None,
    n_reports: int = 1,
) -> dict:
    """Run the paper reproduction and write every aggregate artefact."""
    seed_everything(int(config.get("experiment", {}).get("seed", 42)))
    roots = get_env_roots()
    run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ensure_dir(Path(roots["AGENTIC_OUTPUT_ROOT"]) / "reproduction" / run_id)
    (out_dir / "figures").mkdir(exist_ok=True)
    (out_dir / "reports").mkdir(exist_ok=True)
    model_root = ensure_dir(Path(roots["AGENTIC_MODEL_ROOT"]))

    base_agent = DataAgent(config)
    selected = tickers or _default_tickers(config, base_agent)
    logger.info("Reproduction run_id=%s tickers=%d", run_id, len(selected))

    all_predictions: list[pd.DataFrame] = []
    ticker_rows: list[dict] = []
    baseline_rows: list[dict] = []
    ablation_rows: list[dict] = []
    calibration_rows: list[dict] = []
    training_summary: dict = {}
    status: dict = {}

    for fold in PAPER_FOLDS:
        fold_name = fold["fold"]
        fold_cfg = _fold_config(config, fold)
        # Direct this run's reports into the run directory, not the generic
        # config output_dir, so the run directory is self-contained.
        fold_cfg["experiment"]["output_dir"] = str(out_dir)
        logger.info("=== %s ===", fold_name)

        for idx, ticker in enumerate(selected, 1):
            logger.info("[%d/%d] %s (%s)", idx, len(selected), ticker, fold_name)
            try:
                result = Pipeline(fold_cfg, fold=fold_name).run(
                    ticker, device=device, n_reports=n_reports,
                    report_dir=out_dir / "reports",
                )
            except Exception as exc:
                logger.warning("  %s/%s unavailable: %s", ticker, fold_name, exc)
                status[f"{ticker}/{fold_name}"] = {
                    "status": "unavailable", "reason": str(exc)
                }
                continue

            save_model_bundle(
                result.primary,
                model_root / "trained" / ticker / fold_name,
            )

            all_predictions.append(result.predictions)
            m = result.primary.metrics
            ticker_rows.append({
                "ticker": ticker, "fold": fold_name,
                "n_test": len(result.dataset.test.y),
                "temperature": result.primary.temperature,
                **{k: m.get(k) for k in AGGREGATE_KEYS},
            })
            calibration_rows.append({
                "ticker": ticker, "fold": fold_name,
                "temperature": result.primary.temperature,
                "brier_raw": m.get("brier_raw"),
                "brier_calibrated": m.get("brier_calibrated"),
                "ece_raw": m.get("ece_raw"),
                "ece_calibrated": m.get("ece_calibrated"),
            })
            for name, fm in result.baselines.items():
                baseline_rows.append({
                    "ticker": ticker, "fold": fold_name, "model": name,
                    **{k: fm.metrics.get(k) for k in AGGREGATE_KEYS},
                })
            for variant, values in (result.ablations or {}).items():
                if not isinstance(values, dict) or variant.startswith("_"):
                    continue
                ablation_rows.append({
                        "ticker": ticker, "fold": fold_name, "variant": variant,
                        **{k: values.get(k) for k in AGGREGATE_KEYS},
                        "extra": json.dumps(
                            {k: v for k, v in values.items() if k not in AGGREGATE_KEYS},
                            default=str,
                        ),
                    })
            training_summary[f"{ticker}/{fold_name}"] = {
                "config": result.primary.train_config,
                "history": result.primary.history,
            }
            status[f"{ticker}/{fold_name}"] = {"status": "trained"}

    predictions = (
        pd.concat(all_predictions, ignore_index=True) if all_predictions
        else pd.DataFrame()
    )

    # ---- item 15: cross-sectional Precision@3 computed BY DATE ------------
    # The paper's P@3 ranks tickers within a single date by their UP
    # probability.  We rank on the CALIBRATED probability (the default
    # forecast) and keep the raw probability in the table for audit.
    p3_rows: list[dict] = []
    selection_rows: list[dict] = []
    if not predictions.empty:
        for fold_name, group in predictions.groupby("fold"):
            frame = group.rename(columns={"calibrated_p_up": "p_up"})
            p3 = precision_at_3_cross_sectional(frame, k=3)
            p3_rows.append({
                "fold": fold_name,
                "precision_at_3_up": p3["precision_at_3_up"],
                "precision_at_3_down": p3["precision_at_3_down"],
                "n_dates": p3["n_dates"],
                "probability_used": "calibrated_p_up",
            })
            for sel in p3["selections"]:
                selection_rows.append({"fold": fold_name, **sel})

    # ---- aggregate metrics (documented policy: unweighted mean) -----------
    ticker_df = pd.DataFrame(ticker_rows)
    baseline_df = pd.DataFrame(baseline_rows)
    aggregate: dict = {
        "policy": "unweighted mean of per-ticker test-split metrics",
        "n_ticker_fold_runs": len(ticker_df),
    }
    for key in AGGREGATE_KEYS:
        if key in ticker_df:
            aggregate[key] = float(ticker_df[key].mean(skipna=True))

    # ---- paper comparison -------------------------------------------------
    comparison = _paper_comparison(ticker_df, p3_rows, baseline_df)

    # ---- figures from the real run artefacts ------------------------------
    from agentic_forecaster.evaluation.figures import build_run_figures

    figure_paths = build_run_figures(
        predictions, comparison, p3_rows, out_dir / "figures", run_id
    )

    # ---- write everything ------------------------------------------------
    ticker_df.to_csv(out_dir / "ticker_metrics.csv", index=False)
    atomic_json_dump(aggregate, out_dir / "aggregate_metrics.json")
    if not predictions.empty:
        predictions.to_csv(out_dir / "predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(calibration_rows).to_csv(out_dir / "calibration_metrics.csv", index=False)
    pd.DataFrame(p3_rows).to_csv(out_dir / "precision_at_3.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(out_dir / "p3_daily_selections.csv", index=False)
    baseline_df.to_csv(out_dir / "baseline_metrics.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(out_dir / "ablation_metrics.csv", index=False)
    pd.DataFrame(comparison).to_csv(out_dir / "paper_comparison.csv", index=False)
    atomic_json_dump(training_summary, out_dir / "training_summary.json")
    atomic_json_dump(status, out_dir / "ticker_status.json")
    atomic_json_dump(
        {
            "run_id": run_id,
            "created_utc": datetime.now(UTC).isoformat(),
            "config_name": config.get("experiment", {}).get("name"),
            "device": device,
            "seed": int(config.get("experiment", {}).get("seed", 42)),
            "folds": PAPER_FOLDS,
            "n_tickers_requested": len(selected),
            "n_ticker_fold_runs": len(ticker_df),
            "figures": [Path(f).name for f in figure_paths],
            "outputs": sorted(p.name for p in out_dir.iterdir()),
        },
        out_dir / "manifest.json",
    )

    logger.info("Reproduction complete -> %s", out_dir)
    return {
        "run_id": run_id,
        "run_dir": str(out_dir),
        "run_dir_env": f"$AGENTIC_OUTPUT_ROOT/reproduction/{run_id}",
        "n_ticker_fold_runs": len(ticker_df),
        "aggregate": aggregate,
        "precision_at_3": p3_rows,
    }


def _paper_comparison(
    ticker_df: pd.DataFrame,
    p3_rows: list[dict],
    baseline_df: pd.DataFrame,
) -> list[dict]:
    """Join reproduced values against the PAPER REFERENCE values."""
    from agentic_forecaster.paper_reference import PAPER_REFERENCE

    def _mean(df: pd.DataFrame, model: str) -> float | None:
        if df.empty or model not in set(df["model"]):
            return None
        values = df.loc[df["model"] == model, "accuracy"].dropna()
        return float(values.mean()) if len(values) else None

    reproduced = {
        "attention_lstm_calibrated": (
            float(ticker_df["accuracy"].dropna().mean())
            if not ticker_df.empty and "accuracy" in ticker_df else None
        ),
        "attention_lstm_raw": "raw_brier",
        "plain_lstm": _mean(baseline_df, "lstm"),
        "random_forest": _mean(baseline_df, "random_forest"),
    }

    rows: list[dict] = []

    # P@3 aggregate: UNWEIGHTED MEAN of the fold-level values (each fold is a
    # distinct test year, so weighting by date count would let one year
    # dominate).  Fold-level values are kept alongside for auditability.
    def _p3_mean(field: str) -> float | None:
        values = [r[field] for r in p3_rows if r.get(field) is not None]
        return float(sum(values) / len(values)) if values else None

    p3_up_mean = _p3_mean("precision_at_3_up")
    p3_down_mean = _p3_mean("precision_at_3_down")

    for model, values in PAPER_REFERENCE.items():
        ref_acc = values.get("accuracy")
        if model == "attention_lstm_raw":
            # Accuracy is invariant under a scalar temperature, so the raw
            # variant shares the calibrated accuracy; the Brier differs.
            rep_acc = reproduced["attention_lstm_calibrated"]
        else:
            rep_acc = reproduced.get(model)
        rep_brier = None
        if not ticker_df.empty and model == "attention_lstm_calibrated" \
                and "brier_calibrated" in ticker_df:
            rep_brier = float(ticker_df["brier_calibrated"].dropna().mean())
        elif not ticker_df.empty and model == "attention_lstm_raw" \
                and "brier_raw" in ticker_df:
            rep_brier = float(ticker_df["brier_raw"].dropna().mean())
        rows.append({
            "model": model,
            "accuracy_paper_reference": ref_acc,
            "accuracy_reconstructed": rep_acc,
            "accuracy_difference": (
                (rep_acc - ref_acc) if (rep_acc is not None and ref_acc is not None) else None
            ),
            "brier_paper_reference": values.get("brier"),
            "brier_reconstructed": rep_brier,
            "f1_paper_reference": values.get("f1"),
            "precision_at_3_up_paper_reference": values.get("precision_at_3_up"),
            "precision_at_3_down_paper_reference": values.get("precision_at_3_down"),
            "precision_at_3_up_reconstructed": p3_up_mean,
            "precision_at_3_down_reconstructed": p3_down_mean,
            "p3_aggregate_policy": "unweighted mean of fold-level P@3 "
                                   "across all folds",
            "p3_folds_included": ",".join(r.get("fold", "") for r in p3_rows),
            "precision_at_3_up_fold_0": next(
                (r["precision_at_3_up"] for r in p3_rows if r["fold"] == "fold_0"), None
            ),
            "precision_at_3_down_fold_0": next(
                (r["precision_at_3_down"] for r in p3_rows if r["fold"] == "fold_0"), None
            ),
            "precision_at_3_up_fold_1": next(
                (r["precision_at_3_up"] for r in p3_rows if r["fold"] == "fold_1"), None
            ),
            "precision_at_3_down_fold_1": next(
                (r["precision_at_3_down"] for r in p3_rows if r["fold"] == "fold_1"), None
            ),
            "provenance": "paper_reference values are transcribed from the publication",
        })
    return rows
