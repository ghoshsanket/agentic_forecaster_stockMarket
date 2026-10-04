"""The multi-horizon horizon screen, the neural stage, and the frozen manifest.

STAGE 1 -- CHEAP HORIZON SCREEN
------------------------------
Before any LSTM is trained, two fixed models are run at every horizon and every
development year:

* ``LOGISTIC``               ``LogisticRegression`` on a TRAIN-fitted
  ``StandardScaler``
* ``HIST_GRADIENT_BOOSTING``  ``HistGradientBoostingClassifier`` with recorded,
  fixed settings

No hyper-parameter search happens here.  The screen answers one question: does the
existing PRE-COVID feature set expose more directional information at 3, 5 or 10
trading days than at one day?

STAGE 2 -- NEURAL, ONLY FOR SELECTED HORIZONS
--------------------------------------------
For a horizon that SCREEN-PASSES, the two existing V2 architectures are run
unchanged: the shared LSTM (N1) and the shared LSTM + Transformer (N2).  Context,
multi-task, FiLM and meta-learning are deliberately NOT run: the previous
PRE-COVID programme found the plain shared LSTM strongest, and this programme
exists to test the HORIZON first.

Every gate in this module is a diagnostic threshold, never a target that is
tuned towards.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import horizon_metrics as HM
from . import horizons as HZ
from .dataset import SampleTable, SplitWindow, fit_scalers_on_train
from .horizon_dataset import (
    horizon_split,
    origin_feature_matrix,
)

logger = logging.getLogger("agentic_forecaster.v2.horizon_screen")

#: Fixed, deterministic screen settings.  Recorded in every manifest; never tuned.
LOGISTIC_SETTINGS: dict = {"C": 1.0, "max_iter": 2000, "solver": "lbfgs",
                           "random_state": 42, "penalty": "l2"}

HIST_GRADIENT_BOOSTING_SETTINGS: dict = {
    "loss": "log_loss",
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
    "max_bins": 255,
    "early_stopping": False,
    "random_state": 42,
}

SCREEN_SETTINGS: dict[str, dict] = {
    "LOGISTIC": LOGISTIC_SETTINGS,
    "HIST_GRADIENT_BOOSTING": HIST_GRADIENT_BOOSTING_SETTINGS,
}

#: Fixed training settings of the shared LSTM (unchanged V2-A / V2-B values).
NEURAL_TRAIN_SETTINGS: dict = {
    "optimizer": "AdamW",
    "learning_rate": 3e-4,
    "beta1": 0.9,
    "beta2": 0.999,
    "weight_decay": 1e-4,
    "batch_size": 256,
    "max_epochs": 30,
    "early_stopping_patience": 6,
    "gradient_clip_norm": 1.0,
    "balanced_by_ticker": True,
    "seed": 42,
}

#: Unstable seed spread blocks the lockbox.
MAX_MACRO_STD_ACROSS_SEEDS = 0.015


@dataclass
class ScreenResult:
    """One (horizon, model, fold) screen run."""

    horizon: int
    model: str
    fold: str
    n_train: int
    n_val: int
    predictions: pd.DataFrame
    metrics: dict = field(default_factory=dict)
    common_metrics: dict = field(default_factory=dict)
    non_overlap_metrics: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    @property
    def objective_id(self) -> str:
        return HZ.objective_id(self.horizon)


def _design(samples: SampleTable, mask: np.ndarray, columns) -> np.ndarray:
    return origin_feature_matrix(samples, mask, columns=columns)


def fit_screen_model(model_name: str, x_train: np.ndarray, y_train: np.ndarray, *,
                     settings: dict | None = None) -> tuple[object, dict]:
    """Fit one screen model.  The scaler is fitted on TRAIN ROWS ONLY."""
    settings = dict(settings or SCREEN_SETTINGS[model_name])
    if model_name == "LOGISTIC":
        scaler = StandardScaler().fit(x_train)
        # ``penalty`` is left at its default (l2) and only RECORDED in the settings:
        # passing it explicitly is deprecated in recent scikit-learn versions.
        model = LogisticRegression(
            C=settings["C"], max_iter=settings["max_iter"], solver=settings["solver"],
            random_state=settings["random_state"])
        model.fit(scaler.transform(x_train), y_train)
        return {"scaler": scaler, "model": model}, {
            "settings": settings,
            "scaler": "StandardScaler",
            "scaler_fit_on": "TRAIN_ONLY",
            "n_scaler_rows": int(x_train.shape[0]),
            "n_features": int(x_train.shape[1]),
        }
    if model_name == "HIST_GRADIENT_BOOSTING":
        # A tree ensemble needs no standardisation; the recorded fact that no
        # scaler was fitted is part of the provenance.
        model = HistGradientBoostingClassifier(**settings)
        model.fit(x_train, y_train)
        return {"scaler": None, "model": model}, {
            "settings": settings,
            "scaler": None,
            "scaler_fit_on": "NOT_APPLICABLE_TREE_MODEL",
            "n_features": int(x_train.shape[1]),
        }
    raise ValueError(f"unknown screen model {model_name!r}; "
                     f"expected one of {sorted(SCREEN_SETTINGS)}")


def predict_proba(fitted: dict, x: np.ndarray) -> np.ndarray:
    matrix = x if fitted["scaler"] is None else fitted["scaler"].transform(x)
    return fitted["model"].predict_proba(matrix)[:, 1]


def run_screen_fold(samples: SampleTable, *, horizon: int, window: SplitWindow,
                    model_name: str, common_keys=None,
                    locked: bool = False, settings: dict | None = None,
                    fold: str = "", n_bootstrap: int = HM.DEFAULT_N_BOOTSTRAP,
                    block_length: int = HM.DEFAULT_BLOCK_LENGTH,
                    seed: int = HM.BOOTSTRAP_SEED) -> ScreenResult:
    """Fit on TRAIN, score the validation year, in all three sample views."""
    split = horizon_split(samples, window, fold=fold or window.name, locked=locked,
                          where="screen split")
    frame = samples.frame
    train_frame = frame.loc[split.train]
    val_frame = frame.loc[split.val]
    if train_frame.empty or val_frame.empty:
        raise ValueError(
            f"{HZ.objective_id(horizon)} {model_name} {window.name}: needs both TRAIN "
            f"and validation rows (train={len(train_frame)}, val={len(val_frame)})")

    columns = list(samples.arrays.stock_features)
    x_train = _design(samples, split.train, columns)
    x_val = _design(samples, split.val, columns)
    y_train = train_frame["y_direction"].to_numpy(int)

    fitted, provenance = fit_screen_model(model_name, x_train, y_train, settings=settings)
    p_val = predict_proba(fitted, x_val)

    predictions = val_frame.reset_index(drop=True).copy()
    predictions["y_true"] = predictions["y_direction"].astype(int)
    predictions["p_up"] = p_val
    predictions["model"] = model_name
    predictions["horizon"] = int(horizon)
    predictions["fold"] = window.name

    baseline = float(max(y_train.mean(), 1.0 - y_train.mean()))
    ticker_baselines = HM.ticker_train_majority(train_frame)

    metrics = HM.direction_block(predictions, train_majority=baseline,
                                 ticker_baselines=ticker_baselines, horizon=horizon,
                                 sample_label="all_valid_origins")
    metrics["bootstrap"] = HM.block_bootstrap_ci(
        predictions, block_length=block_length, n_bootstrap=n_bootstrap, seed=seed,
        where=f"{HZ.objective_id(horizon)}/{model_name}/{window.name}")

    common_metrics = _common_origin_metrics(predictions, common_keys, baseline,
                                            ticker_baselines, horizon)

    non_overlap = HM.direction_block(
        predictions.loc[HZ.non_overlap_mask(predictions, horizon=horizon)],
        train_majority=baseline, ticker_baselines=ticker_baselines, horizon=horizon,
        sample_label="non_overlapping")

    return ScreenResult(
        horizon=int(horizon), model=model_name, fold=window.name,
        n_train=len(train_frame), n_val=len(val_frame),
        predictions=predictions, metrics=metrics, common_metrics=common_metrics,
        non_overlap_metrics=non_overlap,
        diagnostics={
            "objective_id": HZ.objective_id(horizon),
            "horizon_phrase": HZ.horizon_phrase(horizon),
            "window": {"train_start": window.train_start, "train_end": window.train_end,
                       "val_start": window.val_start, "val_end": window.val_end},
            "screen_model": provenance,
            "feature_schema": columns,
            "feature_schema_sha256": HZ.hash_payload(columns),
            "target_equation": HZ.TARGET_EQUATION,
            "trading_day_rule": HZ.TRADING_DAY_RULE,
            "non_overlap": HZ.non_overlap_spacing(
                predictions.loc[HZ.non_overlap_mask(predictions, horizon=horizon)],
                horizon=horizon),
            "seed": int(seed),
        })


def _common_origin_metrics(predictions: pd.DataFrame, common_keys, baseline: float,
                           ticker_baselines: dict, horizon: int) -> dict:
    """Score the validation year on the COMMON-ORIGIN subset only.

    ``common_keys`` is the shared (ticker, origin_date) index of the origins that
    are valid at EVERY tested horizon inside the same split, so each horizon is
    measured on exactly the same origins.  This is the comparison that separates
    "this horizon is genuinely easier" from "this horizon was evaluated on a
    different subset".
    """
    if common_keys is None:
        return {}
    index = pd.MultiIndex.from_frame(predictions.loc[:, ["ticker", "origin_date"]])
    common_val = predictions.loc[index.isin(common_keys)]
    if common_val.empty:
        return {}
    out = HM.direction_block(common_val, train_majority=baseline,
                             ticker_baselines=ticker_baselines, horizon=horizon,
                             sample_label="common_origin")
    out["n_dropped_vs_all_valid_origins"] = int(len(predictions) - len(common_val))
    return out


# ---------------------------------------------------------------------------
# neural stage
# ---------------------------------------------------------------------------

def neural_components(model_name: str) -> dict:
    """Component flags for N1 / N2.  Context, multi-task, FiLM stay OFF."""
    if model_name == "SHARED_LSTM":
        return {"use_transformer": False, "use_context": False,
                "use_sector_embedding": False, "use_regime": False,
                "use_multitask": False, "use_film": False, "use_adapter": False}
    if model_name == "LSTM_TRANSFORMER":
        return {"use_transformer": True, "use_context": False,
                "use_sector_embedding": False, "use_regime": False,
                "use_multitask": False, "use_film": False, "use_adapter": False}
    raise ValueError(f"unknown neural model {model_name!r}; "
                     f"expected one of {list(HZ.NEURAL_MODELS)}")


def build_neural_model(model_name: str, arrays, *, seed: int, model_config: dict | None = None):
    """Build the unchanged V2-A / V2-B architecture for this track."""
    from .losses import MultiTaskWeights
    from .model import V2ModelConfig, build_model

    config = V2ModelConfig(
        n_stock_features=arrays.n_stock_features,
        n_context_features=0,
        n_regime_features=0,
        n_tickers=len(arrays.ticker_vocab),
        n_sectors=len(arrays.sector_vocab),
        ticker_vocab=list(arrays.ticker_vocab),
        sector_vocab=list(arrays.sector_vocab),
        **(model_config or {}),
        **neural_components(model_name),
    )
    model = build_model(config, seed=seed)
    return model, MultiTaskWeights(direction=1.0, return_value=0.0, rank=0.0), config


def run_neural_fold(samples: SampleTable, *, horizon: int, window: SplitWindow,
                    model_name: str, seed: int = 42, device: str = "auto",
                    model_config: dict | None = None,
                    train_settings: dict | None = None,
                    common_keys=None,
                    locked: bool = False, n_bootstrap: int = HM.DEFAULT_N_BOOTSTRAP,
                    block_length: int = HM.DEFAULT_BLOCK_LENGTH,
                    fold: str = "") -> ScreenResult:
    """One shared-model fit at one horizon and one year."""
    import torch

    from .dataset import V2SequenceDataset
    from .trainer import V2TrainConfig, V2Trainer

    settings = dict(NEURAL_TRAIN_SETTINGS)
    settings.update(train_settings or {})
    split = horizon_split(samples, window, fold=fold or window.name, locked=locked,
                          where="neural split")
    frame = samples.frame
    train_frame = frame.loc[split.train].reset_index(drop=True)
    val_frame = frame.loc[split.val].reset_index(drop=True)
    if train_frame.empty or val_frame.empty:
        raise ValueError(f"{HZ.objective_id(horizon)} {model_name} {window.name}: "
                         "needs both TRAIN and validation rows")

    scalers = fit_scalers_on_train(samples, split.train)
    train_dataset = V2SequenceDataset(samples, train_frame, scalers=scalers)
    val_dataset = V2SequenceDataset(samples, val_frame, scalers=scalers)

    model, loss_weights, resolved = build_neural_model(
        model_name, samples.arrays, seed=seed, model_config=model_config)
    train_config = V2TrainConfig(
        learning_rate=float(settings["learning_rate"]),
        beta1=float(settings["beta1"]), beta2=float(settings["beta2"]),
        weight_decay=float(settings["weight_decay"]),
        batch_size=int(settings["batch_size"]),
        max_epochs=int(settings["max_epochs"]),
        early_stopping_patience=int(settings["early_stopping_patience"]),
        gradient_clip_norm=float(settings["gradient_clip_norm"]),
        seed=int(seed), balanced_by_ticker=bool(settings["balanced_by_ticker"]))
    trainer = V2Trainer(model, train_config, loss_weights=loss_weights, device=device)
    result = trainer.fit(train_dataset, val_dataset, train_frame=train_frame,
                         val_frame=val_frame)

    probabilities: list[np.ndarray] = []
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=False)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["stock_sequence"].to(trainer.device),
                           batch["ticker_id"].to(trainer.device),
                           context_sequence=batch["context_sequence"].to(trainer.device),
                           sector_id=batch["sector_id"].to(trainer.device),
                           regime_vector=batch["regime_vector"].to(trainer.device),
                           )["direction_logit"].reshape(-1)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    p_val = np.concatenate(probabilities) if probabilities else np.zeros(0)

    predictions = val_frame.copy()
    predictions["y_true"] = predictions["y_direction"].astype(int)
    predictions["p_up"] = p_val
    predictions["model"] = model_name
    predictions["horizon"] = int(horizon)
    predictions["fold"] = window.name

    y_train = train_frame["y_direction"].to_numpy(int)
    baseline = float(max(y_train.mean(), 1.0 - y_train.mean()))
    ticker_baselines = HM.ticker_train_majority(train_frame)

    metrics = HM.direction_block(predictions, train_majority=baseline,
                                 ticker_baselines=ticker_baselines, horizon=horizon,
                                 sample_label="all_valid_origins")
    metrics["bootstrap"] = HM.block_bootstrap_ci(
        predictions, block_length=block_length, n_bootstrap=n_bootstrap, seed=seed,
        where=f"{HZ.objective_id(horizon)}/{model_name}/{window.name}")

    common_metrics = _common_origin_metrics(predictions, common_keys, baseline,
                                            ticker_baselines, horizon)

    non_overlap = HM.direction_block(
        predictions.loc[HZ.non_overlap_mask(predictions, horizon=horizon)],
        train_majority=baseline, ticker_baselines=ticker_baselines, horizon=horizon,
        sample_label="non_overlapping")

    complexity = {
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "best_epoch": int(result.best_epoch),
        "epochs_run": int(result.epochs_run),
        "training_seconds": float(result.train_seconds),
        "stopped_early": bool(result.stopped_early),
        "best_val_total_loss": float(result.best_val_total_loss),
    }
    return ScreenResult(
        horizon=int(horizon), model=model_name, fold=window.name,
        n_train=len(train_frame), n_val=len(val_frame),
        predictions=predictions, metrics=metrics, common_metrics=common_metrics,
        non_overlap_metrics=non_overlap,
        diagnostics={
            "objective_id": HZ.objective_id(horizon),
            "horizon_phrase": HZ.horizon_phrase(horizon),
            "window": {"train_start": window.train_start, "train_end": window.train_end,
                       "val_start": window.val_start, "val_end": window.val_end},
            "architecture": HZ.NEURAL_LABELS[model_name],
            "model_config": resolved.to_dict(),
            "components": neural_components(model_name),
            "train_settings": settings,
            "loss_weights": loss_weights.to_dict(),
            "scaler_provenance": scalers.stock.fit_provenance,
            "training_history": result.history,
            "complexity": complexity,
            "seed": int(seed),
            "not_run_in_this_programme": ["CONTEXT", "MULTITASK", "FILM", "META_LEARNING"],
        })


# ---------------------------------------------------------------------------
# screening gate, ranking and the neural gate
# ---------------------------------------------------------------------------

def evaluate_horizon_gate(natural: dict, common: dict, non_overlap: dict, *,
                          thresholds: dict) -> dict:
    """Apply the HORIZON SCREENING GATE of section 18.

    A candidate horizon passes only when EVERY criterion holds on the natural
    sample set, when the common-origin comparison stays directionally positive and
    when the non-overlapping sensitivity does not collapse back to chance.
    """
    min_auc = float(thresholds["min_mean_roc_auc"])
    min_balanced = float(thresholds["min_mean_balanced_accuracy"])
    min_delta = float(thresholds["min_mean_baseline_delta"])
    min_years_above = int(thresholds["min_years_roc_auc_above_50"])
    n_years = int(thresholds["n_development_years"])
    strong_auc = float(thresholds["strong_min_mean_roc_auc"])
    strong_balanced = float(thresholds["strong_min_mean_balanced_accuracy"])
    strong_years = int(thresholds["strong_min_years_beating_baseline"])

    criteria = {
        "mean_roc_auc": {
            "value": natural.get("mean_roc_auc"), "threshold": min_auc,
            "passed": _at_least(natural.get("mean_roc_auc"), min_auc)},
        "mean_balanced_accuracy": {
            "value": natural.get("mean_balanced_accuracy"), "threshold": min_balanced,
            "passed": _at_least(natural.get("mean_balanced_accuracy"), min_balanced)},
        "mean_accuracy_delta_vs_train_majority": {
            "value": natural.get("mean_baseline_delta"), "threshold": min_delta,
            "passed": _at_least(natural.get("mean_baseline_delta"), min_delta)},
        "years_roc_auc_above_50": {
            "value": natural.get("years_roc_auc_above_50"), "threshold": min_years_above,
            "passed": int(natural.get("years_roc_auc_above_50") or 0) >= min_years_above},
        "common_origin_directionally_positive": {
            "value": common.get("mean_baseline_delta"),
            "threshold": 0.0,
            "passed": _at_least(common.get("mean_baseline_delta"), 0.0)},
        "non_overlap_not_chance": {
            "value": non_overlap.get("mean_roc_auc"), "threshold": 0.50,
            "passed": _at_least(non_overlap.get("mean_roc_auc"), 0.50)},
    }
    passed = all(item["passed"] for item in criteria.values())
    strongly = (passed
                and _at_least(natural.get("mean_roc_auc"), strong_auc)
                and _at_least(natural.get("mean_balanced_accuracy"), strong_balanced)
                and int(natural.get("years_beating_majority_baseline") or 0) >= strong_years)
    return {
        "screen_passed": bool(passed),
        "classification": "STRONGLY_PROMISING" if strongly else (
            "SCREEN_PASS" if passed else "SCREEN_FAIL"),
        "criteria": criteria,
        "thresholds": dict(thresholds),
        "n_development_years": n_years,
        "note": ("diagnostic thresholds, not a target; 1D is a CONTROL and is never "
                 "selected"),
    }


def _at_least(value, floor) -> bool:
    if value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(number):
        return False
    return number >= float(floor)


def rank_horizons(aggregates: dict[int, dict], *, passing: list[int]) -> list[dict]:
    """Rank the SCREEN-PASSING horizons (at most two are ever selected).

    Ranking keys, in order: mean ROC-AUC, mean balanced accuracy, Brier, worst-year
    robustness, ticker breadth.  Closeness to 65% is never a ranking criterion.
    """
    ranked = []
    for horizon in passing:
        block = aggregates.get(horizon) or {}
        ranked.append({
            "horizon": int(horizon),
            "objective_id": HZ.objective_id(horizon),
            "horizon_phrase": HZ.horizon_phrase(horizon),
            "mean_roc_auc": block.get("mean_roc_auc"),
            "mean_balanced_accuracy": block.get("mean_balanced_accuracy"),
            "mean_brier": block.get("mean_brier"),
            "worst_year_roc_auc": block.get("worst_year_roc_auc"),
            "mean_ticker_fraction_beating_own_baseline":
                block.get("mean_ticker_fraction_beating_own_baseline"),
        })
    ranked.sort(key=lambda item: (
        -(_number(item["mean_roc_auc"])), -(_number(item["mean_balanced_accuracy"])),
        _number(item["mean_brier"]), -(_number(item["worst_year_roc_auc"])),
        -(_number(item["mean_ticker_fraction_beating_own_baseline"]))))
    for position, item in enumerate(ranked, start=1):
        item["rank"] = position
    return ranked


def _number(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("inf") if value is None else float("nan")
    return number if np.isfinite(number) else float("inf")


def transformer_delta(n2: dict, n1: dict) -> dict:
    """``N2 - N1`` for the four reported metrics."""
    out = {}
    for key in ("macro_ticker_accuracy", "balanced_accuracy", "roc_auc", "brier"):
        left, right = n2.get(key), n1.get(key)
        out[key] = (None if left is None or right is None
                    else float(left) - float(right))
    out["interpretation"] = ("a POSITIVE delta means the LSTM+Transformer improved the "
                             "metric, except for Brier where a NEGATIVE delta is the "
                             "improvement")
    return out


def evaluate_neural_gate(natural: dict, *, non_overlap: dict, thresholds: dict) -> dict:
    """Apply the NEURAL SIGNAL GATE of section 26."""
    min_auc = float(thresholds["min_mean_roc_auc"])
    min_balanced = float(thresholds["min_mean_balanced_accuracy"])
    min_delta = float(thresholds["min_mean_accuracy_delta"])
    min_years = int(thresholds["min_years_positive_baseline_delta"])
    floor_auc = float(thresholds["min_year_roc_auc"])
    criteria = {
        "mean_roc_auc": {"value": natural.get("mean_roc_auc"), "threshold": min_auc,
                         "passed": _at_least(natural.get("mean_roc_auc"), min_auc)},
        "mean_balanced_accuracy": {
            "value": natural.get("mean_balanced_accuracy"), "threshold": min_balanced,
            "passed": _at_least(natural.get("mean_balanced_accuracy"), min_balanced)},
        "mean_accuracy_vs_train_majority": {
            "value": natural.get("mean_baseline_delta"), "threshold": min_delta,
            "passed": _at_least(natural.get("mean_baseline_delta"), min_delta)},
        "years_with_positive_baseline_delta": {
            "value": natural.get("years_beating_majority_baseline"),
            "threshold": min_years,
            "passed": int(natural.get("years_beating_majority_baseline") or 0) >= min_years},
        "no_year_roc_auc_below_floor": {
            "value": natural.get("min_year_roc_auc"), "threshold": floor_auc,
            "passed": _at_least(natural.get("min_year_roc_auc"), floor_auc)},
        "non_overlap_remains_positive": {
            "value": non_overlap.get("mean_baseline_delta"), "threshold": 0.0,
            "passed": _at_least(non_overlap.get("mean_baseline_delta"), 0.0)},
    }
    passed = all(item["passed"] for item in criteria.values())
    return {
        "passed": bool(passed),
        "criteria": criteria,
        "thresholds": dict(thresholds),
        "preferred": {
            "mean_roc_auc": _at_least(natural.get("mean_roc_auc"),
                                      float(thresholds["preferred_min_mean_roc_auc"])),
            "mean_macro_accuracy": _at_least(
                natural.get("mean_macro_accuracy"),
                float(thresholds["preferred_min_mean_macro_accuracy"])),
        },
        "not_a_65_percent_target": True,
    }


def seed_stability(per_seed: dict[int, list[dict]], *, thresholds: dict | None = None
                   ) -> dict:
    """Aggregate seeds 11 / 42 / 73.  The best seed is never picked."""
    thresholds = thresholds or {}
    max_std = float(thresholds.get("max_macro_std", MAX_MACRO_STD_ACROSS_SEEDS))
    seeds = sorted(per_seed)
    aggregates = {seed: HM.aggregate_years(rows) for seed, rows in per_seed.items()}

    def _spread(key: str) -> dict:
        values = np.asarray([float(aggregates[s].get(key, np.nan)) for s in seeds],
                            dtype=float)
        finite = values[np.isfinite(values)]
        return {
            "values": {str(seed): (None if not np.isfinite(float(aggregates[seed].get(key,
                                                                                      np.nan)))
                                   else float(aggregates[seed].get(key)))
                       for seed in seeds},
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "std": float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
            "min": float(finite.min()) if finite.size else float("nan"),
            "max": float(finite.max()) if finite.size else float("nan"),
            "sign_changes": int(np.sum(np.sign(finite[:-1]) != np.sign(finite[1:])))
            if finite.size > 1 else 0,
        }

    macro = _spread("mean_macro_accuracy")
    unstable = (not np.isfinite(macro["std"]) or float(macro["std"]) > max_std
                or macro["sign_changes"] > 0)
    return {
        "seeds": seeds,
        "aggregates": {str(seed): aggregates[seed] for seed in seeds},
        "accuracy": _spread("mean_accuracy"),
        "balanced_accuracy": _spread("mean_balanced_accuracy"),
        "roc_auc": _spread("mean_roc_auc"),
        "brier": _spread("mean_brier"),
        "macro_accuracy": macro,
        "max_macro_std_allowed": max_std,
        "baseline_delta_sign_changes": int(
            sum(_spread("mean_baseline_delta")["sign_changes"] for _ in [0])),
        "stable": not unstable,
        "classification": "STABLE" if not unstable else "UNSTABLE",
        "rule": (f"macro accuracy std > {max_std} or a baseline-improvement sign change "
                 "across seeds classifies the winner UNSTABLE and blocks the lockbox"),
        "best_seed_selected": False,
    }


# ---------------------------------------------------------------------------
# the frozen manifest
# ---------------------------------------------------------------------------

def build_frozen_manifest(*, horizon: int, architecture: str, screen_rank: list[dict],
                          neural_metrics: dict, common_metrics: dict,
                          non_overlap_metrics: dict, stability: dict,
                          universe_sha256: str, feature_schema_sha256: str,
                          store_sha256: str, target_schema_sha256: str,
                          config_sha256: str, experiment_ids: list[str],
                          seed: int, gates: dict) -> dict:
    """The single record that must exist before 2019 may be opened."""
    manifest = {
        "track": HZ.TRACK_ID,
        "objective": HZ.objective_id(horizon),
        "horizon": int(horizon),
        "horizon_trading_observations": int(horizon),
        "horizon_phrase": HZ.horizon_phrase(horizon),
        "target_equation": HZ.TARGET_EQUATION,
        "return_equation": HZ.RETURN_EQUATION,
        "trading_day_rule": HZ.TRADING_DAY_RULE,
        "selected_architecture": architecture,
        "architecture_label": HZ.NEURAL_LABELS.get(architecture, architecture),
        "supervised_universe_sha256": universe_sha256,
        "feature_schema_sha256": feature_schema_sha256,
        "store_sha256": store_sha256,
        "target_schema_sha256": target_schema_sha256,
        "config_sha256": config_sha256,
        "development_experiment_ids": sorted(experiment_ids),
        "development_metrics": neural_metrics,
        "common_origin_metrics": common_metrics,
        "non_overlapping_metrics": non_overlap_metrics,
        "seed_stability": stability,
        "screen_ranking": screen_rank,
        "gates": gates,
        "seed": int(seed),
        "final_allowed_date": HZ.FINAL_ALLOWED_DATE,
        "lockbox_fold": HZ.LOCKBOX_FOLD,
        "lockbox_year": HZ.LOCKBOX_YEAR,
        "lockbox_env_var": HZ.LOCKBOX_ENV,
        "frozen_at": datetime.now(UTC).isoformat(),
        "immutable_after_freeze": [
            "horizon", "architecture", "thresholds", "features", "seed",
        ],
    }
    manifest["manifest_sha256"] = HZ.hash_payload(manifest)
    return manifest


def verify_frozen_manifest(manifest: dict, *, horizon: int, architecture: str,
                           target_schema_sha256: str, store_sha256: str,
                           universe_sha256: str, config_sha256: str) -> dict:
    """Refuse a lockbox whose horizon, architecture or hashes have changed.

    Section 30: once frozen, the horizon, architecture, thresholds, features and
    seed may not change on the basis of 2019.  Each mismatch is reported by name so
    a reader can see exactly what moved.
    """
    recorded_hash = manifest.get("manifest_sha256")
    recomputed = dict(manifest)
    recomputed.pop("manifest_sha256", None)
    integrity = HZ.hash_payload(recomputed) == recorded_hash

    checks = {
        "manifest_integrity": {
            "recorded": recorded_hash, "recomputed": HZ.hash_payload(recomputed),
            "passed": bool(integrity)},
        "horizon_unchanged": {
            "frozen": manifest.get("horizon"), "requested": int(horizon),
            "passed": int(manifest.get("horizon", -1)) == int(horizon)},
        "objective_unchanged": {
            "frozen": manifest.get("objective"), "requested": HZ.objective_id(horizon),
            "passed": manifest.get("objective") == HZ.objective_id(horizon)},
        "architecture_unchanged": {
            "frozen": manifest.get("selected_architecture"), "requested": architecture,
            "passed": manifest.get("selected_architecture") == architecture},
        "target_schema_unchanged": {
            "frozen": manifest.get("target_schema_sha256"),
            "requested": target_schema_sha256,
            "passed": manifest.get("target_schema_sha256") == target_schema_sha256},
        "store_unchanged": {"frozen": manifest.get("store_sha256"),
                            "requested": store_sha256,
                            "passed": manifest.get("store_sha256") == store_sha256},
        "universe_unchanged": {"frozen": manifest.get("supervised_universe_sha256"),
                               "requested": universe_sha256,
                               "passed": (manifest.get("supervised_universe_sha256")
                                          == universe_sha256)},
        "config_unchanged": {"frozen": manifest.get("config_sha256"),
                             "requested": config_sha256,
                             "passed": manifest.get("config_sha256") == config_sha256},
    }
    failed = [name for name, check in checks.items() if not check["passed"]]
    return {
        "verified": not failed,
        "failed_checks": failed,
        "checks": checks,
        "message": ("frozen horizon/model manifest verified" if not failed else
                    "refusing the 2019 lockbox: " + ", ".join(failed)),
    }


def load_frozen_manifest(path: Path) -> dict:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(
            f"No frozen horizon/model manifest at {target}. The 2019 lockbox may only "
            f"run after horizon and model are frozen.")
    return json.loads(target.read_text())