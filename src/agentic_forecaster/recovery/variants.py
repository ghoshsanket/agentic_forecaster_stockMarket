"""Catalogue of recoverable implementation choices.

The publication pins some values (Adam lr 1e-3, beta1 0.9, beta2 0.999, batch
64, patience 10, 30-day lookback, attention over a 2-layer/64-unit recurrent
stack) and is silent or ambiguous on others.  This module enumerates the
silent/ambiguous ones as explicit, named variants so the search is staged and
auditable rather than an undirected Cartesian grid.

Nothing here is a default recommendation.  The Phase-1 reconstruction values are
recorded as the REFERENCE point, and the search decides.
"""

from __future__ import annotations

# ---------------------------------------------------------------- features

#: The Phase-1 compact set.
F1_INDICATORS: tuple[str, ...] = (
    "log_return",
    "realized_volatility_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "atr_14",
)

#: F2 = F1 plus the features named in the publication's example explanations
#: (5-day MA, 20-day MA, Bollinger %B, OBV) plus their derived spread.
F2_EXTRA_INDICATORS: tuple[str, ...] = (
    "sma_5",
    "sma_20",
    "sma_5_minus_sma_20",
    "bb_upper",
    "bb_lower",
    "bb_percent_b",
    "obv",
)

#: F3 = F2 plus causal short-return / momentum features that ALREADY exist in
#: the engine.  Nothing here is newly invented momentum.
F3_EXTRA_INDICATORS: tuple[str, ...] = (
    "returns_1",
    "returns_5",
    "returns_10",
    "sma_50",
    "ema_12",
    "ema_26",
)

FEATURE_FAMILIES: dict[str, dict] = {
    "F0": {
        "description": "OHLCV only, no indicators.",
        "use_ohlcv": True,
        "indicators": (),
    },
    "F1": {
        "description": "Phase-1 compact set (reference).",
        "use_ohlcv": True,
        "indicators": F1_INDICATORS,
    },
    "F2": {
        "description": "F1 + SMA5/SMA20/spread/Bollinger bands/%B/OBV "
                       "(features named in the paper's example explanations).",
        "use_ohlcv": True,
        "indicators": F1_INDICATORS + F2_EXTRA_INDICATORS,
    },
    "F3": {
        "description": "F2 + already-existing causal short-return/momentum features.",
        "use_ohlcv": True,
        "indicators": F1_INDICATORS + F2_EXTRA_INDICATORS + F3_EXTRA_INDICATORS,
    },
}

DEFAULT_FEATURE_FAMILY = "F1"

#: Each RSI method maps to the indicator that must be swapped into F1.
RSI_METHODS: dict[str, dict] = {
    "R0_rolling": {
        "description": "Rolling arithmetic mean of gains/losses (matches the "
                       "displayed equations literally).",
        "indicator": "rsi_14_rolling",
    },
    "R1_wilder": {
        "description": "Wilder / RMA smoothing (Phase-1 reference).",
        "indicator": "rsi_14",
    },
    "R2_ema": {
        "description": "EMA-smoothed gains/losses.",
        "indicator": "rsi_14_ema",
    },
}

DEFAULT_RSI_METHOD = "R1_wilder"

# --------------------------------------------------------------- training

#: TRAINING-LENGTH RECOVERY - highest priority.  The publication specifies
#: patience 10 but NO epoch cap; the reproduction's 3-epoch cap is a budget,
#: not a property of the paper.
TRAINING_LENGTHS: dict[str, dict] = {
    "T3": {"max_epochs": 3, "patience": 10, "restore_best_checkpoint": False,
           "description": "Phase-1 reproduction budget; included as a control."},
    "T30": {"max_epochs": 30, "patience": 10, "restore_best_checkpoint": True},
    "T50": {"max_epochs": 50, "patience": 10, "restore_best_checkpoint": True},
    "T100": {"max_epochs": 100, "patience": 10, "restore_best_checkpoint": True},
}

DEFAULT_TRAINING_LENGTH = "T100"

#: 30 is AUTHOR-CONFIRMED; the rest are validation-only sensitivity.
LOOKBACKS: dict[str, int] = {"L10": 10, "L20": 20, "L30": 30, "L40": 40, "L60": 60}
DEFAULT_LOOKBACK = 30  # AUTHOR-CONFIRMED reference

# ---------------------------------------------------------------- scalers

SCALERS: dict[str, str] = {
    "standard": "StandardScaler",
    "minmax": "MinMaxScaler",
    "robust": "RobustScaler",
}
DEFAULT_SCALER = "standard"

#: Controlled volume preprocessing variants. Volume preprocessing is an
#: INVESTIGATED choice, not a silent default: the original/reconstruction
#: pipeline used ordinary OHLCV, so ``raw`` is the Stage-A reference. ``log1p``
#: is retained as an explicit Stage-B sensitivity option.
VOLUME_MODES: dict[str, str] = {
    "raw": "raw volume as supplied (Phase-1 reference)",
    "log1p": "log1p(volume) before scaling (Stage-B sensitivity)",
}
DEFAULT_VOLUME_MODE = "raw"

# ----------------------------------------------------------- architecture

#: Reference is 2 layers x 64 units, from the publication's h^(2) and 64-dim
#: hidden state.  No bidirectional recurrent layers are offered.
ARCHITECTURES: dict[str, dict] = {
    "A0": {"num_layers": 2, "hidden_size": 64, "description": "reference"},
    "A1": {"num_layers": 1, "hidden_size": 64},
    "A2": {"num_layers": 2, "hidden_size": 32},
    "A3": {"num_layers": 2, "hidden_size": 128},
}
DEFAULT_ARCHITECTURE = "A0"

DROPOUTS: tuple[float, ...] = (0.0, 0.2, 0.5)
DEFAULT_DROPOUT = 0.2

#: The paper contains an L2 term but never states lambda.
WEIGHT_DECAYS: tuple[float, ...] = (0.0, 1e-6, 1e-5, 1e-4, 1e-3)
DEFAULT_WEIGHT_DECAY = 1e-4

CLASS_WEIGHTINGS: tuple[str, ...] = ("none", "pos_weight")
DEFAULT_CLASS_WEIGHTING = "none"

# ------------------------------------------------------------ calibration

#: The paper reports calibrated probabilities without naming the method.
CALIBRATION_METHODS: tuple[str, ...] = ("none", "temperature", "platt", "isotonic")
DEFAULT_CALIBRATION = "temperature"

# ------------------------------------------------------------------ seeds

#: Seed stability is assessed only for the most promising configurations.
SEEDS: tuple[int, ...] = (11, 23, 42, 73, 101)
DEFAULT_SEED = 42

# ------------------------------------------------------------- validation

#: Ranking uses pre-2022 validation only.  Never the distance to the published
#: test numbers.
SELECTION_METRICS: tuple[str, ...] = (
    "validation_accuracy",
    "validation_f1",
    "validation_brier",
    "validation_ece",
)


def build_feature_indicators(feature_family: str = DEFAULT_FEATURE_FAMILY,
                             rsi_method: str = DEFAULT_RSI_METHOD
                             ) -> tuple[str, ...]:
    """Resolve a feature family + RSI method into the engine's indicator names.

    The selected RSI indicator REPLACES the family's default ``rsi_14`` rather
    than being added alongside it, so a variant never carries two RSI columns.
    """
    if feature_family not in FEATURE_FAMILIES:
        raise KeyError(
            f"Unknown feature family {feature_family!r}. "
            f"Available: {sorted(FEATURE_FAMILIES)}"
        )
    if rsi_method not in RSI_METHODS:
        raise KeyError(
            f"Unknown RSI method {rsi_method!r}. Available: {sorted(RSI_METHODS)}"
        )
    spec = FEATURE_FAMILIES[feature_family]
    target = RSI_METHODS[rsi_method]["indicator"]
    out: list[str] = []
    for name in spec["indicators"]:
        if name == "rsi_14":
            out.append(target)
        elif name == "rsi_14_rolling" or name == "rsi_14_ema":
            continue  # only the chosen RSI variant survives
        else:
            out.append(name)
    return tuple(out)


def describe_variants() -> dict:
    """Machine-readable catalogue, for manifests and the audit script."""
    return {
        "feature_families": {k: {"description": v["description"],
                                 "n_indicators": len(build_feature_indicators(k))}
                             for k, v in FEATURE_FAMILIES.items()},
        "rsi_methods": RSI_METHODS,
        "training_lengths": TRAINING_LENGTHS,
        "lookbacks": LOOKBACKS,
        "scalers": SCALERS,
        "volume_modes": VOLUME_MODES,
        "architectures": ARCHITECTURES,
        "dropouts": list(DROPOUTS),
        "weight_decays": list(WEIGHT_DECAYS),
        "class_weightings": list(CLASS_WEIGHTINGS),
        "calibration_methods": list(CALIBRATION_METHODS),
        "seeds": list(SEEDS),
        "selection_metrics": list(SELECTION_METRICS),
    }


def stage_plan() -> list[dict]:
    """The staged search plan - deliberately not a Cartesian product."""
    return [
        {"stage": "STAGE_0",
         "name": "sanity",
         "actions": ["tiny overfit (F0 harness)", "shuffled-label control"],
         "gate": "STOP if the model cannot overfit a tiny TRAIN subset, or if "
                 "real labels do not beat shuffled labels on pre-2022 validation."},
        {"stage": "STAGE_A",
         "name": "training length x feature set",
         "tickers": "8-10 representative stocks",
         "variant": "unadjusted (primary original-recollection candidate)",
         "folds": ["SEARCH_FOLD_A", "SEARCH_FOLD_B", "SEARCH_FOLD_C"],
         "axes": {"training_length": list(TRAINING_LENGTHS),
                  "feature_family": ["F1", "F2"]},
         "gate": "Choose the training-length family. If best epochs cluster in "
                 "12-25 the 3-epoch reproduction was demonstrably undertrained."},
        {"stage": "STAGE_B",
         "name": "rsi / scaler / lookback",
         "axes": {"rsi_method": list(RSI_METHODS),
                  "scaler": list(SCALERS),
                  "lookback": [10, 20, 30, 40, 60]},
         "gate": "Only after training length and feature set are diagnosed."},
        {"stage": "STAGE_C",
         "name": "architecture / dropout / weight decay / class weighting",
         "axes": {"architecture": list(ARCHITECTURES),
                  "dropout": list(DROPOUTS),
                  "weight_decay": list(WEIGHT_DECAYS),
                  "class_weighting": list(CLASS_WEIGHTINGS)},
         "gate": "Only after Stage B."},
        {"stage": "STAGE_D",
         "name": "calibration / seed stability",
         "axes": {"calibration": list(CALIBRATION_METHODS), "seeds": list(SEEDS)},
         "gate": "Calibration fitted on validation only, never on test."},
        {"stage": "STAGE_E",
         "name": "confirm on all 50",
         "tickers": "all 50 paper-snapshot candidate stocks",
         "folds": ["SEARCH_FOLD_A", "SEARCH_FOLD_B", "SEARCH_FOLD_C"],
         "gate": "Pre-2022 validation only. Then choose ONE configuration."},
    ]
