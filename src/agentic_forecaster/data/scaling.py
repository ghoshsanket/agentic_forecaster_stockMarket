"""Scaler selection and causal volume preprocessing.

The Phase-1 reconstruction always used ``StandardScaler`` regardless of
``data.scaler``, which would have made the recovery search's scaler axis a
no-op.  This module makes the configured choice real while keeping the
SAME contract for every option: the scaler is fitted on TRAIN values only and
then applied to validation and test.

``volume_mode`` is applied BEFORE scaling and is causal (a point transform of
the same bar, so it introduces no look-ahead):

* ``raw``   - volume passes through unchanged
* ``log1p`` - ``log1p(volume)``, which compresses the heavy right tail that
  makes raw volume dominate a fitted scaler

OHLC columns are never altered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

#: Config value -> (class, human description)
SCALER_REGISTRY: dict[str, str] = {
    "standard": "sklearn.preprocessing.StandardScaler",
    "minmax": "sklearn.preprocessing.MinMaxScaler",
    "robust": "sklearn.preprocessing.RobustScaler",
}

VOLUME_MODES = ("raw", "log1p")


def build_scaler(name: str = "standard"):
    """Instantiate the configured scaler. Fitted later, on TRAIN only."""
    key = (name or "standard").lower()
    if key not in SCALER_REGISTRY:
        raise KeyError(
            f"Unknown scaler {name!r}. Expected one of {sorted(SCALER_REGISTRY)}."
        )
    return {"standard": StandardScaler,
            "minmax": MinMaxScaler,
            "robust": RobustScaler}[key]()


def scaler_description(name: str = "standard") -> str:
    return SCALER_REGISTRY.get((name or "standard").lower(), "unknown")


def apply_volume_mode(df: pd.DataFrame, mode: str = "log1p") -> pd.DataFrame:
    """Apply the volume preprocessing causally, leaving OHLC untouched.

    Returns a COPY; the input frame is not modified.
    """
    key = (mode or "log1p").lower()
    if key not in VOLUME_MODES:
        raise KeyError(
            f"Unknown volume_mode {mode!r}. Expected one of {VOLUME_MODES}."
        )
    out = df.copy()
    if "volume" not in out.columns:
        return out
    if key == "log1p":
        out["volume"] = np.log1p(pd.to_numeric(out["volume"], errors="coerce").clip(lower=0.0))
    return out


def fit_scaler_on_train(scaler, train_values: np.ndarray) -> object:
    """Fit a scaler on TRAIN values only.

    ``train_values`` must be the TRAIN split only.  The fit is reshaped to 2-D
    because the sequences are (n_samples, seq_len, n_features).
    """
    flat = np.asarray(train_values, dtype=np.float64)
    scaler.fit(flat.reshape(-1, flat.shape[-1]))
    return scaler


def apply_scaler(scaler, values: np.ndarray) -> np.ndarray:
    if not len(values):
        return values
    flat = np.asarray(values, dtype=np.float64)
    return scaler.transform(flat.reshape(-1, flat.shape[-1])).reshape(flat.shape)
