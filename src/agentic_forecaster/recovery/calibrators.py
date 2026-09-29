"""Calibration method registry for the recovery search.

The publication reports calibrated probabilities but never names the method, so
the search must be able to fit each candidate.

ALL calibration here is fitted on the VALIDATION split of a pre-2022 search
fold.  It must never see the final test labels.  :func:`fit_calibrator` refuses
to fit when the firewall is active and the supplied dates touch 2022/2023.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from agentic_forecaster.recovery.firewall import assert_pre_test_dates


class Calibrator(Protocol):
    """Maps raw UP-probabilities to calibrated UP-probabilities."""

    def predict(self, proba: np.ndarray) -> np.ndarray: ...


class _Identity:
    """The ``none`` method: probabilities pass through unchanged."""

    method = "none"

    def fit(self, proba, y) -> _Identity:
        return self

    def predict(self, proba: np.ndarray) -> np.ndarray:
        return np.asarray(proba, dtype=float)


class _SklearnCalibrator:
    """Wrapper for temperature / Platt / isotonic behind one interface."""

    def __init__(self, method: str) -> None:
        self.method = method
        self._inner = None

    def fit(self, proba, y) -> _SklearnCalibrator:
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        p = np.asarray(proba, dtype=float).reshape(-1, 1)
        yv = np.asarray(y, dtype=int).reshape(-1)
        if self.method == "platt":
            self._inner = LogisticRegression(C=1e6, solver="lbfgs")
            self._inner.fit(p, yv)
        elif self.method == "isotonic":
            self._inner = IsotonicRegression(out_of_bounds="clip")
            self._inner.fit(p.ravel(), yv)
        else:
            # The project's own temperature calibrator is defined on LOGITS and
            # exposes calibrate_proba(), so adapt at the boundary.
            from agentic_forecaster.calibration.temperature import (
                TemperatureCalibrator,
            )
            eps = 1e-7
            clipped = np.clip(p.ravel(), eps, 1.0 - eps)
            logits = np.log(clipped / (1.0 - clipped))
            self._inner = TemperatureCalibrator().fit(logits, yv)
        return self

    def predict(self, proba: np.ndarray) -> np.ndarray:
        p = np.asarray(proba, dtype=float).reshape(-1)
        cal_proba = getattr(self._inner, "calibrate_proba", None)
        if callable(cal_proba):
            # temperature: operates directly on the UP probability
            return np.asarray(cal_proba(p), dtype=float).reshape(-1)
        pred = getattr(self._inner, "predict_proba", None)
        if callable(pred):
            out = np.asarray(pred(p.reshape(-1, 1)), dtype=float)
            # Brier must be computed on the UP probability, never on a flipped
            # directional confidence. Keep column 1 explicitly.
            if out.ndim == 2 and out.shape[1] == 2:
                out = out[:, 1]
            return out.reshape(-1)
        return np.asarray(self._inner.predict(p.reshape(-1, 1)), dtype=float).reshape(-1)


def fit_calibrator(method: str, proba, y, *, dates=None,
                   search: bool | None = None) -> Calibrator:
    """Fit one calibration method on validation data.

    ``dates`` should be the validation dates.  When search mode is active the
    firewall rejects any date in 2022/2023, so calibration provably cannot see
    the final test labels.
    """
    method = (method or "none").lower()
    if method not in ("none", "temperature", "platt", "isotonic"):
        raise ValueError(
            f"Unknown calibration method {method!r}. "
            "Expected one of none/temperature/platt/isotonic."
        )
    if dates is not None:
        assert_pre_test_dates(dates, where=f"calibration[{method}]", search=search)
    if method == "none":
        return _Identity().fit(proba, y)
    return _SklearnCalibrator(method).fit(proba, y)


def brier_score(y_true, proba_up) -> float:
    """Brier score for the UP probability: mean((p_up - y)^2), y in {0,1}.

    Defined explicitly here (rather than only delegating to sklearn) because the
    recovery specification pins this definition, and because a subtle error -
    scoring a flipped directional confidence instead of p_up - is exactly the
    kind of thing that quietly inflates or deflates a recovered metric.
    """
    import pandas as pd
    y = np.asarray(pd.Series(y_true).astype(float).to_numpy(), dtype=float)
    p = np.asarray(pd.Series(proba_up).astype(float).to_numpy(), dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y={y.shape} p_up={p.shape}")
    return float(np.mean((p - y) ** 2))
