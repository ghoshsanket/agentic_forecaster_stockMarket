"""Canonical, serializable calibration registry.

The publication reports calibrated probabilities without naming the method, so
the recovery search must be able to fit any of four.  ``FittedModel`` used to
carry a single temperature float, which cannot represent Platt or isotonic and
would silently reduce every method to temperature scaling.

This registry gives each method a real ``fit`` / ``predict`` / ``to_dict`` /
``from_dict`` pair.  Persistence is therefore method-agnostic, and
``calibration.json`` records the method that was actually used.

Backward compatibility: a bundle whose ``calibration.json`` has no ``method``
key (an old temperature-only bundle) loads as ``temperature``.
"""

from __future__ import annotations

import numpy as np


class IdentityCalibrator:
    method = "none"

    def fit(self, p_up, y):
        return self

    def predict(self, p_up):
        return np.asarray(p_up, dtype=float).reshape(-1)

    calibrate_proba = predict

    def to_dict(self) -> dict:
        return {"method": "none"}

    @classmethod
    def from_dict(cls, payload: dict) -> IdentityCalibrator:
        return cls()


class TemperatureCalibratorAdapter:
    """Wraps the project's TemperatureCalibrator, which is defined on logits."""

    method = "temperature"

    def __init__(self) -> None:
        from agentic_forecaster.calibration.temperature import TemperatureCalibrator
        self._inner = TemperatureCalibrator()

    @staticmethod
    def _logits(p_up: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p_up, dtype=float).reshape(-1), 1e-7, 1 - 1e-7)
        return np.log(p / (1.0 - p))

    def fit(self, p_up, y):
        self._inner.fit(self._logits(p_up), np.asarray(y).reshape(-1))
        return self

    def predict(self, p_up):
        return np.asarray(self._inner.calibrate_proba(
            np.asarray(p_up, dtype=float).reshape(-1)), dtype=float).reshape(-1)

    calibrate_proba = predict

    def to_dict(self) -> dict:
        return {"method": "temperature", "temperature": float(self._inner.temperature)}

    @classmethod
    def from_dict(cls, payload: dict) -> TemperatureCalibratorAdapter:
        obj = cls()
        t = payload.get("temperature")
        if t is not None:
            obj._inner.temperature = float(t)
        return obj


class PlattCalibrator:
    method = "platt"

    def __init__(self) -> None:
        from sklearn.linear_model import LogisticRegression
        self._inner = LogisticRegression(C=1e6, solver="lbfgs")

    def fit(self, p_up, y):
        p = np.asarray(p_up, dtype=float).reshape(-1, 1)
        self._inner.fit(p, np.asarray(y).reshape(-1).astype(int))
        return self

    def predict(self, p_up):
        out = np.asarray(self._inner.predict_proba(
            np.asarray(p_up, dtype=float).reshape(-1, 1)), dtype=float)
        return (out[:, 1] if out.ndim == 2 and out.shape[1] == 2 else out).reshape(-1)

    calibrate_proba = predict

    def to_dict(self) -> dict:
        return {
            "method": "platt",
            "coef": [float(c) for c in np.asarray(self._inner.coef_).reshape(-1)],
            "intercept": [float(i) for i in np.asarray(self._inner.intercept_).reshape(-1)],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> PlattCalibrator:
        obj = cls()
        obj._inner = _FrozenLogistic(float(np.asarray(payload["coef"]).reshape(-1)[0]),
                                     float(np.asarray(payload["intercept"]).reshape(-1)[0]))
        return obj


class _FrozenLogistic:
    """A fitted logistic map reconstructed from coefficients, without re-fitting."""

    def __init__(self, coef: float, intercept: float) -> None:
        self.coef_ = np.array([[coef]])
        self.intercept_ = np.array([intercept])

    def predict_proba(self, X):
        z = self.coef_[0, 0] * np.asarray(X, dtype=float).reshape(-1) + self.intercept_[0]
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p, p])


class IsotonicCalibrator:
    method = "isotonic"

    def __init__(self) -> None:
        from sklearn.isotonic import IsotonicRegression
        self._inner = IsotonicRegression(out_of_bounds="clip")

    def fit(self, p_up, y):
        self._inner.fit(np.asarray(p_up, dtype=float).reshape(-1),
                        np.asarray(y).reshape(-1).astype(int))
        return self

    def predict(self, p_up):
        return np.asarray(self._inner.predict(
            np.asarray(p_up, dtype=float).reshape(-1)), dtype=float).reshape(-1)

    calibrate_proba = predict

    def to_dict(self) -> dict:
        return {
            "method": "isotonic",
            "x_thresholds": [float(v) for v in np.asarray(self._inner.X_thresholds_).reshape(-1)],
            "y_thresholds": [float(v) for v in np.asarray(self._inner.y_thresholds_).reshape(-1)],
            # sklearn's _transform also clips to these bounds, so they must be
            # persisted too or a restored calibrator raises AttributeError.
            "x_min": float(np.min(self._inner.X_thresholds_)),
            "x_max": float(np.max(self._inner.X_thresholds_)),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> IsotonicCalibrator:
        obj = cls()
        xt = np.asarray(payload["x_thresholds"], dtype=float)
        obj._inner.X_thresholds_ = xt
        obj._inner.y_thresholds_ = np.asarray(payload["y_thresholds"], dtype=float)
        obj._inner.X_min_ = float(payload.get("x_min", xt.min()))
        obj._inner.X_max_ = float(payload.get("x_max", xt.max()))
        obj._inner.increasing_ = True
        obj._inner.out_of_bounds_ = "clip"
        # sklearn's _transform delegates to the interpolator f_; rebuild it
        # from the persisted thresholds instead of re-fitting.
        from scipy.interpolate import interp1d
        obj._inner.f_ = interp1d(obj._inner.X_thresholds_,
                                 obj._inner.y_thresholds_, kind="linear",
                                 bounds_error=False)
        return obj


CALIBRATORS = {
    "none": IdentityCalibrator,
    "temperature": TemperatureCalibratorAdapter,
    "platt": PlattCalibrator,
    "isotonic": IsotonicCalibrator,
}

SUPPORTED_METHODS = tuple(CALIBRATORS)


def build_calibrator(method: str):
    key = (method or "temperature").lower()
    if key not in CALIBRATORS:
        raise KeyError(
            f"Unknown calibration method {method!r}. "
            f"Expected one of {sorted(CALIBRATORS)}."
        )
    return CALIBRATORS[key]()


def calibrator_from_payload(payload: dict | None):
    """Rebuild a calibrator from its ``calibration.json`` payload.

    Backward compatible: a payload with no ``method`` key is an old
    temperature-only bundle.
    """
    if not payload:
        return IdentityCalibrator()
    method = str(payload.get("method", "temperature")).lower()
    if method == "temperature" and "temperature" not in payload:
        # legacy bundle that stored only a temperature float
        return TemperatureCalibratorAdapter.from_dict(
            {"temperature": float(payload.get("temperature", 1.0))})
    return build_calibrator(method).from_dict(payload)


def fit_calibrator(method: str, p_up, y, *, dates=None, search=None):
    """Fit one calibration method on VALIDATION data, firewall-checked."""
    if dates is not None:
        from agentic_forecaster.recovery.firewall import assert_pre_test_dates
        assert_pre_test_dates(dates, where=f"calibration[{method}]",
                              search=True if search is None else search)
    return build_calibrator(method).fit(p_up, y)
