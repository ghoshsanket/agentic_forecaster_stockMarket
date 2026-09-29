"""Thin re-export so scoring.py can fit a calibrator without importing the
model layer (and without a circular import).

The canonical implementation lives in
agentic_forecaster.calibration.registry; this module only exposes
fit_calibrator under the name scoring.py uses.
"""

from agentic_forecaster.calibration.registry import (  # noqa: F401
    SUPPORTED_METHODS,
    build_calibrator,
    calibrator_from_payload,
    fit_calibrator,
)

fit_calibrator_registry = fit_calibrator
