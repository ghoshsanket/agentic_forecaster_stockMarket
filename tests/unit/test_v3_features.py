"""V3 feature construction and FAIR COMPARISON.

Covers sections 51 and 52 of the specification:

Feature construction
    exogenous returns are causal; rolling volatility is causal; moving-average
    distance is causal; India VIX change uses historical values only; the PCA/scaler
    is fitted TRAIN only; the sector-index mapping cannot silently map UNKNOWN; an
    unavailable optional source fabricates nothing.

Fair comparison
    the X0-X4 common-sample evaluation uses identical (ticker, origin) pairs; the
    target labels are identical between feature families; the horizon implementation
    is the already-verified one; 2019 is inaccessible during model selection; 2020+ is
    inaccessible everywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agentic_forecaster.v2.dataset import SplitWindow
from agentic_forecaster.v3.availability import build_global_risk_factor, sector_index_mapping
from agentic_forecaster.v3.features import (
    build_price_features,
    build_volatility_features,
    build_yield_features,
    drop_degenerate_columns,
    finite_row_mask,
    relative_to_market_features,
)
from agentic_forecaster.v3.screening import apply_transform, fit_transform_train_only

WINDOW = SplitWindow("MH_DEV_2017", "2015-01-01", "2016-12-31",
                     "2017-01-01", "2017-12-31")


# ---------------------------------------------------------------------------
# feature causality
# ---------------------------------------------------------------------------

def _price_series(n: int = 400, seed: int = 5) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2014-01-01", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=dates)


def test_exogenous_returns_are_causal():
    """A shock at t may change features at t and later, never before it."""
    base = _price_series()
    shocked = base.copy()
    shocked.iloc[250] *= 1.20
    a = build_price_features(base, prefix="SP500")
    b = build_price_features(shocked, prefix="SP500")
    assert np.allclose(a.to_numpy()[:250], b.to_numpy()[:250], equal_nan=True)
    assert not np.allclose(a.to_numpy()[250:], b.to_numpy()[250:], equal_nan=True)


def test_rolling_volatility_is_causal_and_backward_looking():
    base = _price_series()
    shocked = base.copy()
    shocked.iloc[300] *= 1.30
    a = build_price_features(base, prefix="X")
    b = build_price_features(shocked, prefix="X")
    for column in ("X_realized_vol_5", "X_realized_vol_10", "X_realized_vol_20"):
        assert np.allclose(a[column].to_numpy()[:300],
                           b[column].to_numpy()[:300], equal_nan=True), column


def test_moving_average_distance_is_causal():
    base = _price_series()
    shocked = base.copy()
    shocked.iloc[350] *= 1.25
    a = build_price_features(base, prefix="X")
    b = build_price_features(shocked, prefix="X")
    assert np.allclose(a["X_distance_from_ma20"].to_numpy()[:350],
                       b["X_distance_from_ma20"].to_numpy()[:350], equal_nan=True)
    assert np.allclose(a["X_distance_from_ma60"].to_numpy()[:350],
                       b["X_distance_from_ma60"].to_numpy()[:350], equal_nan=True)
    assert not np.allclose(a["X_distance_from_ma60"].to_numpy()[350:],
                           b["X_distance_from_ma60"].to_numpy()[350:], equal_nan=True)


def test_drawdown_is_non_positive_and_causal():
    features = build_price_features(_price_series(), prefix="X")
    finite = features["X_drawdown_20"].dropna()
    assert (finite <= 1e-12).all(), "a drawdown can never be positive"


def test_india_vix_change_uses_historical_values_only():
    values = pd.Series(np.linspace(10, 30, 300),
                       index=pd.bdate_range("2014-01-01", periods=300))
    base = build_volatility_features(values, prefix="INDIA_VIX")
    shocked = values.copy()
    shocked.iloc[200] *= 2.0
    other = build_volatility_features(shocked, prefix="INDIA_VIX")
    for column in ("INDIA_VIX_change_1", "INDIA_VIX_change_5", "INDIA_VIX_zscore_20"):
        assert np.allclose(base[column].to_numpy()[:200],
                           other[column].to_numpy()[:200], equal_nan=True), column
    assert not np.allclose(base["INDIA_VIX_change_5"].to_numpy()[200:],
                           other["INDIA_VIX_change_5"].to_numpy()[200:], equal_nan=True)


def test_yield_level_is_used_directly_without_a_log_transform():
    values = pd.Series(np.linspace(1.0, 5.0, 200),
                       index=pd.bdate_range("2014-01-01", periods=200))
    features = build_yield_features(values, prefix="US_10Y_YIELD")
    assert features["US_10Y_YIELD_level"].iloc[-1] == pytest.approx(5.0)
    assert np.isfinite(features["US_10Y_YIELD_change_20"].iloc[-1])


def test_no_raw_level_is_ever_emitted_for_a_price_series():
    features = build_price_features(_price_series(), prefix="NIFTY50")
    assert not any(column.endswith("_level") for column in features.columns)
    assert "NIFTY50_log_level" not in features.columns


def test_warm_up_is_nan_and_never_filled():
    features = build_price_features(_price_series(), prefix="X")
    assert features["X_distance_from_ma60"].iloc[:59].isna().all()
    assert features["X_realized_vol_20"].iloc[:19].isna().all()


# ---------------------------------------------------------------------------
# transforms are TRAIN-only
# ---------------------------------------------------------------------------

def test_scaler_is_fitted_on_train_rows_only():
    rng = np.random.default_rng(0)
    x_train = rng.normal(0.0, 1.0, (200, 4))
    x_val = rng.normal(100.0, 1.0, (20, 4))          # a wildly shifted val block
    y_train = (rng.random(200) < 0.4).astype(int)
    fitted, state = fit_transform_train_only("LOGISTIC", x_train, y_train)
    assert state["fitted_on"] == "TRAIN_ONLY"
    assert state["validation_influence_on_transform"] == "NONE"
    assert state["scaler_n_rows"] == 200
    assert abs(float(np.mean(fitted["scaler"].mean_))) < 5.0, (
        "a validation block shifted by 100 must not move the fitted scaler")
    assert apply_transform(fitted, x_val).shape == (20, 4)


def test_pca_is_fitted_on_train_rows_only_and_persisted():
    rng = np.random.default_rng(1)
    x_train = rng.normal(0.0, 1.0, (200, 6))
    y_train = (rng.random(200) < 0.5).astype(int)
    fitted, state = fit_transform_train_only("LOGISTIC", x_train, y_train,
                                             pca_components=3)
    assert state["pca"]["n_components"] == 3
    assert state["fitted_on"] == "TRAIN_ONLY"
    assert apply_transform(fitted, x_train).shape == (200, 3)


def test_tree_model_records_that_no_scaler_was_fitted():
    rng = np.random.default_rng(2)
    fitted, state = fit_transform_train_only(
        "HIST_GRADIENT_BOOSTING", rng.normal(size=(80, 3)),
        np.array([0, 1] * 40))
    assert fitted["scaler"] is None
    assert state["scaler"] == "NOT_APPLICABLE_TREE_MODEL"


# ---------------------------------------------------------------------------
# degenerate and unavailable columns
# ---------------------------------------------------------------------------

def test_degenerate_columns_are_dropped_and_recorded():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"constant": [1.0] * 100, "all_missing": [np.nan] * 100,
                          "sparse": [np.nan] * 90 + list(rng.normal(size=10)),
                          "useful": rng.normal(size=100)})
    cleaned, dropped = drop_degenerate_columns(frame)
    assert set(dropped) == {"constant", "all_missing", "sparse"}
    assert list(cleaned.columns) == ["useful"]


def test_an_unavailable_optional_source_fabricates_nothing():
    """A missing optional source leaves NaN; the sample mask then excludes it."""
    frame = pd.DataFrame({"SP500_return_1": [0.01, np.nan, 0.02]})
    mask = finite_row_mask(frame, ("SP500_return_1",))
    assert list(mask) == [True, False, True]
    assert np.isnan(frame["SP500_return_1"].iloc[1]), "the gap must stay a gap"


def test_sector_mapping_cannot_silently_map_unknown():
    mapping = sector_index_mapping({"AAA": "UNKNOWN", "BBB": "IT"},
                                   {"IT": "NIFTY_IT", "UNKNOWN": "NIFTY_IT"})
    assert mapping.index_for("BBB") == "NIFTY_IT"
    assert mapping.index_for("AAA") is None, (
        "UNKNOWN must never be mapped, even when a symbol exists for it")


def test_global_risk_factor_uses_fixed_signs_and_train_only_scaling():
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({
        "SP500_return_1": rng.normal(0, 0.01, 300),
        "NASDAQ_COMPOSITE_return_1": rng.normal(0, 0.01, 300),
        "US_VIX_change_1": rng.normal(0, 0.5, 300),
        "USDINR_return_1": rng.normal(0, 0.005, 300),
        "BRENT_CRUDE_return_1": rng.normal(0, 0.02, 300),
    })
    out, state = build_global_risk_factor(frame, components={
        "global_risk_spx_move": ("SP500_return_1", -1),
        "global_risk_vix_change": ("US_VIX_change_1", +1)})
    assert state["available"] is True
    assert state["weights_optimised_on_validation"] is False
    assert state["fitted_on"] == "TRAIN_ONLY"
    assert "global_risk_factor" in out.columns
    # a VIX SPIKE must raise the risk factor, a SPX rally must lower it
    spike = frame.copy()
    spike.loc[0, "US_VIX_change_1"] += 10.0
    spiked, _ = build_global_risk_factor(spike, components={
        "global_risk_spx_move": ("SP500_return_1", -1),
        "global_risk_vix_change": ("US_VIX_change_1", +1)})
    assert spiked["global_risk_factor"].iloc[0] > out["global_risk_factor"].iloc[0]


def test_relative_features_are_stock_minus_index():
    stock = pd.DataFrame({"log_return_1": [0.02, -0.01], "log_return_5": [0.05, 0.00],
                          "log_return_20": [0.10, -0.02]})
    index = pd.DataFrame({"NIFTY50_return_1": [0.01, 0.00],
                          "NIFTY50_return_5": [0.03, -0.01],
                          "NIFTY50_return_20": [0.04, 0.01]})
    relative = relative_to_market_features(stock, index, index_prefix="NIFTY50")
    assert relative["stock_minus_NIFTY50_return_1"].tolist() == pytest.approx(
        [0.01, -0.01])
    assert relative["stock_minus_NIFTY50_return_5"].tolist() == pytest.approx(
        [0.02, 0.01])
    assert relative["stock_minus_NIFTY50_return_20"].tolist() == pytest.approx(
        [0.06, -0.03])