"""Tests for cross-sectional Precision@3."""

from __future__ import annotations

import pandas as pd

from agentic_forecaster.evaluation.metrics import precision_at_3_cross_sectional


def test_p3_up_and_down():
    df = pd.DataFrame({
        "date": ["2024-01-01"] * 5 + ["2024-01-02"] * 5,
        "ticker": ["A", "B", "C", "D", "E"] * 2,
        "p_up": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
        "y":    [1,   1,   0,   0,   0,   0,   0,   1,   1,   1],
    })
    result = precision_at_3_cross_sectional(df, k=3)
    assert result["n_dates"] == 2
    assert "precision_at_3_up" in result
    assert "precision_at_3_down" in result
    assert "selections" in result


def test_p3_up_perfect():
    df = pd.DataFrame({
        "date": ["2024-01-01"] * 5,
        "ticker": ["A", "B", "C", "D", "E"],
        "p_up": [0.9, 0.8, 0.7, 0.6, 0.5],
        "y":    [1,   1,   1,   0,   0],
    })
    result = precision_at_3_cross_sectional(df, k=3)
    assert result["precision_at_3_up"] == 1.0
    assert abs(result["precision_at_3_down"] - 2 / 3) < 1e-9
