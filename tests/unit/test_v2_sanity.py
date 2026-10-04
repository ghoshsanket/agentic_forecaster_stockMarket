"""V2 sanity-gate tests: the synthetic learnable task and the shuffled control.

These run the real model, the real loss and the real trainer on a synthetic
universe whose next-day direction is a deterministic function of its inputs.  If
the learnable check cannot be learned, every later V2 number would be
uninterpretable, so the gate is itself tested.

Kept tiny (a few epochs on a small synthetic universe) so it stays a unit test.
"""

from __future__ import annotations

import numpy as np
import pytest

from agentic_forecaster.v2.dataset import build_sample_table, split_masks
from agentic_forecaster.v2.sanity import (
    CHANCE,
    LEARNABLE_MIN_ACCURACY,
    SHUFFLED_MAX_ACCURACY,
    build_synthetic_samples,
    make_synthetic_arrays,
    run_synthetic_check,
    sanity_payload,
)


def test_synthetic_universe_is_balanced_and_learnable():
    """The fixture itself must be a fair test: balanced labels, clear signal."""
    arrays, targets = make_synthetic_arrays(n_tickers=3, n_days=600)
    assert targets["y_direction"].mean() == pytest.approx(0.5, abs=0.05)
    samples = build_synthetic_samples(arrays, targets, sequence_length=30)
    assert len(samples) > 500
    frame = samples.frame
    # chronological inside every security (the sample table is ticker-major)
    assert (frame.groupby("ticker")["origin_date"].apply(
        lambda s: s.is_monotonic_increasing)).all()
    # the label is exactly the sign of the last input row's first feature
    for row in frame.head(20).itertuples():
        feature = arrays.matrices[row.ticker].stock[int(row.row), 0]
        assert row.y_direction == int(feature > 0)


def pd_frame(frame):
    import pandas as pd

    return pd.DataFrame(frame)


def test_shuffling_labels_breaks_the_signal():
    arrays, targets = make_synthetic_arrays(n_tickers=3, n_days=600)
    shuffled = build_synthetic_samples(arrays, targets, shuffle_labels=True)
    assert shuffled.diagnostics["shuffled_labels"] is True
    assert abs(shuffled.frame["y_direction"].mean() - 0.5) < 0.1


def test_learnable_task_is_learned_and_shuffled_task_is_not():
    """The gate itself: the learnable task must be learned, the control must not."""
    learnable = run_synthetic_check("synthetic_learnable", epochs=8, n_tickers=2,
                                    n_days=600, device="cpu")
    assert learnable.passed, (
        f"the V2 model failed the synthetic learnability check "
        f"(accuracy {learnable.validation_accuracy_micro:.3f})")
    assert learnable.validation_accuracy_micro >= LEARNABLE_MIN_ACCURACY
    assert learnable.validation_accuracy_micro > CHANCE + 0.2

    shuffled = run_synthetic_check("shuffled_labels", epochs=8, n_tickers=2,
                                   n_days=600, device="cpu", shuffle_labels=True)
    assert shuffled.passed, (
        f"the shuffled-label control still scored "
        f"{shuffled.validation_accuracy_micro:.3f}: the model is not using its inputs")
    assert shuffled.validation_accuracy_micro <= SHUFFLED_MAX_ACCURACY
    assert shuffled.validation_accuracy_micro < CHANCE + 0.1


def test_sanity_payload_reports_the_gate():
    learnable = run_synthetic_check("synthetic_learnable", epochs=8, n_tickers=2,
                                    n_days=600, device="cpu")
    payload = sanity_payload([learnable])
    assert payload["all_passed"] is learnable.passed
    assert payload["no_real_market_data_used"] is True
    assert payload["checks"][0]["name"] == "synthetic_learnable"
    assert "synthetic" in payload["gate"]


def test_synthetic_samples_respect_the_split_rule():
    arrays, targets = make_synthetic_arrays(n_tickers=2, n_days=700)
    samples = build_synthetic_samples(arrays, targets, sequence_length=30)
    from agentic_forecaster.v2.dataset import SplitWindow

    window = SplitWindow("SYNTH", samples.frame["origin_date"].min().strftime("%Y-%m-%d"),
                         "2016-06-30", "2016-07-01", "2016-12-31")
    masks = split_masks(samples, window)
    assert masks["train"].any() and masks["val"].any()
    assert not (masks["train"] & masks["val"]).any()
    frame = samples.frame
    assert (pd_frame(frame).loc[masks["val"], "target_date"].max()
            <= np.datetime64("2016-12-31"))


def test_unknown_sanity_check_name_is_rejected():
    with pytest.raises(ValueError, match="unknown sanity check"):
        run_synthetic_check("something_else", epochs=1, n_tickers=2, n_days=300,
                            device="cpu")


def test_sanity_samples_never_need_a_real_store():
    """The gate must be runnable with no dataset present."""
    arrays, targets = make_synthetic_arrays(n_tickers=2, n_days=400)
    samples = build_synthetic_samples(arrays, targets, sequence_length=20)
    assert set(samples.arrays.ticker_vocab) >= {"SYN00", "SYN01"}
    assert "UNKNOWN" in samples.arrays.sector_vocab
    assert len(samples) > 100
    _ = build_sample_table(arrays, targets, sequence_length=20)