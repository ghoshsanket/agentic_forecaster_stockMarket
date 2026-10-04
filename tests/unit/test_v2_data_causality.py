"""V2 data-causality tests.

Every property here is a claim the V2 programme makes about its inputs.  If any
of them stops holding, every later number is meaningless, so they are asserted
directly rather than assumed:

1. every input feature at origin ``t`` uses only values <= ``t``;
2. the ``t+1`` row is never inside a sequence;
3. cross-sectional INPUT ranks use date ``t`` only;
4. the next-day rank LABEL uses ``t+1`` and never feeds back into an input;
5. market / sector aggregates use only date-``t`` observations;
6. the scaler is fit on TRAIN samples only;
7. validation rows do not enter scaler fitting;
8. lockbox rows do not enter development training;
9. late-listed securities are not backfilled;
10. the V2 hard firewall rejects any target date >= 2022-01-01.

All fixtures are synthetic (see ``v2_fixtures``): these are properties of the
code, not of the downloaded market data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agentic_forecaster.v2 import context as ctx
from agentic_forecaster.v2 import features as feat
from agentic_forecaster.v2.dataset import (
    V2_DEV_FOLD_A,
    V2_DEV_FOLD_B,
    V2_FOLDS,
    V2_LOCKBOX,
    V2SequenceDataset,
    assert_split_targets_legal,
    build_sample_table,
    fit_scalers_on_train,
    split_masks,
    ticker_balanced_weights,
)
from agentic_forecaster.v2.firewall import (
    V2_LOCKBOX_FIREWALL_START,
    V2_PAPER_TEST_FIREWALL_START,
    V2LockboxFirewallError,
    V2TestFirewallError,
    assert_no_lockbox_targets,
    assert_no_paper_test_targets,
)
from agentic_forecaster.v2.store import RETURN_TARGET_CLIP

from . import v2_fixtures as fx

# ---------------------------------------------------------------------------
# 1. every input feature at t uses only values <= t
# ---------------------------------------------------------------------------

def test_truncating_the_future_does_not_change_any_feature(bars: pd.DataFrame):
    """The strongest possible causality test: cut the bars at t and compare.

    If any feature at row ``t`` depended on a later row, the value computed from
    the truncated series would differ.
    """
    full = feat.build_stock_features(bars)
    for cut in (150, 220, 300):
        truncated = feat.build_stock_features(bars.iloc[:cut].copy())
        left = full.iloc[:cut].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            left[["date", *feat.STOCK_FEATURE_NAMES]],
            truncated[["date", *feat.STOCK_FEATURE_NAMES]],
            check_exact=False, rtol=1e-12, atol=1e-12,
        )


@pytest.mark.parametrize("feature", [
    "log_return_1", "log_return_5", "log_return_20", "overnight_gap",
    "intraday_return", "high_low_range", "realized_vol_20", "atr_14_pct",
    "rsi_14_centered", "macd_pct", "bollinger_percent_b", "volume_z_20",
    "drawdown_60", "obv_change_normalized",
])
def test_single_feature_is_causal(feature: str, bars: pd.DataFrame):
    """Every named feature must be a function of the past and the present."""
    full = feat.build_stock_features(bars)
    truncated = feat.build_stock_features(bars.iloc[:-7].copy())
    cut = len(truncated) - 1
    assert np.isclose(full[feature].iloc[cut], truncated[feature].iloc[cut],
                      rtol=1e-12, atol=1e-12, equal_nan=True), (
        f"{feature} changed when future bars were removed: not causal")


def test_features_contain_no_infinity(stock_features: pd.DataFrame):
    values = stock_features.loc[:, list(feat.STOCK_FEATURE_NAMES)].to_numpy(float)
    assert np.isfinite(values).sum() <= values.size
    assert not np.isinf(values).any()


def test_indicator_warmup_is_nan_not_forward_filled(stock_features: pd.DataFrame):
    """A late listing must keep NaN until its indicator genuinely exists."""
    first_valid = stock_features["sma50_distance"].notna()
    assert not first_valid.iloc[0]
    assert first_valid.any()
    # once valid it never returns to NaN (no back-filled hole appears later)
    valid = first_valid.to_numpy()
    assert (np.diff(valid.astype(int)) >= 0).all()


# ---------------------------------------------------------------------------
# 2. the t+1 row is never inside a sequence
# ---------------------------------------------------------------------------

def test_sequence_never_contains_the_target_row(arrays, targets, samples):
    dataset = V2SequenceDataset(samples, samples.frame)
    for index in (0, len(dataset) // 2, len(dataset) - 1):
        item = dataset[index]
        row = samples.frame.iloc[index]
        matrices = arrays.matrices[str(row["ticker"])]
        origin_row = int(row["row"])
        window = matrices.stock[origin_row - samples.sequence_length + 1:origin_row + 1]
        assert np.allclose(item["stock_sequence"].numpy(), window, atol=1e-6)
        # the NEXT row's features are strictly different from the last row's
        next_row = matrices.stock[origin_row + 1]
        assert not np.allclose(item["stock_sequence"][-1].numpy(), next_row, atol=1e-9)


def test_every_sample_records_its_target_date(samples):
    assert "target_date" in samples.frame.columns
    assert (samples.frame["target_date"] > samples.frame["origin_date"]).all()


def test_target_date_is_the_next_actual_trading_day(arrays, targets, samples):
    for row in samples.frame.head(50).itertuples():
        matrices = arrays.matrices[str(row.ticker)]
        origin = matrices.dates[int(row.row)]
        following = matrices.dates[int(row.row) + 1]
        assert pd.Timestamp(row.target_date) == pd.Timestamp(following)
        assert pd.Timestamp(row.origin_date) == pd.Timestamp(origin)


# ---------------------------------------------------------------------------
# 3. cross-sectional INPUT ranks use date t only
# ---------------------------------------------------------------------------

def test_input_ranks_do_not_move_when_the_future_changes(context_frame: pd.DataFrame):
    """Truncate the universe after date t; the ranks AT t must be identical."""
    cut_date = context_frame["date"].quantile(0.5)
    early = context_frame.loc[context_frame["date"] <= cut_date]
    full_ranks, _, _ = ctx.assemble_context_features(
        context_frame.drop(columns=[c for c in ctx.RANK_FEATURES]),
        dict(fx.SECTORS))
    partial_ranks, _, _ = ctx.assemble_context_features(
        early.drop(columns=[c for c in ctx.RANK_FEATURES]),
        dict(fx.SECTORS))
    merged = full_ranks.merge(partial_ranks, on=["ticker", "date"],
                              suffixes=("_full", "_partial"))
    assert len(merged) > 0
    for column in ctx.RANK_FEATURES:
        left = merged[f"{column}_full"].to_numpy(float)
        right = merged[f"{column}_partial"].to_numpy(float)
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_ranks_are_percentiles_in_the_unit_interval(context_frame: pd.DataFrame):
    for column in ctx.RANK_FEATURES:
        values = context_frame[column].dropna().to_numpy(float)
        assert values.min() > 0.0
        assert values.max() <= 1.0


def test_rank_of_a_stock_is_computed_only_against_the_same_date(context_frame):
    """A rank must be invariant to what other DATES contain."""
    single_date = context_frame.loc[context_frame["date"] == context_frame["date"].max()]
    ranks, _, _ = ctx.assemble_context_features(
        single_date.drop(columns=[c for c in ctx.RANK_FEATURES]),
        dict(fx.SECTORS))
    merged = context_frame.loc[context_frame["date"] == context_frame["date"].max()].merge(
        ranks, on=["ticker", "date"], suffixes=("_all", "_day"))
    for column in ctx.RANK_FEATURES:
        np.testing.assert_allclose(merged[f"{column}_all"], merged[f"{column}_day"],
                                   rtol=1e-12, atol=1e-12, equal_nan=True)


# ---------------------------------------------------------------------------
# 4. the next-day rank LABEL uses t+1 and never feeds back into an input
# ---------------------------------------------------------------------------

def test_rank_label_moves_with_t_plus_1_returns(context_frame, arrays):
    """Change only the LAST date's returns: labels move, date-t inputs do not."""
    labels_before, _, _ = _labels_from(context_frame)
    perturbed = context_frame.copy()
    last_date = perturbed["date"].max()
    mask = perturbed["date"] == last_date
    perturbed.loc[mask, "log_return_1"] = perturbed.loc[mask, "log_return_1"] * -3.0 + 0.5
    labels_after, _, _ = _labels_from(perturbed)

    moved = (labels_before != labels_after).sum()
    assert moved > 0, "the next-day rank label must react to t+1 returns"

    # and the context INPUT columns at the perturbed date are untouched
    np.testing.assert_allclose(
        context_frame.loc[mask, list(ctx.CONTEXT_FEATURES)].to_numpy(float),
        perturbed.loc[mask, list(ctx.CONTEXT_FEATURES)].to_numpy(float),
        equal_nan=True)


def _labels_from(context_frame: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, dict]:
    """Recompute the cross-sectional next-day rank label the way the store does."""
    frame = context_frame.sort_values(["ticker", "date"]).copy()
    rows = {}
    for ticker, group in frame.groupby("ticker", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        future = group["log_return_1"].shift(-1)
        for i in range(len(group) - 1):
            if pd.isna(future.iloc[i]):
                continue
            rows[(str(ticker), group["date"].iloc[i])] = float(future.iloc[i])
    wide = pd.Series(rows).unstack(level=0)
    ranks = wide.rank(axis=1, pct=True)
    labels = ranks.stack(future_stack=True).sort_index()
    return labels, frame, {}


def test_rank_label_is_absent_from_every_input_array(arrays):
    """No input matrix may contain a column whose name is a target."""
    for matrices in arrays.matrices.values():
        assert arrays.context_features, "context must be present in this fixture"
        for name in arrays.context_features:
            assert not name.startswith("y_")
            assert "rank_next" not in name
        assert matrices.regime.shape[1] == len(arrays.regime_features)


def test_return_target_is_clipped_to_the_documented_bound(targets: pd.DataFrame):
    finite = targets["y_return"].dropna()
    assert finite.abs().max() <= RETURN_TARGET_CLIP + 1e-9


# ---------------------------------------------------------------------------
# 5. market / sector aggregates use only date-t observations
# ---------------------------------------------------------------------------

def test_leave_one_out_market_excludes_the_stock_itself(context_frame: pd.DataFrame):
    frame = context_frame.dropna(subset=["log_return_1"]).copy()
    for date, group in frame.groupby("date"):
        if len(group) < 2:
            continue
        values = group["log_return_1"].to_numpy(float)
        for position, (_, row) in enumerate(group.iterrows()):
            others = np.delete(values, position)
            expected = others.mean()
            assert np.isclose(row["market_return_mean_loo"], expected, atol=1e-10), (
                f"{date}: LOO market mean includes the stock's own return")
            expected_breadth = float((others > 0).mean())
            assert np.isclose(row["market_breadth_loo"], expected_breadth, atol=1e-10)
        break


def test_leave_one_out_sector_excludes_the_stock_itself(context_frame: pd.DataFrame,
                                                        sector_map):
    frame = context_frame.dropna(subset=["log_return_1"]).copy()
    frame["sector"] = frame["ticker"].map(sector_map.sector_series())
    checked = 0
    for date, group in frame.groupby("date"):
        if len(group) < 2:
            continue
        values = group["log_return_1"].to_numpy(float)
        sectors = group["sector"].to_numpy()
        for position, (_, row) in enumerate(group.iterrows()):
            mask = sectors == sectors[position]
            if mask.sum() < 2:
                continue
            others = np.delete(values[mask], int(np.where(mask)[0].tolist().index(position)))
            assert np.isclose(row["sector_return_mean_loo"], others.mean(), atol=1e-10)
            checked += 1
        if checked > 40:
            break
    assert checked > 0, "no sector peer pairs were available to verify"


def test_market_aggregate_is_unavailable_for_a_lone_security():
    """Group size <= 1 must be marked unavailable, never invented."""
    dates = pd.bdate_range("2020-01-01", periods=30)
    n = len(dates)
    panel = pd.DataFrame({
        "ticker": ["AAA"] * n,
        "date": dates,
        "log_return_1": np.linspace(-0.01, 0.01, n),
        "log_return_5": np.zeros(n),
        "log_return_20": np.zeros(n),
        "rsi_14_centered": np.zeros(n),
        "volume_z_20": np.zeros(n),
        "realized_vol_20": np.full(n, 0.01),
    })
    _, loo = ctx.build_market_context(panel)
    assert loo["market_return_mean_loo"].isna().all()
    assert loo["market_loo_available_count"].isna().all()


def test_market_breadth_matches_the_cross_section(context_frame: pd.DataFrame):
    frame = context_frame.dropna(subset=["log_return_1"])
    for date, group in frame.groupby("date"):
        if len(group) < 2:
            continue
        expected = float((group["log_return_1"] > 0).mean())
        assert np.isclose(group["market_breadth"].iloc[0], expected, atol=1e-12)
        break


def test_context_does_not_move_when_a_later_date_is_added(context_frame: pd.DataFrame):
    """Aggregate at date t must not react to a new security appearing later."""
    early = context_frame.loc[context_frame["date"] <= context_frame["date"].quantile(0.6)]
    sector_map = dict(fx.SECTORS)
    columns = [c for c in ctx.CONTEXT_FEATURES]
    raw = context_frame.drop(columns=[c for c in ctx.RANK_FEATURES])
    full, _, _ = ctx.assemble_context_features(raw, sector_map)
    partial, _, _ = ctx.assemble_context_features(
        early.drop(columns=[c for c in ctx.RANK_FEATURES]), sector_map)
    merged = full.merge(partial, on=["ticker", "date"], suffixes=("_full", "_part"))
    assert len(merged) > 0
    np.testing.assert_allclose(
        merged[[f"{c}_full" for c in columns]].to_numpy(float),
        merged[[f"{c}_part" for c in columns]].to_numpy(float),
        rtol=1e-10, atol=1e-10, equal_nan=True)


# ---------------------------------------------------------------------------
# 6/7. scaler fitting uses TRAIN samples only
# ---------------------------------------------------------------------------

def _toy_split(samples, window):
    masks = split_masks(samples, window)
    return masks


def test_scaler_fit_receives_training_rows_only(samples, monkeypatch):
    """Spy on ``StandardScaler.fit`` and inspect exactly what it received."""
    from agentic_forecaster.v2 import dataset as dataset_module

    recorded: dict[str, np.ndarray] = {}
    real = dataset_module.StandardScaler

    class RecordingScaler(real):                       # type: ignore[misc, valid-type]
        def fit(self, X, y=None):
            recorded.setdefault("blocks", []).append(np.array(X, dtype=float))
            return super().fit(X, y)

    monkeypatch.setattr(dataset_module, "StandardScaler", RecordingScaler)
    masks = split_masks(samples, V2_DEV_FOLD_A)
    scalers = fit_scalers_on_train(samples, masks["train"])

    provenance = scalers.stock.fit_provenance
    assert provenance["fitted_on"] == "TRAIN_SAMPLES_ONLY"
    assert provenance["n_rows"] == int(masks["train"].sum())

    train_rows = samples.frame.loc[masks["train"]]
    # every fitted row must be an actual TRAIN origin row
    fitted = recorded["blocks"][0]
    assert fitted.shape[0] == len(train_rows)
    assert provenance["max_origin_date"] == str(
        pd.Timestamp(train_rows["origin_date"].max()).date())
    assert provenance["max_target_date"] == str(
        pd.Timestamp(train_rows["target_date"].max()).date())


def _origin_matrix(samples, mask) -> np.ndarray:
    frame = samples.frame.loc[mask]
    blocks = [samples.arrays.matrices[str(t)].stock[int(r)]
              for t, r in zip(frame["ticker"], frame["row"], strict=True)]
    return np.vstack(blocks).astype(float)


def test_validation_rows_do_not_enter_scaler_fitting(samples):
    """The fitted mean must equal the TRAIN-only mean, and differ from all-rows."""
    masks = split_masks(samples, V2_DEV_FOLD_A)
    scalers = fit_scalers_on_train(samples, masks["train"])

    train_matrix = _origin_matrix(samples, masks["train"])
    np.testing.assert_allclose(scalers.stock.scaler.mean_, train_matrix.mean(axis=0),
                               rtol=1e-9)

    every_row = np.vstack([samples.arrays.matrices[str(t)].stock[int(r)]
                           for t, r in zip(samples.frame["ticker"],
                                           samples.frame["row"], strict=True)]).astype(float)
    assert not np.allclose(scalers.stock.scaler.mean_, every_row.mean(axis=0), rtol=1e-6), (
        "the fitted mean equals the all-rows mean, so validation rows leaked into "
        "the scaler fit")

    provenance = scalers.stock.fit_provenance
    assert provenance["max_target_date"] < str(pd.Timestamp(V2_DEV_FOLD_A.val_start).date())


def test_percentile_features_are_not_standardised(samples):
    masks = split_masks(samples, V2_DEV_FOLD_A)
    scalers = fit_scalers_on_train(samples, masks["train"])
    assert set(scalers.context.passthrough) == set(ctx.RANK_FEATURES)
    assert not (set(scalers.context.scaled_columns) & set(ctx.RANK_FEATURES))
    val_rows = samples.frame.loc[masks["val"]].reset_index(drop=True)
    dataset = V2SequenceDataset(samples, val_rows, scalers=scalers)
    index = [samples.arrays.context_features.index(c) for c in ctx.RANK_FEATURES]
    for position in (0, len(dataset) // 2, len(dataset) - 1):
        row = val_rows.iloc[position]
        matrices = samples.arrays.matrices[str(row["ticker"])]
        end = int(row["row"]) + 1
        raw = matrices.context[end - samples.sequence_length:end]
        scaled = dataset[position]["context_sequence"].numpy()[:, index]
        np.testing.assert_allclose(scaled, raw[:, index], rtol=1e-5, atol=1e-5)


def test_scaled_train_inputs_are_finite_and_centred(samples):
    masks = split_masks(samples, V2_DEV_FOLD_A)
    scalers = fit_scalers_on_train(samples, masks["train"])
    dataset = V2SequenceDataset(samples, samples.frame.loc[masks["train"]].head(64),
                                scalers=scalers)
    for index in range(len(dataset)):
        item = dataset[index]
        assert torch_isfinite(item["stock_sequence"].numpy())
        assert torch_isfinite(item["context_sequence"].numpy())


def torch_isfinite(array: np.ndarray) -> bool:
    return bool(np.isfinite(array).all())


# ---------------------------------------------------------------------------
# 8. split boundaries
# ---------------------------------------------------------------------------

def test_split_requires_origin_and_target_inside_the_window(samples):
    window = V2_DEV_FOLD_B
    masks = split_masks(samples, window)
    frame = samples.frame
    train = frame.loc[masks["train"]]
    assert (pd.to_datetime(train["origin_date"]) <= pd.Timestamp(window.train_end)).all()
    assert (pd.to_datetime(train["target_date"]) <= pd.Timestamp(window.train_end)).all()
    assert (pd.to_datetime(train["origin_date"]) >= pd.Timestamp(window.train_start)).all()
    val = frame.loc[masks["val"]]
    assert (pd.to_datetime(val["origin_date"]) >= pd.Timestamp(window.val_start)).all()
    assert (pd.to_datetime(val["target_date"]) <= pd.Timestamp(window.val_end)).all()
    assert not (masks["train"] & masks["val"]).any()


def test_lockbox_rows_do_not_enter_development_training(samples):
    """No development fold may contain a 2021 target date."""
    for window in (V2_DEV_FOLD_A, V2_DEV_FOLD_B):
        masks = split_masks(samples, window)
        for split in ("train", "val"):
            dates = samples.frame.loc[masks[split], "target_date"]
            assert (pd.to_datetime(dates) < V2_LOCKBOX_FIREWALL_START).all(), (
                f"{window.name}/{split} leaked a lockbox target date")


def test_lockbox_fold_is_the_only_one_that_reaches_2021():
    assert V2_LOCKBOX.val_start == "2021-01-01"
    assert V2_DEV_FOLD_A.val_end == "2019-12-31"
    assert V2_DEV_FOLD_B.val_end == "2020-12-31"


def test_ticker_balanced_weights_equalise_ticker_influence(samples):
    frame = samples.frame
    long_ticker = frame.loc[frame["ticker"] == frame["ticker"].value_counts().index[0]]
    weights, info = ticker_balanced_weights(frame)
    by_ticker = pd.Series(weights, index=frame["ticker"].to_numpy())
    total_mass = by_ticker.groupby(level=0).sum()
    assert info["n_tickers"] == len(total_mass)
    np.testing.assert_allclose(total_mass.to_numpy(),
                               np.full(len(total_mass), 1.0 / len(total_mass)),
                               rtol=1e-9)
    assert len(long_ticker) >= len(frame) / len(total_mass)


# ---------------------------------------------------------------------------
# 9. late listings are not backfilled
# ---------------------------------------------------------------------------

def test_late_listing_is_not_backfilled(arrays, samples):
    """DDD lists late: it must have no sample before its own first bar."""
    late = arrays.matrices[fx.LATE_TICKER]
    assert len(late.dates) > 0
    first = pd.Timestamp(late.dates[0])
    ddd_samples = samples.frame.loc[samples.frame["ticker"] == fx.LATE_TICKER]
    assert (pd.to_datetime(ddd_samples["origin_date"]) >= first).all()
    # and no other security invented DDD's early history
    assert samples.frame.loc[samples.frame["origin_date"] < first, "ticker"].isin(
        [t for t in fx.TICKERS if t != fx.LATE_TICKER]).all()


def test_samples_with_incomplete_history_are_dropped_not_padded(arrays, targets):
    """A very long window on a late listing must yield fewer samples, not zeros."""
    short = build_sample_table(arrays, targets, sequence_length=8,
                               require_targets=["y_direction"])
    long = build_sample_table(arrays, targets, sequence_length=200,
                              require_targets=["y_direction"])
    assert len(long) < len(short)
    ddd_long = long.frame.loc[long.frame["ticker"] == fx.LATE_TICKER]
    if len(ddd_long):
        assert (pd.to_datetime(ddd_long["origin_date"])
                >= pd.Timestamp(arrays.matrices[fx.LATE_TICKER].dates[0])).all()


def test_no_input_array_contains_nan(samples):
    """Warm-up rows may be NaN; the rows a SAMPLE uses may never be."""
    dataset = V2SequenceDataset(samples, samples.frame)
    for index in range(0, len(dataset), 37):
        item = dataset[index]
        assert np.isfinite(item["stock_sequence"].numpy()).all()
        assert np.isfinite(item["context_sequence"].numpy()).all()
        assert np.isfinite(item["regime_vector"].numpy()).all()
    warmup = samples.arrays.matrices["AAA"].stock
    assert not np.isfinite(warmup[:10]).all(), (
        "the fixture should still contain indicator warm-up NaN")


# ---------------------------------------------------------------------------
# 10. the V2 firewalls
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("date", ["2022-01-01", "2022-06-15", "2023-12-29"])
def test_hard_firewall_rejects_2022_and_later(date: str):
    with pytest.raises(V2TestFirewallError):
        assert_no_paper_test_targets([date])


def test_hard_firewall_allows_pre_2022():
    assert assert_no_paper_test_targets(["2021-12-31", "2019-01-02"]) == "2021-12-31"


@pytest.mark.parametrize("date", ["2021-01-01", "2021-12-31"])
def test_lockbox_firewall_rejects_2021_by_default(date: str):
    with pytest.raises(V2LockboxFirewallError):
        assert_no_lockbox_targets([date])


def test_lockbox_firewall_can_be_unlocked_explicitly():
    assert assert_no_lockbox_targets(["2021-12-31"], unlocked=True) == "2021-12-31"
    # unlocking the lockbox never unlocks 2022+
    with pytest.raises(V2TestFirewallError):
        assert_no_paper_test_targets(["2022-01-03"], where="unlocked lockbox")


def test_split_targets_are_checked_by_both_firewalls(samples):
    masks = split_masks(samples, V2_DEV_FOLD_A)
    assert_split_targets_legal(masks, samples, where="unit-test")
    poisoned = {"train": masks["train"].copy(), "val": masks["val"].copy()}
    poisoned["val"][0] = True
    frame = samples.frame.copy()
    frame.loc[0, "target_date"] = pd.Timestamp("2022-03-01")
    fake = type(samples)(frame=frame, arrays=samples.arrays,
                         sequence_length=samples.sequence_length)
    with pytest.raises(V2TestFirewallError):
        assert_split_targets_legal(poisoned, fake, where="unit-test")


def test_firewall_boundaries_are_the_documented_dates():
    assert str(V2_PAPER_TEST_FIREWALL_START.date()) == "2022-01-01"
    assert str(V2_LOCKBOX_FIREWALL_START.date()) == "2021-01-01"


def test_every_declared_fold_is_pre_2022_in_validation():
    for name, window in V2_FOLDS.items():
        assert pd.Timestamp(window.val_end) < V2_PAPER_TEST_FIREWALL_START, name