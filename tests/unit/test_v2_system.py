"""V2 checkpoint, sector-map, metrics, ledger and config-level tests.

Covers the parts of the V2 system that are not the model and not the data path:
the persisted artefacts, the static sector identity, the reported metrics, the
append-only ledger, and the config-level firewall that stops a development run
from even being configured to score the lockbox year.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from agentic_forecaster.v2 import checkpoint as ckpt
from agentic_forecaster.v2 import metrics as M
from agentic_forecaster.v2.dataset import GroupFeatureScaler, V2ScalerBundle
from agentic_forecaster.v2.experiment import VARIANT_FLAGS, load_run_config, merge_configs
from agentic_forecaster.v2.ledger import (
    LEDGER_COLUMNS,
    V2_VARIANTS,
    append_experiment,
    ledger_summary,
    read_ledger,
)
from agentic_forecaster.v2.model import ContextualLSTMTransformer
from agentic_forecaster.v2.sectors import (
    UNKNOWN_LABEL,
    build_sector_map,
    build_vocabulary,
    load_sector_map,
    vocabulary_index,
    write_sector_map_csv,
)
from agentic_forecaster.v2.store import RETURN_TARGET_CLIP

from .v2_fixtures import make_model_config

# ---------------------------------------------------------------------------
# checkpoint round trip
# ---------------------------------------------------------------------------

def _scalers(samples, n_rows: int = 8) -> V2ScalerBundle:
    stock = GroupFeatureScaler(samples.arrays.stock_features).fit(
        np.random.default_rng(0).normal(size=(n_rows, samples.arrays.n_stock_features)))
    scaled = [c for c in samples.arrays.context_features
              if c not in set(samples.arrays.percentile_features)]
    context = GroupFeatureScaler(samples.arrays.context_features,
                                 passthrough=samples.arrays.percentile_features).fit(
        np.random.default_rng(1).normal(size=(n_rows, len(scaled))))
    return V2ScalerBundle(stock=stock, context=context)


def test_checkpoint_round_trip_reproduces_predictions(samples, tmp_path):
    config = make_model_config(
        use_adapter=True, use_multitask=True,
        n_stock_features=samples.arrays.n_stock_features,
        n_context_features=samples.arrays.n_context_features,
        n_regime_features=samples.arrays.n_regime_features,
        n_tickers=max(len(samples.arrays.ticker_vocab), 1),
        n_sectors=max(len(samples.arrays.sector_vocab), 1),
        ticker_vocab=list(samples.arrays.ticker_vocab),
        sector_vocab=list(samples.arrays.sector_vocab),
    )
    torch.manual_seed(3)
    model = ContextualLSTMTransformer(config).eval()
    scalers = _scalers(samples)

    ckpt.save_checkpoint(tmp_path / "ckpt", model, scalers=scalers,
                         feature_schema={"stock_features": samples.arrays.stock_features},
                         sector_map_hash="a" * 64, feature_store_hash="b" * 64,
                         training_history={"best_epoch": 3}, manifest={"seed": 42},
                         model_config=config)

    batch = {
        "stock_sequence": torch.randn(3, 6, config.n_stock_features),
        "context_sequence": torch.randn(3, 6, config.n_context_features),
        "regime_vector": torch.randn(3, config.n_regime_features),
        "ticker_id": torch.tensor([0, 1, 2]),
        "sector_id": torch.tensor([1, 2, 0]),
    }
    restored = ckpt.load_checkpoint(tmp_path / "ckpt", device="cpu")
    worst = ckpt.assert_predictions_match(model, restored.model, batch)
    assert worst == 0.0

    assert restored.ticker_vocab == config.ticker_vocab
    assert restored.sector_vocab == config.sector_vocab
    assert restored.training_history["best_epoch"] == 3
    assert ckpt.checkpoint_hashes(tmp_path / "ckpt") == {
        "sector_map_hash": "a" * 64, "feature_store_hash": "b" * 64}


def test_checkpoint_saves_every_required_file(samples, tmp_path):
    config = make_model_config(
        n_stock_features=samples.arrays.n_stock_features,
        n_context_features=samples.arrays.n_context_features,
        n_regime_features=samples.arrays.n_regime_features,
        ticker_vocab=list(samples.arrays.ticker_vocab),
        sector_vocab=list(samples.arrays.sector_vocab),
        n_tickers=max(len(samples.arrays.ticker_vocab), 1),
        n_sectors=max(len(samples.arrays.sector_vocab), 1),
    )
    torch.manual_seed(4)
    model = ContextualLSTMTransformer(config)
    ckpt.save_checkpoint(tmp_path / "full", model, scalers=_scalers(samples),
                         feature_schema={}, sector_map_hash="s", feature_store_hash="f",
                         training_history={}, manifest={}, model_config=config)
    for name in ckpt.CHECKPOINT_FILES:
        assert (tmp_path / "full" / name).is_file(), name
    assert not (tmp_path / "full" / "meta_config.json").exists()


def test_meta_checkpoint_adds_the_meta_files(samples, tmp_path):
    from agentic_forecaster.v2.meta import MetaConfig

    config = make_model_config(use_adapter=True,
                               n_stock_features=samples.arrays.n_stock_features,
                               n_context_features=samples.arrays.n_context_features,
                               n_regime_features=samples.arrays.n_regime_features,
                               ticker_vocab=list(samples.arrays.ticker_vocab),
                               sector_vocab=list(samples.arrays.sector_vocab),
                               n_tickers=max(len(samples.arrays.ticker_vocab), 1),
                               n_sectors=max(len(samples.arrays.sector_vocab), 1))
    torch.manual_seed(5)
    model = ContextualLSTMTransformer(config)
    meta_config = MetaConfig().to_dict()
    ckpt.save_checkpoint(tmp_path / "meta", model, scalers=_scalers(samples),
                         feature_schema={}, sector_map_hash="s", feature_store_hash="f",
                         training_history={}, manifest={},
                         meta_config=meta_config,
                         meta_initialization={n: p.detach().clone() for n, p in
                                             model.named_parameters()
                                             if n in set(model.adaptable_parameter_names())},
                         model_config=config)
    for name in ckpt.META_FILES:
        assert (tmp_path / "meta" / name).is_file(), name
    restored = ckpt.load_checkpoint(tmp_path / "meta")
    assert restored.meta_config["label"] == "REPTILE_STYLE_HEAD_ADAPTER"
    assert restored.meta_initialization


def test_scalers_survive_the_round_trip(samples, tmp_path):
    config = make_model_config(
        n_stock_features=samples.arrays.n_stock_features,
        n_context_features=samples.arrays.n_context_features,
        n_regime_features=samples.arrays.n_regime_features,
        ticker_vocab=list(samples.arrays.ticker_vocab),
        sector_vocab=list(samples.arrays.sector_vocab),
        n_tickers=max(len(samples.arrays.ticker_vocab), 1),
        n_sectors=max(len(samples.arrays.sector_vocab), 1),
    )
    model = ContextualLSTMTransformer(config)
    scalers = _scalers(samples)
    ckpt.save_checkpoint(tmp_path / "s", model, scalers=scalers, feature_schema={},
                         sector_map_hash="s", feature_store_hash="f",
                         training_history={}, manifest={}, model_config=config)
    restored = ckpt.load_checkpoint(tmp_path / "s")
    matrix = np.random.default_rng(2).normal(size=(5, samples.arrays.n_stock_features))
    np.testing.assert_allclose(scalers.stock.transform(matrix),
                               restored.stock_scaler.transform(matrix), rtol=1e-9)
    assert restored.context_scaler.passthrough == scalers.context.passthrough


def test_missing_checkpoint_file_is_reported(tmp_path):
    (tmp_path / "partial").mkdir()
    (tmp_path / "partial" / "model.pt").write_bytes(b"")
    with pytest.raises(ckpt.V2CheckpointError, match="missing"):
        ckpt.load_checkpoint(tmp_path / "partial")


def test_sector_vocabulary_must_contain_unknown():
    with pytest.raises(ckpt.V2CheckpointError, match="UNKNOWN"):
        ckpt.vocabulary_frames(["AAA", "BBB"], ["TECH"])
    vocab = ckpt.vocabulary_frames(["AAA", "BBB"], [UNKNOWN_LABEL, "TECH"])
    assert vocab[1][0] == UNKNOWN_LABEL


# ---------------------------------------------------------------------------
# sector map
# ---------------------------------------------------------------------------

def test_sector_map_reports_unmapped_tickers(tmp_path):
    csv = tmp_path / "ind_nifty50list.csv"
    csv.write_text(
        "Company Name,Industry,Symbol,Series,ISIN Code\n"
        "AAA Ltd,Information Technology,AAA,EQ,INE1\n"
        "BBB Ltd,Financial Services,BBB,EQ,INE2\n")
    sector_map = build_sector_map(["AAA", "BBB", "ZZZ"], csv)
    assert sector_map.unmapped_tickers == ["ZZZ"]
    assert sector_map.sector_for("ZZZ") == UNKNOWN_LABEL
    coverage = sector_map.coverage()
    assert coverage["n_tickers"] == 3
    assert coverage["n_mapped"] == 2
    assert coverage["coverage_fraction"] == pytest.approx(2 / 3)
    assert coverage["unmapped_tickers"] == ["ZZZ"]


def test_sector_map_records_full_provenance(tmp_path):
    csv = tmp_path / "ind_nifty50list.csv"
    csv.write_text("Company Name,Industry,Symbol,Series,ISIN Code\n"
                   "AAA Ltd,Information Technology,AAA,EQ,INE1\n")
    sector_map = build_sector_map(["AAA"], csv, retrieved_at="2026-01-01T00:00:00+00:00")
    row = sector_map.frame.iloc[0]
    for column in ("ticker", "company_name", "industry", "broad_sector", "source",
                   "source_sha256", "retrieved_at"):
        assert column in sector_map.frame.columns
        assert str(row[column])
    assert len(str(row["source_sha256"])) == 64
    assert row["retrieved_at"] == "2026-01-01T00:00:00+00:00"
    assert row["broad_sector"] == "IT_AND_TELECOM"


def test_unknown_industry_is_not_guessed(tmp_path):
    csv = tmp_path / "list.csv"
    csv.write_text("Company Name,Industry,Symbol,Series,ISIN Code\n"
                   "AAA Ltd,Intergalactic Widgets,AAA,EQ,INE1\n")
    sector_map = build_sector_map(["AAA"], csv)
    assert sector_map.frame.iloc[0]["broad_sector"] == UNKNOWN_LABEL
    assert "Intergalactic Widgets" in sector_map.unmapped_industries


def test_sector_map_csv_round_trip(sector_map, tmp_path):
    path = write_sector_map_csv(sector_map, tmp_path / "sector_map.csv")
    restored = load_sector_map(path)
    assert restored.sha256() == sector_map.sha256()
    assert restored.sector_series() == sector_map.sector_series()


def test_vocabulary_reserves_the_unknown_slot():
    vocab = build_vocabulary(["TECH", "FIN"])
    assert vocab[0] == UNKNOWN_LABEL
    assert vocabulary_index(vocab, "TECH") == vocab.index("TECH")
    assert vocabulary_index(vocab, "NOT_A_SECTOR") == vocab.index(UNKNOWN_LABEL)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def _prediction_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tickers = [f"T{i}" for i in range(6)]
    dates = pd.bdate_range("2020-01-01", periods=n // len(tickers))
    rows = []
    for ticker in tickers:
        for date in dates:
            rows.append({"ticker": ticker, "target_date": date,
                         "y_true": int(rng.random() < 0.5),
                         "p_up": float(rng.random())})
    return pd.DataFrame(rows)


def test_direction_metrics_report_counts_and_macro():
    frame = _prediction_frame()
    metrics = M.direction_metrics(frame)
    assert metrics["n"] == len(frame)
    assert metrics["n_tickers"] == 6
    assert 0.0 <= metrics["accuracy_micro"] <= 1.0
    assert 0.0 <= metrics["accuracy_macro_ticker"] <= 1.0
    assert set(metrics["per_ticker"]) == set(frame["ticker"].unique())


def test_macro_accuracy_equals_the_mean_of_per_ticker_accuracies():
    frame = _prediction_frame(seed=3)
    per_ticker = frame.groupby("ticker").apply(
        lambda g: ((g["p_up"] >= 0.5).astype(int) == g["y_true"]).mean(),
        include_groups=False)
    assert M.macro_ticker_accuracy(frame) == pytest.approx(per_ticker.mean())


def test_perfect_and_inverted_predictions():
    frame = _prediction_frame(seed=1)
    frame["p_up"] = np.where(frame["y_true"] == 1, 0.9, 0.1)
    metrics = M.direction_metrics(frame)
    assert metrics["accuracy_micro"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["brier"] < 0.05
    frame["p_up"] = 1.0 - frame["p_up"]
    assert M.direction_metrics(frame)["accuracy_micro"] == 0.0


def test_coverage_curve_is_monotone_in_coverage_and_reports_counts():
    frame = _prediction_frame(seed=2)
    curve = M.coverage_curve(frame)
    assert list(curve["coverage_target"]) == list(M.COVERAGE_LEVELS)
    assert curve["n"].is_monotonic_decreasing
    assert curve["coverage_retained"].is_monotonic_decreasing
    for _, row in curve.iterrows():
        assert row["coverage_retained"] == pytest.approx(row["n"] / len(frame))
        assert 0.0 <= row["accuracy"] <= 1.0
        assert 0.0 <= row["f1"] <= 1.0


def test_coverage_selects_the_most_confident_predictions():
    frame = _prediction_frame(seed=4)
    frame["p_up"] = np.where(frame["y_true"] == 1, 0.99, 0.49)  # correct + confident
    curve = M.coverage_curve(frame)
    top = curve.iloc[0]
    assert top["accuracy"] > 0.9
    assert top["coverage_retained"] == 1.0


def test_selective_accuracy_requires_a_meaningful_sample():
    frame = _prediction_frame(seed=5, n=60)
    frame["p_up"] = np.where(frame["y_true"] == 1, 0.99, 0.01)
    result = M.max_coverage_at_accuracy(frame, 0.65, min_observations=200,
                                       min_coverage=0.10)
    assert result["coverage"] is None, "a 60-row frame must not yield a 65% claim"
    assert result["reason"] != "meaningful"

    big = _prediction_frame(seed=5, n=1200)
    big["p_up"] = np.where(big["y_true"] == 1, 0.99, 0.01)
    allowed = M.max_coverage_at_accuracy(big, 0.65, min_observations=200,
                                         min_coverage=0.10)
    assert allowed["coverage"] == 1.0
    assert allowed["n"] >= 200


def test_precision_at_3_is_computed_per_date():
    frame = _prediction_frame(seed=6)
    p3 = M.precision_at_k(frame, k=3)
    assert p3["k"] == 3
    assert p3["n_dates"] > 0
    assert 0.0 <= p3["precision_at_3_up"] <= 1.0
    assert 0.0 <= p3["precision_at_3_down"] <= 1.0
    assert len(p3["selections"]) == p3["n_dates"]


def test_precision_at_3_rewards_perfect_ranking():
    """Exactly three risers per date, perfectly ranked: both P@3 must be 1.0."""
    rows = []
    for date in pd.bdate_range("2020-01-01", periods=20):
        for i in range(6):
            up = i >= 3                       # T3, T4, T5 rise; T0..T2 fall
            rows.append({"ticker": f"T{i}", "target_date": date,
                         "y_true": int(up),
                         "p_up": 0.9 if up else 0.1 + 0.01 * i})
    frame = pd.DataFrame(rows)
    p3 = M.precision_at_k(frame, k=3)
    assert p3["n_dates"] == 20
    assert p3["precision_at_3_up"] == 1.0
    assert p3["precision_at_3_down"] == 1.0


def test_precision_at_3_is_chance_for_a_random_ranking():
    frame = _prediction_frame(seed=7)
    p3 = M.precision_at_k(frame, k=3)
    assert 0.2 < p3["precision_at_3_up"] < 0.8


def test_return_metrics_report_correlations_and_raw_errors():
    frame = pd.DataFrame({
        "y_true": [1.0, -1.0, 0.5, -0.5, 2.0, -2.0],
        "prediction": [0.8, -0.9, 0.4, -0.4, 1.5, -1.8],
        "raw_next_return": [0.01, -0.02, 0.005, -0.005, 0.03, -0.04],
        "predicted_next_return": [0.008, -0.018, 0.004, -0.004, 0.024, -0.036],
    })
    metrics = M.return_metrics(frame)
    assert metrics["return_mae"] > 0
    assert metrics["return_rmse"] >= metrics["return_mae"]
    assert metrics["return_spearman"] > 0.9
    assert "raw_return_mae" in metrics


def test_rank_metrics_report_daily_ic_even_when_negative():
    rows = []
    for date in pd.bdate_range("2020-01-01", periods=8):
        for i in range(5):
            rows.append({"target_date": date, "ticker": f"T{i}",
                         "rank_prediction": 1.0 - i / 5.0, "y_rank": 1.0 - i / 5.0})
    good = pd.DataFrame(rows)
    positive = M.rank_metrics(good)
    assert positive["mean_rank_ic"] == pytest.approx(1.0)

    inverted = good.copy()
    inverted["rank_prediction"] = 1.0 - inverted["rank_prediction"]
    negative = M.rank_metrics(inverted)
    assert negative["mean_rank_ic"] == pytest.approx(-1.0), (
        "a negative rank IC must be reported, not hidden")


def test_multi_head_agreement_is_diagnostic_only():
    frame = pd.DataFrame({
        "y_true": [1, 1, 0, 0, 1],
        "direction_logit": [2.0, 1.0, -1.0, -2.0, 0.5],
        "return_prediction": [0.4, 0.3, -0.2, -0.4, 0.1],
        "rank_prediction": [0.8, 0.7, 0.3, 0.2, 0.55],
    })
    result = M.multi_head_agreement(frame)
    assert result["available"] is True
    assert result["coverage"] == pytest.approx(1.0)
    assert result["accuracy_when_agreeing"] == pytest.approx(1.0)
    assert "diagnostic only" in result["note"]

    single = frame[["y_true", "direction_logit"]]
    assert M.multi_head_agreement(single)["available"] is False


def test_majority_baseline_is_the_train_majority():
    assert M.train_majority_baseline([1, 1, 1, 0]) == pytest.approx(0.75)
    assert M.train_majority_baseline([0, 0]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ledger
# ---------------------------------------------------------------------------

def test_ledger_appends_and_refuses_a_2022_claim(tmp_path):
    path = tmp_path / "ledger.csv"
    row = {"variant": "V2-A", "fold": "V2_DEV_FOLD_A", "seed": 42,
           "validation_accuracy_macro": 0.51, "test_2022_2023_evaluated": "false"}
    appended = append_experiment(row, path=path)
    assert appended["test_2022_2023_evaluated"] == "false"
    assert len(read_ledger(path)) == 1

    with pytest.raises(ValueError, match="2022"):
        append_experiment({**row, "test_2022_2023_evaluated": "true"}, path=path)
    assert len(read_ledger(path)) == 1, "a refused row must not be written"


def test_ledger_columns_cover_the_required_fields():
    required = ("experiment_id", "timestamp", "git_commit", "variant", "fold", "seed",
                "tickers", "data_variant", "universe_id", "config_sha256",
                "feature_store_sha256", "sequence_length", "n_parameters", "max_epochs",
                "best_epoch", "validation_accuracy_micro", "validation_accuracy_macro",
                "validation_f1", "validation_auc", "validation_brier",
                "validation_ece", "train_majority_baseline", "precision_at_3_up",
                "precision_at_3_down", "return_mae", "return_spearman", "rank_ic",
                "test_2022_2023_evaluated")
    assert set(required).issubset(set(LEDGER_COLUMNS))


def test_ledger_summary_counts_rows(tmp_path):
    path = tmp_path / "ledger.csv"
    append_experiment({"variant": "V2-A", "fold": "V2_DEV_FOLD_A"}, path=path)
    append_experiment({"variant": "V2-B", "fold": "V2_DEV_FOLD_B", "seed": 11},
                      path=path)
    summary = ledger_summary(path)
    assert summary["total_experiments"] == 2
    assert summary["rows_claiming_2022_2023"] == 0
    assert summary["variants"] == ["V2-A", "V2-B"]


# ---------------------------------------------------------------------------
# configs and the variant ladder
# ---------------------------------------------------------------------------

def test_variant_ladder_is_a_strict_addition_chain():
    """Each variant must be the previous one plus named components."""
    assert list(VARIANT_FLAGS) == list(V2_VARIANTS)
    previous: set[str] = set()
    for variant, flags in VARIANT_FLAGS.items():
        enabled = {k for k, v in flags.items() if v}
        if previous:
            assert previous.issubset(enabled), (
                f"{variant} removes components from its predecessor: "
                f"{previous - enabled}")
            assert enabled - previous, f"{variant} adds nothing to its predecessor"
        previous = enabled


def test_config_merge_is_a_deep_merge():
    merged = merge_configs({"a": {"x": 1, "y": 2}, "b": 3}, {"a": {"y": 9}})
    assert merged == {"a": {"x": 1, "y": 9}, "b": 3}


def test_variant_configs_declare_their_own_letter():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    for variant in V2_VARIANTS:
        letter = variant.split("-")[1].lower()
        matches = sorted((repo / "configs" / "v2").glob(f"v2_{letter}_*.yaml"))
        assert matches, f"no config file for {variant}"
        text = matches[0].read_text()
        assert f"variant: {variant}" in text
        assert "base: base.yaml" in text


def test_base_config_declares_the_fixed_programme_settings():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    base = (repo / "configs" / "v2" / "base.yaml").read_text()
    for token in ("learning_rate: 0.0003", "batch_size: 256", "max_epochs: 30",
                  "early_stopping_patience: 6", "gradient_clip_norm: 1.0",
                  "sequence_length: 60", "return_target_clip: 5.0",
                  "direction: 1.00", "return: 0.50", "rank: 0.25",
                  "balanced_by_ticker: true", "meta_step_size: 0.10",
                  "inner_steps: 3", "max_date: \"2021-12-31\""):
        assert token in base, f"base config is missing {token!r}"


def test_load_run_config_refuses_the_lockbox_without_the_switch():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    with pytest.raises(AssertionError, match="lockbox"):
        load_run_config(repo / "configs" / "v2" / "v2_a_shared_lstm.yaml",
                        fold="V2_LOCKBOX")


def test_load_run_config_resolves_the_variant_and_fold():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    config = load_run_config(repo / "configs" / "v2" / "v2_c_contextual.yaml",
                             fold="V2_DEV_FOLD_B", seed=11, device="cpu")
    assert config.variant == "V2-C"
    assert config.fold == "V2_DEV_FOLD_B"
    assert config.seed == 11
    assert config.device == "cpu"
    assert config.sequence_length == 60
    assert config.window.val_start == "2020-01-01"
    assert config.sha256 == config.payload["config_sha256"]
    assert config.payload["experiment"]["is_original_paper_model"] is False
    assert config.payload["experiment"]["classification"] == "NEW_EXPERIMENTAL_ARCHITECTURE"
    assert config.loss_weights.direction == 1.0
    assert config.loss_weights.return_value == 0.5
    assert config.loss_weights.rank == 0.25
    assert config.return_target_clip == RETURN_TARGET_CLIP


def test_load_run_config_rejects_an_unknown_variant():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    with pytest.raises(ValueError, match="unknown V2 variant"):
        load_run_config(repo / "configs" / "v2" / "v2_a_shared_lstm.yaml",
                        variant="V2-Z")


def test_development_configs_never_mention_a_test_window():
    """No V2 config may declare a 2022+ evaluation window."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    for path in sorted((repo / "configs" / "v2").glob("*.yaml")):
        text = path.read_text()
        for forbidden in ("test_start", "test_end", "2022-", "2023-"):
            assert forbidden not in text, f"{path.name} references {forbidden}"


def test_v2_never_imports_the_paper_reference():
    """V2 summary code must not be able to compare against a published metric."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    forbidden = ("from agentic_forecaster.paper_reference",
                 "import paper_reference",
                 "paper_reference.",
                 "PAPER_REFERENCE =")
    for path in sorted((repo / "src" / "agentic_forecaster" / "v2").glob("*.py")):
        text = path.read_text()
        for pattern in forbidden:
            assert pattern not in text, f"{path.name} references {pattern}"
    for name in ("summarize_v2_dev.py", "run_v2_experiment.py", "run_v2_lockbox.py",
                 "run_v2_sanity.py", "build_v2_context.py", "run_v2_dev_pilot.sh"):
        text = (repo / "scripts" / name).read_text()
        for pattern in forbidden:
            assert pattern not in text, f"{name} references {pattern}"


def test_v2_does_not_touch_the_final_test_switch():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    v2_files = list((repo / "src" / "agentic_forecaster" / "v2").glob("*.py"))
    v2_files += [repo / "scripts" / name for name in
                 ("run_v2_experiment.py", "run_v2_lockbox.py",
                  "summarize_v2_dev.py", "run_v2_sanity.py",
                  "build_v2_context.py", "run_v2_dev_pilot.sh")]
    for path in v2_files:
        text = path.read_text()
        # the switch may be NAMED in a comment, but it may never be SET or called
        assert "FINAL_TEST=" not in text, path.name
        assert "run_recovered_paper.py\"" not in text, path.name
        assert "python scripts/run_recovered_paper.py" not in text, path.name


def test_shipped_sector_map_is_not_guessed():
    """configs/v2/sector_map.yaml must exist and be explicit about UNKNOWN."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    path = repo / "configs" / "v2" / "sector_map.yaml"
    assert path.is_file()
    payload = json.loads(json.dumps(
        __import__("yaml").safe_load(path.read_text())))
    assert payload["n_tickers"] == 50
    assert payload["unmapped_tickers"], "WIPRO must be reported, not hidden"
    assert all(entry["ticker"] for entry in payload["entries"])
    assert len(payload["entries"]) == 50