"""Ablations must reuse already-trained models, not retrain them (item 8)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_forecaster.agents.ablation_agent import AblationAgent
from agentic_forecaster.agents.model_agent import ModelAgent
from agentic_forecaster.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def trained():
    config = load_config(REPO_ROOT / "configs" / "demo.yaml")
    from agentic_forecaster.data.agent import DataAgent

    ds = DataAgent(config).run(ticker="SYN00")
    agent = ModelAgent(config)
    primary = agent.train_ticker("SYN00", ds, fold="fold_0", device="cpu")
    baselines = agent.train_baselines("SYN00", ds, fold="fold_0", device="cpu")
    return config, ds, primary, baselines


def test_ablation_reuses_supplied_primary(trained, monkeypatch):
    config, ds, primary, baselines = trained
    agent = AblationAgent(config)

    # Any attempt to train the primary or the baselines must be detected.
    def _boom(*args, **kwargs):
        raise AssertionError("ablation retrained an already-trained model")

    monkeypatch.setattr(agent, "_trainer", _boom, raising=False)
    monkeypatch.setattr(
        __import__("agentic_forecaster.agents.model_agent", fromlist=["ModelAgent"]).ModelAgent,
        "train_ticker",
        staticmethod(_boom),
    )
    monkeypatch.setattr(
        __import__("agentic_forecaster.agents.model_agent", fromlist=["ModelAgent"]).ModelAgent,
        "train_baselines",
        staticmethod(_boom),
    )

    out = agent.run_ticker(
        "SYN00", ds, fold="fold_0", device="cpu",
        already_trained_primary=primary,
        already_trained_baselines=baselines,
    )
    assert out["_reused_primary"] is True
    assert out["attention_lstm"]["accuracy"] == primary.metrics["accuracy"]
    assert out["plain_lstm"]["accuracy"] == baselines["lstm"].metrics["accuracy"]


def test_ablation_variants_present(trained):
    config, ds, primary, baselines = trained
    out = AblationAgent(config).run_ticker(
        "SYN00", ds, fold="fold_0", device="cpu",
        already_trained_primary=primary,
        already_trained_baselines=baselines,
    )
    for name in ("attention_lstm", "plain_lstm", "raw_vs_calibrated",
                 "attention_vs_plain_lstm", "ohlcv_only",
                 "ohlcv_plus_technical"):
        assert name in out, f"missing ablation variant {name}"
    assert out["ohlcv_only"].get("accuracy") is not None


def test_pipeline_passes_trained_models_to_ablation(monkeypatch, tmp_path):
    """Pipeline must hand its trained models to the AblationAgent."""
    from agentic_forecaster.orchestration import Pipeline

    captured = {}

    class _SpyAblation:
        def __init__(self, config):
            pass

        def run_ticker(self, ticker, dataset, fold="fold_0", device=None,
                       already_trained_primary=None,
                       already_trained_baselines=None):
            captured["primary"] = already_trained_primary
            captured["baselines"] = already_trained_baselines
            return {"attention_lstm": {"accuracy": 0.0}}

    import agentic_forecaster.agents.ablation_agent as mod
    monkeypatch.setattr(mod, "AblationAgent", _SpyAblation)

    config = load_config(REPO_ROOT / "configs" / "demo.yaml")
    config["experiment"]["output_dir"] = str(tmp_path / "run")
    config["ablations"] = {"enabled": True}
    Pipeline(config, fold="fold_0").run("SYN00", device="cpu", n_reports=0)

    assert captured.get("primary") is not None, "primary model not passed to ablation"
    assert captured.get("baselines") is not None, "baselines not passed to ablation"
    assert "lstm" in captured["baselines"]
