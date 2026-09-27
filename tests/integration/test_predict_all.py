"""predict-all must actually predict, not merely load models (item 18)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from agentic_forecaster.config import load_config
from agentic_forecaster.models.checkpoint import save_model_bundle

REPO_ROOT = Path(__file__).resolve().parents[2]


def _train_demo_bundle(tmp_path):

    from agentic_forecaster.agents.model_agent import ModelAgent
    from agentic_forecaster.data.agent import DataAgent

    config = load_config(REPO_ROOT / "configs" / "demo.yaml")
    ds = DataAgent(config).run(ticker="SYN00")
    fitted = ModelAgent(config).train_ticker("SYN00", ds, fold="fold_0", device="cpu")
    fitted._scaler = ds.scaler
    fitted._background = np.random.default_rng(0).normal(
        size=(8, ds.train.X.shape[1], ds.train.X.shape[2])
    ).astype(np.float32)
    return config, ds, fitted


def test_predict_all_emits_predictions(tmp_path, monkeypatch, capsys):
    from agentic_forecaster.cli import main

    _config, _ds, fitted = _train_demo_bundle(tmp_path)
    model_root = tmp_path / "models"
    save_model_bundle(fitted, model_root / "trained" / "SYN00" / "fold_0")

    # predict-all reads the configured data agent, so point the config at the
    # synthetic demo dataset and give it a ticker universe of one symbol.
    cfg_path = tmp_path / "demo_one.yaml"
    cfg_path.write_text(
        (REPO_ROOT / "configs" / "demo.yaml").read_text()
        + "\ndata:\n  synthetic: true\n  n_tickers: 1\n  n_days: 600\n"
          "  sequence_length: 30\n  train_fraction: 0.7\n  val_fraction: 0.15\n"
    )
    monkeypatch.setenv("AGENTIC_MODEL_ROOT", str(model_root))

    rc = main(["predict-all", "--config", str(cfg_path), "--device", "cpu",
               "--out", str(tmp_path / "preds.csv")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "SYN00" in out
    assert "UP" in out or "DOWN" in out
    assert (tmp_path / "preds.csv").is_file()
    text = (tmp_path / "preds.csv").read_text()
    for column in ("ticker", "date", "p_up", "direction", "confidence"):
        assert column in text
