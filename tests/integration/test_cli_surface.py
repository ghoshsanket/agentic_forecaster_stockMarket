"""CLI surface tests: report, prepare-data, device handling, predict-all (items 4, 19, 20)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from agentic_forecaster.cli import build_parser, main
from agentic_forecaster.training import resolve_device

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_all_documented_commands_parse():
    parser = build_parser()
    for argv in (
        ["train", "--ticker", "RELIANCE", "--config", "configs/paper.yaml", "--device", "cpu"],
        ["train-all", "--config", "configs/paper.yaml", "--device", "auto"],
        ["evaluate", "--model", "m", "--data", "d"],
        ["predict", "--ticker", "X", "--model", "m", "--data", "d"],
        ["predict-all", "--config", "configs/paper.yaml", "--device", "cuda:0"],
        ["walk-forward", "--config", "configs/paper.yaml", "--device", "cpu"],
        ["reproduce-paper", "--config", "configs/paper.yaml",
         "--device", "cpu", "--export-final-results"],
        ["report", "--config", "configs/demo.yaml", "--demo"],
        ["prepare-data", "--config", "configs/paper.yaml"],
        ["validate-submission", "--allow-pretraining"],
        ["package-submission"],
        ["verify-ticker-universe", "--config", "configs/paper.yaml"],
        ["verify-dataset-provenance"],
    ):
        args = parser.parse_args(argv)
        assert args.command == argv[0]


def test_reproduce_paper_accepts_device_and_export():
    args = build_parser().parse_args(
        ["reproduce-paper", "--config", "configs/paper.yaml",
         "--device", "cuda:0", "--export-final-results"]
    )
    assert args.device == "cuda:0"
    assert args.export_final_results is True


def test_resolve_device_never_returns_auto():
    for value in (None, "auto", "cpu"):
        device = resolve_device(value)
        assert device.type in {"cpu", "cuda"}
        assert str(device) != "auto"


def test_load_model_bundle_device_auto(monkeypatch, tmp_path):
    """load_model_bundle(device='auto') must not pass 'auto' to torch."""

    from agentic_forecaster.agents.model_agent import FittedModel
    from agentic_forecaster.models import AttentionLSTM
    from agentic_forecaster.models.checkpoint import load_model_bundle, save_model_bundle

    model = AttentionLSTM(input_size=3, hidden_size=4, num_layers=1, dropout=0.0)
    fitted = FittedModel(
        name="attention_lstm", kind="torch", model=model, ticker="T", fold="fold_0",
        metrics={"accuracy": 0.5}, calibration={"method": "temperature", "temperature": 1.3},
        temperature=1.3, feature_names=["a", "b", "c"],
        train_config={"hidden_size": 4, "num_layers": 1, "dropout": 0.0},
    )
    from sklearn.preprocessing import StandardScaler

    fitted._scaler = StandardScaler().fit(np.random.randn(10, 3))
    fitted._background = np.random.randn(8, 30, 3).astype(np.float32)
    bundle = save_model_bundle(fitted, tmp_path / "b")

    loaded = load_model_bundle(bundle, device="auto")
    assert str(next(loaded.model.parameters()).device) != "auto"
    assert loaded.temperature == 1.3
    assert loaded._background is not None
    assert loaded._scaler is not None


def test_cli_report_demo_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_OUTPUT_ROOT", str(tmp_path))
    rc = main(["report", "--config", str(REPO_ROOT / "configs" / "demo.yaml"), "--demo"])
    assert rc == 0
    produced = list((tmp_path).rglob("DEMO_*.html"))
    assert produced, "demo report html not produced"


def test_cli_prepare_data_synthetic(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_OUTPUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("AGENTIC_PROCESSED_DATA_ROOT", str(tmp_path / "proc"))
    rc = main(["prepare-data", "--config", str(REPO_ROOT / "configs" / "demo.yaml")])
    assert rc == 0
    assert (tmp_path / "proc" / "synthetic").is_dir()
