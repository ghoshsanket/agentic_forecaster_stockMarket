"""Command-line interface for agentic-forecaster."""

from __future__ import annotations

import argparse
import logging

import numpy as np

from agentic_forecaster import PAPER_DOI, __version__
from agentic_forecaster.config import load_config
from agentic_forecaster.utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agentic-forecaster",
        description="Explanation-First Agentic Forecaster for Stock Market",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("download-dataset", help="Download the Kaggle dataset")
    sub.add_parser("validate-submission", help="Validate repository readiness")
    sub.add_parser("package-submission", help="Export final artefacts into the repository")

    inspect = sub.add_parser("inspect-data", help="Inspect the raw dataset")
    inspect.add_argument("--raw-root", default=None)

    prepare = sub.add_parser("prepare-data", help="Prepare/resample data from raw to processed")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--out", default=None)

    sub.add_parser("list-tickers", help="List available tickers")

    train = sub.add_parser("train", help="Train a single stock model")
    train.add_argument("--ticker", required=True)
    train.add_argument("--config", required=True)
    train.add_argument("--device", default=None)

    train_all = sub.add_parser("train-all", help="Train all stock models independently")
    train_all.add_argument("--config", required=True)
    train_all.add_argument("--device", default=None)

    evaluate = sub.add_parser("evaluate", help="Evaluate a trained model")
    evaluate.add_argument("--model", required=True, help="Path to model bundle directory")
    evaluate.add_argument("--data", required=True, help="Path to processed data directory")
    evaluate.add_argument("--device", default=None)

    walk_forward = sub.add_parser("walk-forward", help="Run paper walk-forward evaluation")
    walk_forward.add_argument("--config", required=True)
    walk_forward.add_argument("--device", default=None)

    predict = sub.add_parser("predict", help="Run prediction for a single stock")
    predict.add_argument("--ticker", required=True)
    predict.add_argument("--model", required=True, help="Path to model bundle directory")
    predict.add_argument("--data", required=True, help="Path to processed data directory")
    predict.add_argument("--device", default=None)

    predict_all = sub.add_parser("predict-all", help="Run predictions for all stocks")
    predict_all.add_argument("--config", required=True)
    predict_all.add_argument("--device", default=None)

    explain = sub.add_parser("explain", help="Generate explanation for a prediction")
    explain.add_argument("--model", required=True)
    explain.add_argument("--data", required=True)
    explain.add_argument("--index", type=int, default=0)
    explain.add_argument("--device", default=None)

    report = sub.add_parser("report", help="Generate a report")
    report.add_argument("--config", required=True)
    report.add_argument("--run-dir", default=None)

    reproduce = sub.add_parser("reproduce-paper", help="Run the full paper reproduction")
    reproduce.add_argument("--config", required=True)
    reproduce.add_argument("--device", default=None)
    reproduce.add_argument("--export-final-results", action="store_true")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()
    logging.getLogger("agentic_forecaster").info("DOI: %s", PAPER_DOI)

    if args.command == "download-dataset":
        from agentic_forecaster.commands.download_dataset import main as dl_main
        return dl_main([])

    if args.command == "validate-submission":
        from agentic_forecaster.commands.validate_submission import main as val_main
        return val_main([])

    if args.command == "package-submission":
        from agentic_forecaster.commands.package_submission import main as pkg_main
        return pkg_main([])

    if args.command == "inspect-data":
        from agentic_forecaster.config import get_env_roots
        from agentic_forecaster.data.dataset import discover_ticker_files
        roots = get_env_roots()
        raw_root = args.raw_root or roots["AGENTIC_RAW_DATA_ROOT"]
        files = discover_ticker_files(raw_root)
        print(f"Raw dataset root: {raw_root}")
        print(f"Found {len(files)} ticker(s):")
        for ticker, path in sorted(files.items()):
            print(f"  {ticker}: {path}")
        return 0

    if args.command == "prepare-data":
        from agentic_forecaster.data.agent import DataAgent
        config = load_config(args.config)
        agent = DataAgent(config)
        dataset = agent.run()
        out = args.out or config.get("data", {}).get("processed_root")
        if out:
            dataset.save(out)
            print(f"Processed data saved to: {out}")
        print(f"Train: {len(dataset.train.y)}, Val: {len(dataset.val.y)}, Test: {len(dataset.test.y)}")
        return 0

    if args.command == "list-tickers":
        from agentic_forecaster.config import get_env_roots
        from agentic_forecaster.data.dataset import discover_ticker_files
        roots = get_env_roots()
        files = discover_ticker_files(roots["AGENTIC_RAW_DATA_ROOT"])
        for ticker in sorted(files):
            print(ticker)
        return 0

    if args.command == "train":
        from agentic_forecaster.agents.model_agent import ModelAgent
        from agentic_forecaster.data.agent import DataAgent
        config = load_config(args.config)
        data_agent = DataAgent(config)
        dataset = data_agent.run(ticker=args.ticker)
        model_agent = ModelAgent(config)
        fitted = model_agent.train_ticker(args.ticker, dataset, device=args.device)
        from agentic_forecaster.models.checkpoint import save_model_bundle
        roots = __import__("agentic_forecaster.config", fromlist=["get_env_roots"]).get_env_roots()
        model_root = roots["AGENTIC_MODEL_ROOT"]
        bundle_dir = save_model_bundle(fitted, f"{model_root}/trained/{args.ticker}/fold_0")
        print(f"Trained model for {args.ticker}")
        print(f"Metrics: {fitted.metrics}")
        print(f"Bundle: {bundle_dir}")
        return 0

    if args.command == "train-all":
        from agentic_forecaster.agents.model_agent import ModelAgent
        from agentic_forecaster.config import get_env_roots
        from agentic_forecaster.data.agent import DataAgent
        from agentic_forecaster.models.checkpoint import save_model_bundle
        config = load_config(args.config)
        data_agent = DataAgent(config)
        model_agent = ModelAgent(config)
        roots = get_env_roots()
        model_root = roots["AGENTIC_MODEL_ROOT"]
        ticker_files = data_agent._discover()
        tickers = sorted(ticker_files.keys())
        summary = {}
        for idx, ticker in enumerate(tickers, 1):
            print(f"[{idx}/{len(tickers)}] {ticker}")
            try:
                dataset = data_agent.run(ticker=ticker)
                fitted = model_agent.train_ticker(ticker, dataset, device=args.device)
                bundle_dir = save_model_bundle(fitted, f"{model_root}/trained/{ticker}/fold_0")
                summary[ticker] = {"status": "trained", "bundle": str(bundle_dir)}
            except Exception as exc:
                summary[ticker] = {"status": "unavailable", "reason": str(exc)}
                print(f"  unavailable: {exc}")
        import json
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "evaluate":
        from agentic_forecaster.data.agent import ProcessedDataset
        from agentic_forecaster.evaluation import compute_metrics
        from agentic_forecaster.models.checkpoint import load_model_bundle
        dataset = ProcessedDataset.load(args.data)
        fitted = load_model_bundle(args.model, device=args.device)
        p_cal = fitted.predict_proba(dataset.test.X)
        metrics = compute_metrics(dataset.test.y, p_cal)
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        return 0

    if args.command == "walk-forward":
        from agentic_forecaster.orchestration.walk_forward import run_walk_forward
        config = load_config(args.config)
        result = run_walk_forward(config, device=args.device)
        print(f"Walk-forward complete: {result}")
        return 0

    if args.command == "predict":
        from agentic_forecaster.data.agent import ProcessedDataset
        from agentic_forecaster.models.checkpoint import load_model_bundle
        dataset = ProcessedDataset.load(args.data)
        fitted = load_model_bundle(args.model, device=args.device)
        p_cal = fitted.predict_proba(dataset.test.X)
        mask = dataset.test.tickers == args.ticker
        ticker_proba = p_cal[mask]
        if len(ticker_proba) == 0:
            print(f"No predictions found for {args.ticker}")
            return 1
        latest = ticker_proba[-1]
        direction = "UP" if latest >= 0.5 else "DOWN"
        conviction = float(latest if direction == "UP" else 1 - latest)
        print(f"{args.ticker}: {direction} (conviction: {conviction:.2%})")
        return 0

    if args.command == "predict-all":
        from agentic_forecaster.config import get_env_roots
        from agentic_forecaster.models.checkpoint import load_model_bundle
        roots = get_env_roots()
        model_root = roots["AGENTIC_MODEL_ROOT"]
        import os
        for ticker_dir in sorted(os.listdir(f"{model_root}/trained")):
            bundle = f"{model_root}/trained/{ticker_dir}/fold_0"
            if not os.path.isdir(bundle):
                continue
            try:
                fitted = load_model_bundle(bundle, device=args.device)
                print(f"{ticker_dir}: loaded (T={fitted.temperature:.4f})")
            except Exception as exc:
                print(f"{ticker_dir}: FAILED ({exc})")
        return 0

    if args.command == "explain":
        from agentic_forecaster.data.agent import ProcessedDataset
        from agentic_forecaster.models.checkpoint import load_model_bundle
        dataset = ProcessedDataset.load(args.data)
        fitted = load_model_bundle(args.model, device=args.device)
        x = dataset.test.X[args.index]
        p_cal = fitted.predict_proba(x[np.newaxis, :])[0]
        from agentic_forecaster.agents.explainer_agent import ExplainerAgent
        config = load_config("configs/paper.yaml")
        explainer = ExplainerAgent(config)
        explanation = explainer.explain_prediction(
            fitted, x, fitted.ticker, str(dataset.test.dates[args.index]), p_cal
        )
        print(f"Explanation for {fitted.ticker} at index {args.index}:")
        for feat in explanation.get("top_features", []):
            print(f"  {feat['feature']}: {feat['mean_abs_shap']:.4f}")
        return 0

    if args.command == "report":
        from agentic_forecaster.agents.report_agent import ReportAgent
        config = load_config(args.config)
        agent = ReportAgent(config)
        run_dir = args.run_dir or config.get("experiment", {}).get("output_dir", "./outputs/run")
        result = agent.run(
            ticker="REPORT", origin_date="2024-01-01", target_date="2024-01-02",
            latest_close=100.0, p_up_raw=0.6, p_up_calibrated=0.65,
            direction="UP", confidence=0.65, confidence_level="MEDIUM",
            explanation={"top_features": [], "llm_narrative": "Summary report"},
            risk_decision=None, metrics={}, technical_indicators={},
            output_dir=run_dir,
        )
        print(f"Report generated: {result['html']}")
        return 0

    if args.command == "reproduce-paper":
        from agentic_forecaster.orchestration.walk_forward import run_walk_forward
        config = load_config(args.config)
        result = run_walk_forward(config, device=args.device)
        print(f"Reproduction complete: {result}")
        if args.export_final_results:
            from agentic_forecaster.commands.package_submission import main as pkg_main
            return pkg_main([])
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
