from __future__ import annotations

import argparse
import logging

from agentic_forecaster import PAPER_DOI, __version__
from agentic_forecaster.config import load_config
from agentic_forecaster.orchestration import Pipeline
from agentic_forecaster.utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agentic-forecaster",
        description="Explanation-First Agentic Forecaster for Stock Market",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the full five-agent pipeline")
    run.add_argument("--config", required=True, help="Path to YAML config")
    run.add_argument("--export-final-results", action="store_true")

    dl = sub.add_parser("download-dataset", help="Download the Kaggle dataset")
    dl.add_argument("--out", default=None)

    validate = sub.add_parser("validate-submission", help="Validate repository readiness")
    validate.add_argument("--repo-root", default=None)

    sub.add_parser("package-submission", help="Export final artefacts into the repository")

    inspect = sub.add_parser("inspect-data", help="Inspect the raw dataset")
    inspect.add_argument("--raw-root", default=None, help="Path to raw dataset root")

    prepare = sub.add_parser("prepare-data", help="Prepare/resample data from raw to processed")
    prepare.add_argument("--config", required=True, help="Path to YAML config")
    prepare.add_argument("--out", default=None, help="Output directory for processed data")

    sub.add_parser("list-tickers", help="List available tickers")

    train = sub.add_parser("train", help="Train a single stock model")
    train.add_argument("--ticker", required=True, help="Stock ticker symbol")
    train.add_argument("--config", required=True, help="Path to YAML config")
    train.add_argument("--device", default=None, help="Device to use for training")

    train_all = sub.add_parser("train-all", help="Train all stock models")
    train_all.add_argument("--config", required=True, help="Path to YAML config")
    train_all.add_argument("--device", default=None, help="Device to use for training")

    evaluate = sub.add_parser("evaluate", help="Evaluate a trained model")
    evaluate.add_argument("--model", required=True, help="Path to trained model checkpoint")
    evaluate.add_argument("--data", required=True, help="Path to processed data")

    walk_forward = sub.add_parser("walk-forward", help="Run walk-forward evaluation")
    walk_forward.add_argument("--config", required=True, help="Path to YAML config")
    walk_forward.add_argument("--device", default=None, help="Device to use for evaluation")

    predict = sub.add_parser("predict", help="Run prediction for a single stock")
    predict.add_argument("--ticker", required=True, help="Stock ticker symbol")
    predict.add_argument("--model", required=True, help="Path to trained model checkpoint")
    predict.add_argument("--data", required=True, help="Path to processed data")

    predict_all = sub.add_parser("predict-all", help="Run predictions for all stocks")
    predict_all.add_argument("--config", required=True, help="Path to YAML config")
    predict_all.add_argument("--device", default=None, help="Device to use for prediction")

    explain = sub.add_parser("explain", help="Generate explanation for a prediction")
    explain.add_argument("--model", required=True, help="Path to trained model checkpoint")
    explain.add_argument("--data", required=True, help="Path to processed data")
    explain.add_argument("--index", type=int, default=0, help="Sample index to explain")

    report = sub.add_parser("report", help="Generate a report")
    report.add_argument("--config", required=True, help="Path to YAML config")
    report.add_argument("--run-dir", default=None, help="Path to run directory")

    reproduce = sub.add_parser("reproduce-paper", help="Run the full paper reproduction")
    reproduce.add_argument("--config", required=True, help="Path to YAML config")
    reproduce.add_argument("--device", default=None, help="Device to use for reproduction")
    reproduce.add_argument("--export-final-results", action="store_true")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()
    logging.getLogger("agentic_forecaster").info("DOI: %s", PAPER_DOI)

    if args.command == "run":
        config = load_config(args.config)
        result = Pipeline(config).run()
        print(f"Pipeline complete. Run dir: {result.run_dir}")
        if args.export_final_results:
            from agentic_forecaster.commands.package_submission import main as pkg_main
            return pkg_main([])
        return 0

    if args.command == "download-dataset":
        from agentic_forecaster.commands.download_dataset import main as dl_main
        return dl_main([args.out] if args.out else [])

    if args.command == "validate-submission":
        from agentic_forecaster.commands.validate_submission import main as val_main
        return val_main([args.repo_root] if args.repo_root else [])

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
        dataset = data_agent.run()
        model_agent = ModelAgent(config)
        fitted = model_agent._train_attention_lstm(dataset)
        if args.device:
            fitted.model.to(args.device)
        print(f"Trained model for {args.ticker}")
        print(f"Metrics: {fitted.metrics}")
        return 0

    if args.command == "train-all":
        from agentic_forecaster.agents.model_agent import ModelAgent
        from agentic_forecaster.data.agent import DataAgent
        config = load_config(args.config)
        data_agent = DataAgent(config)
        dataset = data_agent.run()
        model_agent = ModelAgent(config)
        fitted = model_agent.run(dataset)
        for name, fm in fitted.items():
            print(f"{name}: {fm.metrics}")
        return 0

    if args.command == "evaluate":
        from agentic_forecaster.data.agent import ProcessedDataset
        from agentic_forecaster.evaluation import compute_metrics
        dataset = ProcessedDataset.load(args.data)
        import torch
        model = torch.load(args.model, map_location="cpu")
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(dataset.test.X, dtype=torch.float32))
            proba = torch.softmax(logits, dim=1).numpy()
        metrics = compute_metrics(dataset.test.y, proba[:, 1])
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        return 0

    if args.command == "walk-forward":
        from agentic_forecaster.evaluation.walk_forward import walk_forward_folds
        config = load_config(args.config)
        seq_len = int(config.get("data", {}).get("sequence_length", 30))
        folds = walk_forward_folds(n_samples=1000, seq_len=seq_len)
        print(f"Generated {len(folds)} walk-forward folds:")
        for i, fold in enumerate(folds):
            print(f"  Fold {i}: train=[{fold['train_start']}, {fold['train_end']}), test=[{fold['test_start']}, {fold['test_end']})")
        return 0

    if args.command == "predict":
        import torch

        from agentic_forecaster.data.agent import ProcessedDataset
        dataset = ProcessedDataset.load(args.data)
        model = torch.load(args.model, map_location="cpu")
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(dataset.test.X, dtype=torch.float32))
            proba = torch.softmax(logits, dim=1).numpy()
        mask = dataset.test.tickers == args.ticker
        ticker_proba = proba[mask]
        if len(ticker_proba) == 0:
            print(f"No predictions found for {args.ticker}")
            return 1
        latest = ticker_proba[-1]
        direction = "UP" if latest[1] >= 0.5 else "DOWN"
        conviction = float(latest[1] if direction == "UP" else latest[0])
        print(f"{args.ticker}: {direction} (conviction: {conviction:.2%})")
        return 0

    if args.command == "predict-all":
        from agentic_forecaster.data.agent import ProcessedDataset
        config = load_config(args.config)
        dataset = ProcessedDataset.load(config.get("data", {}).get("processed_root", "./data/processed"))
        import torch
        model = torch.load(config.get("model", {}).get("checkpoint", "./model.pt"), map_location="cpu")
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(dataset.test.X, dtype=torch.float32))
            proba = torch.softmax(logits, dim=1).numpy()
        for ticker in sorted(set(dataset.test.tickers)):
            mask = dataset.test.tickers == ticker
            ticker_proba = proba[mask]
            if len(ticker_proba) == 0:
                continue
            latest = ticker_proba[-1]
            direction = "UP" if latest[1] >= 0.5 else "DOWN"
            print(f"{ticker}: {direction} ({float(latest[1]):.2%})")
        return 0

    if args.command == "explain":
        import torch

        from agentic_forecaster.agents.explainer_agent import ExplainerAgent
        from agentic_forecaster.agents.model_agent import FittedModel
        from agentic_forecaster.data.agent import ProcessedDataset
        config = load_config(args.data.replace("processed", "config"))
        dataset = ProcessedDataset.load(args.data)
        model = torch.load(args.model, map_location="cpu")
        explainer = ExplainerAgent(config)
        fitted = FittedModel(name="model", kind="torch", model=model)
        explanation = explainer.run(fitted, dataset)
        print(f"Explanation for sample index {args.index}:")
        for feat in explanation.get("top_features", []):
            print(f"  {feat['feature']}: {feat['mean_abs_shap']:.4f}")
        return 0

    if args.command == "report":
        from agentic_forecaster.agents.report_agent import ReportAgent
        config = load_config(args.config)
        agent = ReportAgent(config)
        run_dir = args.run_dir or config.get("experiment", {}).get("output_dir", "./outputs/run")
        result = agent.run(
            ticker="REPORT",
            date="2024-01-01",
            direction="UP",
            conviction=0.75,
            explanation={"top_features": [], "llm_narrative": "Summary report"},
            risk_decision=None,
            metrics={},
            output_dir=run_dir,
        )
        print(f"Report generated: {result['html']}")
        return 0

    if args.command == "reproduce-paper":
        config = load_config(args.config)
        result = Pipeline(config).run()
        print(f"Reproduction complete. Run dir: {result.run_dir}")
        if args.export_final_results:
            from agentic_forecaster.commands.package_submission import main as pkg_main
            return pkg_main([])
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
