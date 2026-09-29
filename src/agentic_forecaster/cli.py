"""Command-line interface for agentic-forecaster."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_forecaster import PAPER_DOI, __version__
from agentic_forecaster.config import get_env_roots, load_config
from agentic_forecaster.utils import ensure_dir, setup_logging

logger = logging.getLogger("agentic_forecaster.cli")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agentic-forecaster",
        description="Explanation-First Agentic Forecaster for Stock Market",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download-dataset", help="Download the Kaggle dataset")
    dl.add_argument("--out", default=None)

    val = sub.add_parser("validate-submission", help="Validate repository readiness")
    val.add_argument("--repo-root", default=None)
    val.add_argument("--allow-pretraining", action="store_true")

    sub.add_parser("package-submission", help="Export final artefacts into the repository")

    ins = sub.add_parser("inspect-data", help="Inspect the raw dataset")
    ins.add_argument("--raw-root", default=None)

    prep = sub.add_parser("prepare-data", help="Resample/cache configured tickers")
    prep.add_argument("--config", required=True)
    prep.add_argument("--ticker", default=None)

    sub.add_parser("list-tickers", help="List available tickers")

    ver = sub.add_parser("verify-ticker-universe", help="Write results/ticker_availability.csv")
    ver.add_argument("--config", default="configs/paper.yaml")
    ver.add_argument("--out", default="results/ticker_availability.csv")

    prov = sub.add_parser("verify-dataset-provenance", help="Regenerate dataset metadata")
    prov.add_argument("--raw-root", default=None)
    prov.add_argument("--metadata-root", default=None)
    prov.add_argument("--max-files", type=int, default=None)

    train = sub.add_parser("train", help="Train one stock model")
    train.add_argument("--ticker", required=True)
    train.add_argument("--config", required=True)
    train.add_argument("--device", default=None)

    train_all = sub.add_parser("train-all", help="Train every selected stock independently")
    train_all.add_argument("--config", required=True)
    train_all.add_argument("--device", default=None)
    train_all.add_argument("--baselines", action="store_true", help="Also train baselines")
    # --tickers is read by _cmd_train_all().  It was previously undefined, so any
    # invocation raised AttributeError.  Kept optional and comma-separated for
    # consistency with walk-forward / reproduce-paper.
    train_all.add_argument("--tickers", default=None, help="Comma-separated subset")

    ev = sub.add_parser("evaluate", help="Evaluate a saved model bundle")
    ev.add_argument("--model", required=True, help="Model bundle directory")
    ev.add_argument("--data", required=True, help="Processed data directory")
    ev.add_argument("--device", default=None)

    wf = sub.add_parser("walk-forward", help="Run the paper walk-forward reproduction")
    wf.add_argument("--config", required=True)
    wf.add_argument("--device", default=None)
    wf.add_argument("--tickers", default=None, help="Comma-separated subset")
    wf.add_argument("--run-id", default=None)

    pred = sub.add_parser("predict", help="Predict for one stock")
    pred.add_argument("--ticker", required=True)
    pred.add_argument("--model", required=True, help="Model bundle directory")
    pred.add_argument("--data", required=True, help="Processed data directory")
    pred.add_argument("--device", default=None)

    pred_all = sub.add_parser("predict-all", help="Predict for every trained stock")
    pred_all.add_argument("--config", required=True)
    pred_all.add_argument("--device", default=None)
    pred_all.add_argument("--fold", default="fold_0")
    pred_all.add_argument("--out", default=None, help="Optional CSV output path")

    exp = sub.add_parser("explain", help="Explain one prediction")
    exp.add_argument("--model", required=True)
    exp.add_argument("--data", required=True)
    exp.add_argument("--index", type=int, default=-1)
    exp.add_argument("--config", default="configs/paper.yaml")
    exp.add_argument("--device", default=None)

    rep = sub.add_parser("report", help="Generate a report")
    rep.add_argument("--config", required=True)
    rep.add_argument("--ticker", default=None, help="Ticker; required for a real report")
    rep.add_argument("--fold", default="fold_0")
    rep.add_argument("--index", type=int, default=-1, help="Test-split row to report on")
    rep.add_argument("--device", default=None)
    rep.add_argument("--run-dir", default=None)
    rep.add_argument("--demo", action="store_true", help="Use an explicit demo report")

    rep_paper = sub.add_parser("reproduce-paper", help="Run the full paper reproduction")
    rep_paper.add_argument("--config", required=True)
    rep_paper.add_argument("--device", default=None)
    rep_paper.add_argument("--tickers", default=None)
    rep_paper.add_argument("--run-id", default=None)
    rep_paper.add_argument("--export-final-results", action="store_true")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()
    logger.info("DOI: %s", PAPER_DOI)

    handler = {
        "download-dataset": _cmd_download_dataset,
        "validate-submission": _cmd_validate,
        "package-submission": _cmd_package,
        "inspect-data": _cmd_inspect,
        "prepare-data": _cmd_prepare,
        "list-tickers": _cmd_list_tickers,
        "verify-ticker-universe": _cmd_verify_universe,
        "verify-dataset-provenance": _cmd_verify_provenance,
        "train": _cmd_train,
        "train-all": _cmd_train_all,
        "evaluate": _cmd_evaluate,
        "walk-forward": _cmd_walk_forward,
        "predict": _cmd_predict,
        "predict-all": _cmd_predict_all,
        "explain": _cmd_explain,
        "report": _cmd_report,
        "reproduce-paper": _cmd_reproduce,
    }[args.command]
    return handler(args)


# ----------------------------------------------------------------- helpers

def _resolve_device(device: str | None):
    from agentic_forecaster.training import resolve_device

    return resolve_device(device)


def _bundle_dir(model_root: Path, ticker: str, fold: str = "fold_0") -> Path:
    return Path(model_root) / "trained" / ticker / fold


# ------------------------------------------------------------------ cmds

def _cmd_download_dataset(args) -> int:
    from agentic_forecaster.commands.download_dataset import main as dl_main

    return dl_main([args.out] if args.out else [])


def _cmd_validate(args) -> int:
    script = Path(__file__).resolve().parents[2] / "scripts" / "validate_submission.py"
    argv = [str(script)]
    if args.repo_root:
        argv += ["--repo-root", args.repo_root]
    if args.allow_pretraining:
        argv.append("--allow-pretraining")
    import runpy

    sys.argv = argv
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def _cmd_package(args) -> int:
    from agentic_forecaster.config import get_env_roots as _roots
    from agentic_forecaster.packaging import export_final_artifacts

    copied = export_final_artifacts(_roots())
    print(f"package-submission: exported {len(copied)} item(s)")
    for item in copied:
        print(f"  + {item}")
    return 0


def _cmd_inspect(args) -> int:
    from agentic_forecaster.data.dataset import discover_ticker_files

    roots = get_env_roots()
    raw_root = args.raw_root or roots["AGENTIC_RAW_DATA_ROOT"]
    files = discover_ticker_files(raw_root)
    print(f"Raw dataset root: {raw_root}")
    print(f"Discovered {len(files)} unique symbols")
    for ticker, path in sorted(files.items())[:20]:
        print(f"  {ticker}: {path.name}")
    if len(files) > 20:
        print(f"  ... and {len(files) - 20} more")
    return 0


def _cmd_prepare(args) -> int:
    """Resample/cache every configured ticker (real mode) or the synthetic set."""
    from agentic_forecaster.data.agent import DataAgent

    config = load_config(args.config)
    agent = DataAgent(config)
    roots = get_env_roots()
    processed_root = Path(config["data"].get("processed_root")
                          or roots["AGENTIC_PROCESSED_DATA_ROOT"])

    if config["data"].get("synthetic"):
        from agentic_forecaster.data import make_synthetic_dataset

        frames = make_synthetic_dataset(
            n_tickers=int(config["data"].get("n_tickers", 3)),
            n_days=int(config["data"].get("n_days", 600)),
            seed=int(config.get("experiment", {}).get("seed", 42)),
        )
        print(f"Synthetic mode: {len(frames)} tickers")
        for ticker, frame in frames.items():
            ds = agent.run_ticker(ticker, frame)
            out = ensure_dir(processed_root / "synthetic" / ticker)
            ds.save(out)
            print(f"  {ticker}: train={len(ds.train.y)} val={len(ds.val.y)} "
                  f"test={len(ds.test.y)} -> {out}")
        return 0

    if args.ticker:
        symbols = [args.ticker.upper()]
    else:
        symbols = sorted(agent._discover())

    print(f"Preparing {len(symbols)} ticker(s) from {config['data']['raw_root']}")
    ok = 0
    for i, symbol in enumerate(symbols, 1):
        try:
            ds = agent.run(ticker=symbol)
            ok += 1
            print(f"  [{i}/{len(symbols)}] {symbol}: train={len(ds.train.y)} "
                  f"val={len(ds.val.y)} test={len(ds.test.y)}")
        except Exception as exc:
            print(f"  [{i}/{len(symbols)}] {symbol}: FAILED — {exc}")
    print(f"Prepared {ok}/{len(symbols)} ticker(s). Daily cache: {processed_root / 'daily'}")
    return 0


def _cmd_list_tickers(args) -> int:
    from agentic_forecaster.data.agent import DataAgent

    config = load_config("configs/paper.yaml")
    agent = DataAgent(config)
    universe = agent.universe()
    for symbol in universe.requested:
        dataset_symbol = universe.available.get(symbol)
        if dataset_symbol:
            alias = "" if dataset_symbol == symbol else f" (as {dataset_symbol})"
            print(f"{symbol}{alias}")
        else:
            print(f"{symbol} — UNAVAILABLE: {universe.unavailable.get(symbol, '')}")
    return 0


def _cmd_verify_universe(args) -> int:
    import runpy

    script = Path(__file__).resolve().parents[2] / "scripts" / "verify_ticker_universe.py"
    sys.argv = [str(script), "--config", args.config, "--out", args.out]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def _cmd_verify_provenance(args) -> int:
    import runpy

    script = Path(__file__).resolve().parents[2] / "scripts" / "verify_dataset_provenance.py"
    sys.argv = [str(script)]
    if args.raw_root:
        sys.argv += ["--raw-root", args.raw_root]
    if args.metadata_root:
        sys.argv += ["--metadata-root", args.metadata_root]
    if args.max_files:
        sys.argv += ["--max-files", str(args.max_files)]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def _cmd_train(args) -> int:
    from agentic_forecaster.agents.model_agent import ModelAgent
    from agentic_forecaster.data.agent import DataAgent
    from agentic_forecaster.models.checkpoint import save_model_bundle

    config = load_config(args.config)
    ticker = args.ticker.upper()
    dataset = DataAgent(config).run(ticker=ticker)
    fitted = ModelAgent(config).train_ticker(
        ticker, dataset, fold="fold_0", device=args.device
    )
    roots = get_env_roots()
    bundle = save_model_bundle(
        fitted, _bundle_dir(roots["AGENTIC_MODEL_ROOT"], ticker)
    )
    print(f"Trained {ticker} on {len(dataset.train.y)} train rows")
    print(f"  metrics    : {json.dumps(fitted.metrics, default=str)}")
    print(f"  temperature: {fitted.temperature:.6f}")
    print(f"  bundle     : {bundle}")
    return 0


def _cmd_train_all(args) -> int:
    from agentic_forecaster.agents.model_agent import ModelAgent
    from agentic_forecaster.data.agent import DataAgent
    from agentic_forecaster.models.checkpoint import save_model_bundle
    from agentic_forecaster.utils import atomic_json_dump

    config = load_config(args.config)
    agent = DataAgent(config)
    model_agent = ModelAgent(config)
    roots = get_env_roots()
    model_root = Path(roots["AGENTIC_MODEL_ROOT"])

    if args.tickers:
        requested = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        universe = agent.universe()
        requested = list(universe.requested)
    discovered = agent._discover()

    summary: dict = {}
    total = len(requested)
    for i, symbol in enumerate(requested, 1):
        dataset_symbol = (
            symbol if symbol in discovered
            else next((d for s, d in
                       agent.universe().available.items() if s == symbol), None)
        )
        if dataset_symbol is None:
            summary[symbol] = {"status": "unavailable",
                               "reason": agent.universe().unavailable.get(symbol, "not found")}
            print(f"[{i}/{total}] {symbol}: unavailable")
            continue
        print(f"[{i}/{total}] {symbol}")
        try:
            ds = agent.run(ticker=dataset_symbol)
            fitted = model_agent.train_ticker(symbol, ds, fold="fold_0", device=args.device)
            if args.baselines:
                model_agent.train_baselines(symbol, ds, fold="fold_0", device=args.device)
            bundle = save_model_bundle(fitted, _bundle_dir(model_root, symbol))
            summary[symbol] = {"status": "trained", "dataset_symbol": dataset_symbol,
                               "metrics": fitted.metrics, "bundle": str(bundle)}
            print(f"  -> {bundle}")
        except Exception as exc:
            summary[symbol] = {"status": "unavailable", "reason": str(exc)}
            print(f"  unavailable: {exc}")

    atomic_json_dump(summary, model_root / "train_all_summary.json")
    trained = sum(1 for v in summary.values() if v["status"] == "trained")
    print(f"Trained {trained}/{total}. Summary: {model_root / 'train_all_summary.json'}")
    return 0


def _cmd_evaluate(args) -> int:
    from agentic_forecaster.data.agent import ProcessedDataset
    from agentic_forecaster.evaluation import compute_metrics
    from agentic_forecaster.models.checkpoint import load_model_bundle

    dataset = ProcessedDataset.load(args.data)
    fitted = load_model_bundle(args.model, device=args.device)
    p_cal = fitted.predict_proba(dataset.test.X)
    p_raw = fitted.predict_proba_raw(dataset.test.X)
    metrics = compute_metrics(dataset.test.y, p_cal)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print(f"brier_raw: {float(np.mean((p_raw - dataset.test.y) ** 2)):.4f}")
    print(f"brier_calibrated: {float(np.mean((p_cal - dataset.test.y) ** 2)):.4f}")
    return 0


def _cmd_walk_forward(args) -> int:
    from agentic_forecaster.orchestration.walk_forward import run_walk_forward

    config = load_config(args.config)
    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    result = run_walk_forward(config, device=args.device, run_id=args.run_id, tickers=tickers)
    print(json.dumps(result, indent=2, default=str))
    return 0


def _cmd_predict(args) -> int:
    from agentic_forecaster.data.agent import ProcessedDataset
    from agentic_forecaster.models.checkpoint import load_model_bundle

    dataset = ProcessedDataset.load(args.data)
    fitted = load_model_bundle(args.model, device=args.device)
    p_cal = fitted.predict_proba(dataset.test.X)
    mask = dataset.test.tickers == args.ticker
    if not mask.any():
        print(f"No predictions found for {args.ticker}")
        return 1
    idx = np.flatnonzero(mask)[-1]
    value = float(p_cal[idx])
    direction = "UP" if value >= 0.5 else "DOWN"
    print(f"{args.ticker} {dataset.test.dates[idx]}: {direction} "
          f"p_up={value:.4f} confidence={max(value, 1 - value):.4f}")
    return 0


def _cmd_predict_all(args) -> int:
    """Load each trained ticker model and emit a real calibrated prediction."""
    from agentic_forecaster.data.agent import DataAgent
    from agentic_forecaster.models.checkpoint import load_model_bundle

    config = load_config(args.config)
    roots = get_env_roots()
    model_root = Path(roots["AGENTIC_MODEL_ROOT"]) / "trained"
    data_agent = DataAgent(config)

    if not model_root.is_dir():
        print(f"No trained models under {model_root}")
        return 1

    rows = []
    for ticker_dir in sorted(model_root.iterdir()):
        if not ticker_dir.is_dir():
            continue
        bundle = ticker_dir / args.fold
        if not (bundle / "model.pt").is_file():
            continue
        try:
            fitted = load_model_bundle(bundle, device=args.device)
            dataset_symbol = fitted.ticker or ticker_dir.name
            ds = data_agent.run(ticker=dataset_symbol)
            if not len(ds.test.X):
                print(f"{ticker_dir.name}: no test data")
                continue
            p_cal = fitted.predict_proba(ds.test.X)
            p_raw = fitted.predict_proba_raw(ds.test.X)
            idx = len(p_cal) - 1
            value = float(p_cal[idx])
            direction = "UP" if value >= 0.5 else "DOWN"
            rows.append({
                "ticker": ticker_dir.name,
                "date": str(ds.test.dates[idx]),
                "p_up": value,
                "raw_p_up": float(p_raw[idx]),
                "direction": direction,
                "confidence": max(value, 1.0 - value),
            })
        except Exception as exc:
            print(f"{ticker_dir.name}: FAILED — {exc}")

    if not rows:
        print("No predictions produced.")
        return 1

    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)
        print(f"\nWrote {out}")
    return 0


def _cmd_explain(args) -> int:
    from agentic_forecaster.agents.explainer_agent import ExplainerAgent
    from agentic_forecaster.data.agent import ProcessedDataset
    from agentic_forecaster.models.checkpoint import load_model_bundle

    dataset = ProcessedDataset.load(args.data)
    fitted = load_model_bundle(args.model, device=args.device)
    index = args.index if args.index >= 0 else len(dataset.test.X) + args.index
    p_cal = fitted.predict_proba(dataset.test.X[index:index + 1])[0]
    p_raw = fitted.predict_proba_raw(dataset.test.X[index:index + 1])[0]
    explainer = ExplainerAgent(load_config(args.config))
    explanation = explainer.explain_prediction(
        fitted, dataset.test.X[index], fitted.ticker,
        str(dataset.test.dates[index]), float(p_cal), p_up_raw=float(p_raw),
    )
    print(f"{explanation['ticker']} {explanation['date']}: "
          f"{explanation['direction']} confidence={explanation['confidence']:.4f}")
    print(f"method: {explanation['attribution_method']} "
          f"shape={explanation['attribution_shape']}")
    for feat in explanation["top_features"]:
        print(f"  {feat['feature']}: {feat['mean_abs_shap']:.6f}")
    return 0


def _cmd_report(args) -> int:
    """Generate a real report, or an explicitly-labelled demo report."""
    from agentic_forecaster.agents.explainer_agent import ExplainerAgent
    from agentic_forecaster.agents.report_agent import ReportAgent
    from agentic_forecaster.data.agent import DataAgent
    from agentic_forecaster.models.checkpoint import load_model_bundle
    from agentic_forecaster.orchestration.pipeline import REPORT_INDICATORS
    from agentic_forecaster.risk import RiskAgent

    config = load_config(args.config)
    report_agent = ReportAgent(config)
    roots = get_env_roots()
    run_dir = Path(args.run_dir or config["experiment"]["output_dir"]) / "reports"

    if args.demo or not args.ticker:
        if not args.demo:
            print("note: --ticker not given, generating an explicitly-labelled demo report")
        risk = RiskAgent().decide(
            ticker="DEMO", date="2024-01-02", p_up=0.72, close=100.0, atr=2.0
        )
        explanation = {
            "llm_narrative": (
                "SYNTHETIC DEMO REPORT — generated from explicit demo inputs, "
                "not a real market prediction."
            ),
            "top_features": [{"feature": "rsi_14", "mean_abs_shap": 0.01}],
            "attention_evidence": [{"day_offset": 0, "attention_weight": 0.2}],
            "reason_codes": ["demo_report"],
            "attribution_method": "demo",
        }
        result = report_agent.run(
            ticker="DEMO", origin_date="2024-01-01", target_date="2024-01-02",
            latest_close=100.0, p_up_raw=0.68, p_up_calibrated=0.72,
            direction=risk.direction, confidence=risk.confidence,
            confidence_level=risk.confidence_level,
            explanation=explanation, risk_decision=risk,
            metrics={"accuracy": 0.0}, technical_indicators={"atr_14": 2.0},
            model_id="demo", config_id="demo", run_id="demo",
            output_dir=run_dir,
        )
        print(f"Demo report: {result['html']}")
        return 0

    ticker = args.ticker.upper()
    bundle = _bundle_dir(roots["AGENTIC_MODEL_ROOT"], ticker, args.fold)
    if not (bundle / "model.pt").is_file():
        print(f"No trained model for {ticker} at {bundle}. Run train --ticker {ticker} first.")
        return 1

    data_agent = DataAgent(config)
    fitted = load_model_bundle(bundle, device=args.device)
    ds = data_agent.run(ticker=ticker)
    index = args.index if args.index >= 0 else len(ds.test.X) + args.index
    p_cal = float(fitted.predict_proba(ds.test.X[index:index + 1])[0])
    p_raw = float(fitted.predict_proba_raw(ds.test.X[index:index + 1])[0])
    snapshot = data_agent.origin_snapshot(ds.test, index)
    indicators = {k: v for k, v in snapshot.items() if k in REPORT_INDICATORS}
    close = float(snapshot.get("close", 0.0))
    atr = float(snapshot.get("atr_14", 0.0))
    if not (close > 0 and atr > 0):
        print("Unscaled metadata unavailable for this row; cannot build an ATR report.")
        return 1

    explanation = ExplainerAgent(config).explain_prediction(
        fitted, ds.test.X[index], ticker, str(ds.test.dates[index]),
        p_cal, indicator_values=indicators, p_up_raw=p_raw,
    )
    risk = RiskAgent().decide(
        ticker=ticker, date=str(ds.test.dates[index]), p_up=p_cal,
        close=close, atr=atr,
    )
    result = report_agent.run(
        ticker=ticker,
        origin_date=str(ds.test.dates[index]),
        target_date=str(ds.test.target_dates[index]),
        latest_close=close, p_up_raw=p_raw, p_up_calibrated=p_cal,
        direction=risk.direction, confidence=risk.confidence,
        confidence_level=risk.confidence_level,
        explanation=explanation, risk_decision=risk, metrics=fitted.metrics,
        technical_indicators=indicators,
        model_id=f"{ticker}:{args.fold}:attention_lstm",
        config_id=str(config.get("experiment", {}).get("name", "paper")),
        output_dir=run_dir,
    )
    print(f"Report: {result['html']}")
    if result.get("pdf"):
        print(f"PDF   : {result['pdf']}")
    return 0


def _cmd_reproduce(args) -> int:
    from agentic_forecaster.orchestration.walk_forward import run_walk_forward

    config = load_config(args.config)
    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    result = run_walk_forward(config, device=args.device, run_id=args.run_id, tickers=tickers)
    print(json.dumps(result, indent=2, default=str))
    if args.export_final_results:
        from agentic_forecaster.config import get_env_roots as _roots
        from agentic_forecaster.packaging import export_final_artifacts

        copied = export_final_artifacts(_roots(), run_dir=Path(result["run_dir"]))
        print(f"\nExported {len(copied)} artefact(s) into the repository:")
        for item in copied:
            print(f"  + {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
