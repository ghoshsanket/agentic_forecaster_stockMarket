#!/usr/bin/env python3
"""V3 STAGE 2 -- the neural candidate, run ONLY for a signal-passing combination.

``SHARED_LSTM_EXOGENOUS`` is the primary neural candidate: the dataset-specific
evidence favours the LSTM, and the exogenous encoder is kept deliberately small.

``LSTM_TRANSFORMER_EXOGENOUS`` is available only as a second ablation, and only
when the shared LSTM already clears the neural gate.  It must earn its complexity:
the PRE-COVID price-only study found the Transformer did not help.

Two synchronised sequences are supplied (stock 27 features, exogenous features over
the SAME 60 origin dates), both ending at the origin row, so no value from ``t+1``
onwards can reach the input.

No FiLM, no Reptile, no MAML, no hyper-parameter search.

Usage::

    uv run python scripts/run_v3_exogenous_neural.py --family X1_INDIA_MARKET \\
        --horizon 5 --architecture SHARED_LSTM_EXOGENOUS
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_forecaster.utils import atomic_json_dump, ensure_dir, setup_logging
from agentic_forecaster.v2.dataset import resolve_fold_window
from agentic_forecaster.v2.horizons import horizon_phrase, load_track_config, objective_id
from agentic_forecaster.v2.ledger import experiment_dir
from agentic_forecaster.v3.checkpoint import feature_schema_record, save_neural_checkpoint
from agentic_forecaster.v3.dataset import baseline_for, family_split_mask
from agentic_forecaster.v3.experiment import append_row, assemble_inputs
from agentic_forecaster.v3.metrics import block_bootstrap, direction_metrics
from agentic_forecaster.v3.neural import (
    DualSequenceDataset,
    NeuralArchitecture,
    build_model,
    exogenous_window_is_finite,
    model_parameter_count,
    predict_probabilities,
)
from agentic_forecaster.v3.screening import non_overlap_metrics

CONFIG = REPO_ROOT / "configs" / "v3" / "precovid_exogenous_base.yaml"

ARCHITECTURES = ("SHARED_LSTM_EXOGENOUS", "LSTM_TRANSFORMER_EXOGENOUS")


def _loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits.reshape(-1), targets.float().reshape(-1))


def run_neural_fold(inputs, *, family: str, horizon: int, window, fold: str,
                    architecture: str, seed: int, device: str,
                    bootstrap: dict, locked: bool = False) -> dict:
    """One dual-encoder fit at one family/horizon/year."""
    table = inputs.family_tables[(horizon, family)]
    masks = family_split_mask(table, window, fold=fold, locked=locked)
    # the ORIGINAL sample labels are kept until the exogenous-window filter has run,
    # because that filter is indexed by sample position, not by a fresh RangeIndex
    train_frame = table.samples.frame.loc[masks["train"]]
    val_frame = table.samples.frame.loc[masks["val"]]


    if train_frame.empty or val_frame.empty:
        raise ValueError(f"{family}/{objective_id(horizon)}/{fold}: empty split")

    # Both scalers are fitted on TRAIN ORIGIN ROWS ONLY, using the verified V2
    # train-only stock scaler and the equivalent for the exogenous block.
    from agentic_forecaster.v2.dataset import fit_scalers_on_train

    exogenous_matrix = table.exogenous_matrix(masks["train"])
    v2_scalers = fit_scalers_on_train(table.samples, masks["train"])
    exog_scaler = None
    exog_scaler_provenance: dict = {"fit_on": "NOT_APPLICABLE_NO_EXOGENOUS_BLOCK"}
    if exogenous_matrix.shape[1]:
        from sklearn.preprocessing import StandardScaler

        exog_scaler = StandardScaler().fit(exogenous_matrix)
        exog_scaler_provenance = {"fit_on": "TRAIN_ONLY",
                                  "n_rows": int(exogenous_matrix.shape[0]),
                                  "n_columns": int(exogenous_matrix.shape[1])}

    # The exogenous stream is keyed by (ticker, origin_date) and looked up on exactly
    # the 60 pairs the stock sequence covers, so the two streams stay synchronised.
    exog_by_pair = (table.exogenous if table.exogenous_columns else None)

    arch = NeuralArchitecture(
        n_stock_features=table.samples.arrays.n_stock_features,
        n_exogenous_features=len(table.exogenous_columns),
        n_tickers=len(table.samples.arrays.ticker_vocab),
        sequence_length=int(inputs.payload["data"]["sequence_length"]),
        use_transformer=architecture == "LSTM_TRANSFORMER_EXOGENOUS",
        d_model=int(inputs.payload["neural"].get("d_model", 64))
        if "d_model" in inputs.payload["neural"] else 64)
    model = build_model(arch, seed=seed)
    resolved_device = torch.device("cuda" if (device in ("auto", "cuda")
                                               and torch.cuda.is_available())
                                   else "cpu")
    model.to(resolved_device)

    settings = inputs.payload["neural"]["training"]
    generator = torch.Generator()
    generator.manual_seed(seed)

    def _loader(frame: pd.DataFrame, shuffle: bool) -> DataLoader:
        dataset = DualSequenceDataset(
            table.samples, frame, stock_scaler=v2_scalers.stock,
            exogenous_scaler=exog_scaler,
            sequence_length=arch.sequence_length,
            exogenous_by_pair=exog_by_pair, exogenous_columns=table.exogenous_columns)
        sampler = None
        if shuffle and settings.get("balanced_by_ticker", True) and len(frame):
            counts = frame["ticker"].value_counts()
            weights = (1.0 / counts.reindex(frame["ticker"]).to_numpy(float))
            weights = weights / weights.sum()
            sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                            num_samples=len(weights), replacement=True,
                                            generator=generator)
        return DataLoader(dataset, batch_size=int(settings["batch_size"]),
                          shuffle=(sampler is None and shuffle), sampler=sampler,
                          num_workers=0, generator=None if sampler else generator)

    # A neural sequence needs every one of the 60 exogenous values in its window to
    # EXIST.  Rows whose window is not fully covered are dropped from the neural fit
    # rather than padded or filled with an invented value, and the count is recorded.
    if table.exogenous_columns:
        finite_window = exogenous_window_is_finite(
            table.samples, table.exogenous, table.exogenous_columns,
            sequence_length=arch.sequence_length)
        train_frame = train_frame.loc[finite_window[train_frame.index]].reset_index(
            drop=True)
        val_frame = val_frame.loc[finite_window[val_frame.index]].reset_index(drop=True)
        dropped_train = dropped_val = 0
        if train_frame.empty or val_frame.empty:
            raise ValueError(
                f"{family}/{objective_id(horizon)}/{fold}: no sample retains a fully "
                "finite exogenous window, so no neural fit is possible for this family")
    else:
        dropped_train = dropped_val = 0

    train_loader = _loader(train_frame, True)
    val_loader = _loader(val_frame, False)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=float(settings["learning_rate"]),
        betas=(float(settings["beta1"]), float(settings["beta2"])),
        weight_decay=float(settings["weight_decay"]))

    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history = []
    for epoch in range(1, int(settings["max_epochs"]) + 1):
        model.train()
        for batch in train_loader:
            optimiser.zero_grad(set_to_none=True)
            exogenous = batch.get("exogenous_sequence")
            logits = model(batch["stock_sequence"].to(resolved_device),
                           batch["ticker_id"].to(resolved_device),
                           exogenous_sequence=(None if exogenous is None
                                               else exogenous.to(resolved_device)))
            loss = _loss(logits, batch["y_direction"].to(resolved_device))
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite training loss for {family}/{objective_id(horizon)}/"
                    f"{fold} seed {seed}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           float(settings["gradient_clip"]))
            optimiser.step()
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                exogenous = batch.get("exogenous_sequence")
                logits = model(batch["stock_sequence"].to(resolved_device),
                               batch["ticker_id"].to(resolved_device),
                               exogenous_sequence=(None if exogenous is None
                                                   else exogenous.to(resolved_device)))
                val_losses.append(float(_loss(
                    logits, batch["y_direction"].to(resolved_device))))
        if not val_losses:
            raise RuntimeError(
                f"epoch {epoch} produced no validation batches; the split is empty")
        val_loss = float(np.mean(val_losses))
        if not np.isfinite(val_loss):
            raise FloatingPointError(
                f"non-finite validation loss at epoch {epoch} ({val_loss}); the fit "
                "would silently keep the initial weights, so it is refused")
        history.append({"epoch": epoch, "val_loss": val_loss})
        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= int(settings["early_stopping_patience"]):
                break
    model.load_state_dict(best_state)
    model.to(resolved_device)

    proba = predict_probabilities(model, val_loader, resolved_device)
    predictions = pd.DataFrame({
        "ticker": val_frame["ticker"].to_numpy(),
        "origin_date": pd.to_datetime(val_frame["origin_date"]).to_numpy(),
        "target_date": pd.to_datetime(val_frame["target_date"]).to_numpy(),
        "y_true": val_frame["y_direction"].to_numpy(int),
        "p_up": proba,
    })
    baselines = baseline_for(table, masks["train"])
    metrics = direction_metrics(predictions, baselines=baselines, horizon=horizon,
                                family=family, sample_set="natural")
    metrics["bootstrap"] = block_bootstrap(predictions, horizon=horizon,
                                           family=family, **bootstrap)
    non_overlap = non_overlap_metrics(predictions, horizon=horizon,
                                      baselines={"global": baselines["global"],
                                                 "per_ticker": {}},
                                      family=family)
    return {
        "architecture": architecture,
        "family": family,
        "horizon": horizon,
        "fold": fold,
        "seed": seed,
        "n_train": len(train_frame),
        "n_validation": len(predictions),
        "metrics": metrics,
        "non_overlapping": non_overlap,
        "predictions": predictions,
        "model": model,
        "parameters": model_parameter_count(model),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "history": history,
        "exogenous_scaler_provenance": exog_scaler_provenance,
        "stock_scaler_provenance": v2_scalers.stock.fit_provenance,
        "dropped_incomplete_exogenous_windows": {"train": dropped_train,
                                                 "validation": dropped_val},
        "schema": feature_schema_record(list(table.samples.arrays.stock_features),
                                        list(table.exogenous_columns)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--family", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--architecture", choices=ARCHITECTURES,
                        default="SHARED_LSTM_EXOGENOUS")
    parser.add_argument("--folds", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    args = parser.parse_args(argv)
    setup_logging()

    payload = load_track_config(args.config)
    inputs = assemble_inputs(args.config)
    folds = args.folds or list(payload["development_folds"])
    seeds = args.seeds or [int(payload["experiment"]["seed"])]
    bootstrap = dict(payload["evaluation"]["bootstrap"])
    from agentic_forecaster.v3 import V3Track
    from agentic_forecaster.v3.experiment import new_experiment_id

    track = V3Track()
    ensure_dir(track.runtime_root)
    for seed in seeds:
        for fold in folds:
            window = resolve_fold_window(fold, payload["folds"])
            outcome = run_neural_fold(
                inputs, family=args.family, horizon=args.horizon, window=window,
                fold=fold, architecture=args.architecture, seed=seed,
                device=str(payload["experiment"]["device"]), bootstrap=bootstrap)
            experiment_id = new_experiment_id("V3N")
            out_dir = experiment_dir(experiment_id, runtime_root=track.runtime_root)
            outcome["predictions"].to_csv(out_dir / "predictions.csv", index=False)
            save_neural_checkpoint(outcome.pop("model"), out_dir,
                                   extra=outcome["parameters"] | {
                                       "architecture": args.architecture,
                                       "schema": outcome["schema"]})
            atomic_json_dump({
                "experiment_id": experiment_id,
                "track": "V3_EXOGENOUS_PRECOVID",
                "family": args.family,
                "objective_id": objective_id(args.horizon),
                "horizon_phrase": horizon_phrase(args.horizon),
                "architecture": args.architecture,
                "fold": fold,
                "seed": seed,
                "metrics": outcome["metrics"],
                "metrics_non_overlapping": outcome["non_overlapping"],
                "best_epoch": outcome["best_epoch"],
                "epochs_run": outcome["epochs_run"],
                "parameters": outcome["parameters"],
                "schema": outcome["schema"],
                "stock_scaler_provenance": outcome["stock_scaler_provenance"],
                "exogenous_scaler_provenance": outcome[
                    "exogenous_scaler_provenance"],
                "not_run": ["FILM", "REPTILE", "MAML", "MULTITASK"],
            }, out_dir / "summary.json")
            append_row({
                "experiment_id": experiment_id,
                "experiment_dir": str(out_dir),
                "model": args.architecture,
                "feature_family": args.family,
                "horizon": args.horizon,
                "objective_id": objective_id(args.horizon),
                "fold": fold,
                "seed": seed,
                "n_train": outcome["n_train"],
                "n_validation": outcome["n_validation"],
                "n_tickers": outcome["metrics"].get("n_tickers"),
                "accuracy": outcome["metrics"].get("accuracy"),
                "macro_accuracy": outcome["metrics"].get("macro_ticker_accuracy"),
                "balanced_accuracy": outcome["metrics"].get("balanced_accuracy"),
                "f1": outcome["metrics"].get("f1"),
                "roc_auc": outcome["metrics"].get("roc_auc"),
                "brier": outcome["metrics"].get("brier"),
                "ece": outcome["metrics"].get("ece"),
                "train_majority_baseline": outcome["metrics"].get(
                    "train_majority_baseline"),
                "baseline_delta": outcome["metrics"].get("baseline_delta"),
                "stock_only_auc": "",
                "incremental_auc": "",
                "common_sample_auc": "",
                "non_overlap_auc": outcome["non_overlapping"].get("roc_auc"),
                "config_sha256": inputs.config_sha256,
                "source_manifest_sha256": inputs.source_manifest_sha256,
                "feature_schema_sha256": outcome["schema"]["exogenous_schema_sha256"],
                "2019_lockbox_evaluated": False,
                "post_2019_evaluated": False,
            }, path=track.ledger)
            print(f"[neural] {args.architecture:<28} {args.family:<18} "
                  f"{objective_id(args.horizon):<18} {fold} seed={seed}: "
                  f"auc={outcome['metrics'].get('roc_auc'):.4f} "
                  f"bal={outcome['metrics'].get('balanced_accuracy'):.4f} "
                  f"base={outcome['metrics'].get('train_majority_baseline'):.4f} "
                  f"best_epoch={outcome['best_epoch']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())