"""V2 experiment orchestration: config -> data -> model -> metrics -> artefacts.

One shared model per (variant, fold, seed).  Every experiment directory contains

    resolved_config.yaml     manifest.json        training_history.json
    predictions.csv          ticker_metrics.csv   aggregate_metrics.json
    coverage_curve.csv       attention_summary.json
    model_complexity.json
    multitask_metrics.json   (V2-D onward)
    meta_episode_summary.json (V2-F only)

plus a ``checkpoint/`` directory (outside Git) holding the complete V2
checkpoint with the hashes that prove which data produced it.

Variant switches (V2-A .. V2-F) are declared ONCE, in :data:`VARIANT_FLAGS`, so
an ablation cannot drift: adding a component to one variant silently changes
only that variant.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from agentic_forecaster.config import load_config
from agentic_forecaster.utils import atomic_json_dump, ensure_dir, setup_logging

from . import context as ctx
from . import features as feat
from .checkpoint import save_checkpoint
from .dataset import (
    FeatureArrays,
    SampleTable,
    SplitWindow,
    V2ScalerBundle,
    V2SequenceDataset,
    assert_split_targets_legal,
    build_feature_arrays,
    build_sample_table,
    fit_scalers_on_train,
    resolve_fold_window,
    sample_metadata,
    split_masks,
)
from .firewall import (
    PRECOVID_LOCKBOX_ENV,
    PRECOVID_LOCKBOX_YEAR,
    V2_LOCKBOX_YEAR,
    assert_no_lockbox_targets,
    assert_no_paper_test_targets,
    assert_pre_covid_dates,
    describe_firewall,
    describe_precovid_firewall,
    lockbox_unlocked,
    pre_covid_lockbox_unlocked,
)
from .ledger import (
    V2_VARIANT_LABELS,
    append_experiment,
    experiment_dir,
    hash_payload,
    new_experiment_id,
)
from .ledger import (
    runtime_v2_root as v2_runtime_root,
)
from .losses import MultiTaskWeights
from .metrics import (
    COVERAGE_TARGETS,
    coverage_curve,
    direction_metrics,
    max_coverage_at_accuracy,
    multi_head_agreement,
    precision_at_k,
    rank_metrics,
    return_metrics,
    train_majority_baseline,
)
from .model import ContextualLSTMTransformer, V2ModelConfig
from .sectors import load_sector_map
from .store import (
    assert_store_within_pre_covid,
    load_store,
    store_fingerprints,
    v2_processed_root,
)
from .trainer import V2TrainConfig, V2Trainer, resolve_device

REPO_ROOT_DIR = Path(__file__).resolve().parents[3]
REPO_RESULTS_ROOT = REPO_ROOT_DIR / "results" / "v2"

logger = logging.getLogger("agentic_forecaster.v2.experiment")

#: The one place where the ablation is declared.  Each variant is exactly the
#: previous one plus the components named in the V2 programme.
VARIANT_FLAGS: dict[str, dict[str, bool]] = {
    "V2-A": {
        "use_transformer": False, "use_context": False, "use_sector_embedding": False,
        "use_regime": False, "use_multitask": False, "use_film": False,
        "use_adapter": False,
    },
    "V2-B": {
        "use_transformer": True, "use_context": False, "use_sector_embedding": False,
        "use_regime": False, "use_multitask": False, "use_film": False,
        "use_adapter": False,
    },
    "V2-C": {
        "use_transformer": True, "use_context": True, "use_sector_embedding": True,
        "use_regime": True, "use_multitask": False, "use_film": False,
        "use_adapter": False,
    },
    "V2-D": {
        "use_transformer": True, "use_context": True, "use_sector_embedding": True,
        "use_regime": True, "use_multitask": True, "use_film": False,
        "use_adapter": False,
    },
    "V2-E": {
        "use_transformer": True, "use_context": True, "use_sector_embedding": True,
        "use_regime": True, "use_multitask": True, "use_film": True,
        "use_adapter": False,
    },
    "V2-F": {
        "use_transformer": True, "use_context": True, "use_sector_embedding": True,
        "use_regime": True, "use_multitask": True, "use_film": True,
        "use_adapter": True,
    },
}

V2_VARIANT_DESCRIPTIONS: dict[str, str] = {
    "V2-A": "shared LSTM + ticker embedding + direction head",
    "V2-B": "V2-A + causal Transformer + attention pooling + LSTM residual",
    "V2-C": "V2-B + market/sector/relative/rank context + sector + regime encoder",
    "V2-D": "V2-C + normalised-return and cross-sectional-rank heads",
    "V2-E": "V2-D + FiLM ticker/sector/regime conditioning",
    "V2-F": "V2-E + residual adapter + Reptile-style head/adapter meta-learning",
}


@dataclass
class V2RunConfig:
    """A resolved V2 run configuration."""

    path: Path
    payload: dict
    variant: str
    fold: str
    seed: int
    device: str

    @property
    def sha256(self) -> str:
        return str(self.payload["config_sha256"])

    @property
    def dataset_root(self) -> Path:
        return Path(self.payload["data"]["dataset_root"])

    @property
    def data_variant(self) -> str:
        return str(self.payload["data"]["dataset_variant"])

    @property
    def universe_config(self) -> Path:
        return Path(self.payload["data"]["universe_config"])

    @property
    def max_date(self) -> str:
        return str(self.payload["data"]["max_date"])

    @property
    def supervised_tickers(self) -> list[str]:
        """The SUPERVISED population.

        A frozen ``data.supervised_universe`` declaration wins (the PRE-COVID
        eligible list is derived and frozen before training); an explicit
        ``data.supervised_tickers`` list is the ordinary V2 behaviour.
        """
        declared = self.payload["data"].get("supervised_universe")
        if declared:
            import yaml

            path = Path(declared)
            if not path.is_absolute() and not path.exists():
                path = Path(__file__).resolve().parents[3] / path
            if not path.is_file():
                raise FileNotFoundError(
                    f"frozen supervised universe not found: {path}. Run "
                    "scripts/build_v2_precovid_universe.py first; the list is derived, "
                    "never guessed."
                )
            frozen = yaml.safe_load(path.read_text()) or {}
            tickers = [str(x).upper() for x in frozen.get("eligible_tickers", [])]
            if not tickers:
                raise ValueError(f"{path}: eligible_tickers is empty")
            return tickers
        return [str(t).upper() for t in self.payload["data"]["supervised_tickers"]]

    @property
    def sequence_length(self) -> int:
        return int(self.payload["data"]["sequence_length"])

    @property
    def window(self) -> SplitWindow:
        """The resolved split window: config-declared folds win over defaults."""
        return resolve_fold_window(self.fold, self.payload.get("folds"))

    @property
    def pre_covid_mode(self) -> bool:
        return bool(self.payload.get("pre_covid_mode", False))

    @property
    def experiment_regime(self) -> str:
        return str(self.payload.get("experiment_regime", "ORDINARY_V2"))

    @property
    def final_allowed_date(self) -> str | None:
        """PRE-COVID boundary; ``None`` for the ordinary V2 programme."""
        if not self.pre_covid_mode:
            return None
        value = self.payload.get("final_allowed_date") or self.payload[
            "data"].get("final_allowed_date")
        return str(value) if value else None

    @property
    def processed_root(self) -> Path:
        """Processed branch that holds the feature store for this track."""
        declared = self.payload["data"].get("store_root")
        return Path(declared) if declared else v2_processed_root()

    @property
    def results_root(self) -> Path:
        """Track results directory, resolved against the repository root.

        A relative declaration must not depend on the caller's working
        directory: the ledger and the freeze file have to land in the same place
        however the script was invoked.
        """
        declared = self.payload.get("results_root")
        if not declared:
            return REPO_RESULTS_ROOT
        path = Path(declared)
        return path if path.is_absolute() else (REPO_ROOT_DIR / path)

    @property
    def runtime_root(self) -> Path:
        declared = self.payload.get("runtime_root")
        if declared:
            return Path(declared)
        return v2_runtime_root()

    @property
    def is_lockbox_fold(self) -> bool:
        return "LOCKBOX" in self.fold.upper()

    @property
    def flags(self) -> dict[str, bool]:
        """Component switches: the variant ladder, then any config override.

        The override exists so a multi-task continuation can inherit the ACTUAL
        winning base (for example PRE-V2-D on top of PRE-V2-B without silently
        re-enabling the context block that PRE-V2-C needed).  The historical
        V2 variant definitions are never mutated.
        """
        flags = dict(VARIANT_FLAGS[self.variant])
        for key, value in (self.payload.get("components") or {}).items():
            if key not in flags:
                raise ValueError(
                    f"unknown component flag {key!r}; known flags are {sorted(flags)}")
            flags[key] = bool(value)
        return flags

    @property
    def sector_map_path(self) -> Path:
        return Path(self.payload["data"]["sector_map_csv"])

    @property
    def ece_bins(self) -> int:
        return int(self.payload.get("evaluation", {}).get("ece_bins", 10))

    @property
    def train_config(self) -> V2TrainConfig:
        raw = dict(self.payload.get("training", {}))
        raw["seed"] = self.seed
        return V2TrainConfig(**raw)

    @property
    def loss_weights(self) -> MultiTaskWeights:
        """The fixed multi-task weights declared in the config."""
        raw = dict(self.payload.get("loss", {}))
        return MultiTaskWeights(direction=float(raw.get("direction", 1.0)),
                                return_value=float(raw.get("return", 0.5)),
                                rank=float(raw.get("rank", 0.25)))

    @property
    def return_target_clip(self) -> float:
        return float(self.payload.get("loss", {}).get("return_target_clip", 5.0))


def merge_configs(base: dict, override: dict) -> dict:
    """Deep-merge ``override`` onto ``base`` (variant configs stay short)."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_run_config(path: str | Path, *, variant: str | None = None,
                    fold: str | None = None, seed: int | None = None,
                    device: str | None = None) -> V2RunConfig:
    """Load a V2 config, apply CLI overrides and validate the firewall rules."""
    path = Path(path)
    payload = load_config(path)
    base_ref = payload.pop("base", None)
    if base_ref:
        base_path = Path(base_ref)
        if not base_path.is_absolute() and not base_path.exists():
            base_path = path.parent / base_path
        payload = merge_configs(load_config(base_path), payload)
        payload["base_config"] = str(base_path)
    resolved_variant = str(variant or payload["variant"]).upper()
    if resolved_variant not in VARIANT_FLAGS:
        raise ValueError(f"unknown V2 variant {resolved_variant!r}")
    payload["variant"] = resolved_variant
    payload["variant_label"] = V2_VARIANT_LABELS[resolved_variant]
    payload["variant_description"] = V2_VARIANT_DESCRIPTIONS[resolved_variant]
    payload["variant_flags"] = VARIANT_FLAGS[resolved_variant]

    resolved_fold = str(fold or payload.get("fold", "V2_DEV_FOLD_A")).upper()
    # resolve_fold_window raises for a name that is neither config-declared nor a
    # built-in default, so an unknown fold is rejected here.
    payload["fold"] = resolved_fold
    payload["fold_window"] = dataclasses.asdict(
        resolve_fold_window(resolved_fold, payload.get("folds")))

    payload["seed"] = int(seed if seed is not None else payload["experiment"]["seed"])
    payload["device"] = str(device or payload["experiment"].get("device", "auto"))

    pre_covid = bool(payload.get("pre_covid_mode", False))
    window = resolve_fold_window(resolved_fold, payload.get("folds"))
    if "LOCKBOX" in resolved_fold.upper():
        if pre_covid:
            if not pre_covid_lockbox_unlocked():
                raise AssertionError(
                    f"the PRE-COVID lockbox ({PRECOVID_LOCKBOX_YEAR} evaluation) "
                    f"requires {PRECOVID_LOCKBOX_ENV}=1, set only by "
                    "scripts/run_v2_precovid_lockbox.py. The PRE-COVID development "
                    "path must never score 2019."
                )
        elif not lockbox_unlocked():
            raise AssertionError(
                "V2_LOCKBOX requires the explicit lockbox switch (V2_LOCKBOX=1, used "
                "only by scripts/run_v2_lockbox.py). The development script must never "
                f"score {V2_LOCKBOX_YEAR}."
            )
    assert_no_paper_test_targets([window.val_end], where="v2 config")
    if pre_covid:
        assert_pre_covid_dates(feature_dates=[window.val_end],
                               origin_dates=[window.val_end],
                               target_dates=[window.val_end],
                               where="v2 precovid config")
    else:
        assert_no_lockbox_targets([window.val_end], where="v2 config",
                                   unlocked=lockbox_unlocked())

    payload["config_path"] = str(path)
    payload["resolved_components"] = {
        key: bool(value)
        for key, value in ((payload.get("components") or {})
                           | {k: v for k, v in VARIANT_FLAGS[resolved_variant].items()
                              if k not in (payload.get("components") or {})}).items()
    }
    payload["config_sha256"] = hash_payload(payload)
    return V2RunConfig(path=path, payload=payload, variant=resolved_variant,
                       fold=resolved_fold, seed=payload["seed"], device=payload["device"])


# ---------------------------------------------------------------------------
# data assembly
# ---------------------------------------------------------------------------

@dataclass
class V2RunData:
    """Everything one experiment needs from the cached store."""

    store: object
    sector_map: object
    samples: SampleTable
    arrays: FeatureArrays
    masks: dict[str, np.ndarray]
    scalers: V2ScalerBundle
    store_meta: dict = field(default_factory=dict)
    access_audit: dict = field(default_factory=dict)

    def split_frame(self, split: str) -> pd.DataFrame:
        """Sample rows of one split, with the store's realised outcome attached."""
        frame = self.samples.frame.loc[self.masks[split]].reset_index(drop=True)
        targets = self.store.targets.loc[
            :, ["ticker", "origin_date", "realized_vol_20", "raw_next_return"]]
        frame = frame.merge(targets, on=["ticker", "origin_date"], how="left",
                            validate="one_to_one")
        frame["position"] = np.arange(len(frame))
        return frame


def assemble_context_frame(store) -> pd.DataFrame:
    """Merge the cached per-(security, date) context tables into one frame.

    The store keeps the date-level market proxy, the per-security sector context,
    the cross-sectional ranks and the regime vector in separate files for
    auditability; the model needs them side by side.
    """
    context = store.sector.merge(store.cross_sectional, on=["ticker", "date"],
                                 how="left", validate="one_to_one")
    context = context.merge(store.market, on="date", how="left",
                            validate="many_to_one")
    missing = [c for c in (*ctx.CONTEXT_FEATURES, *ctx.RANK_FEATURES, *ctx.REGIME_FEATURES)
               if c not in context.columns]
    if missing:
        raise KeyError(
            f"the cached store is missing context column(s) {missing[:5]}; rebuild it "
            "with scripts/build_v2_context.py"
        )
    return context


def assemble_run_data(config: V2RunConfig) -> V2RunData:
    """Load the store, build samples, split them and fit the global scalers."""
    final_allowed = config.final_allowed_date
    store = load_store(config.processed_root, final_allowed_date=final_allowed)
    if final_allowed is not None:
        assert_store_within_pre_covid(store.metadata, final_allowed_date=final_allowed)
    sector_map = load_sector_map(config.sector_map_path)
    flags = config.flags

    arrays = build_feature_arrays(
        store.stock, assemble_context_frame(store), sector_map,
        tickers=config.supervised_tickers, use_context=bool(flags["use_context"]))

    require = ["y_direction"]
    if flags["use_multitask"]:
        require += ["y_return", "y_rank"]
    samples = build_sample_table(arrays, store.targets,
                                sequence_length=config.sequence_length,
                                require_targets=require, max_date=config.max_date,
                                final_allowed_date=final_allowed)

    masks = split_masks(samples, config.window)
    assert_split_targets_legal(masks, samples, where=f"{config.variant}/{config.fold}",
                               unlocked=lockbox_unlocked(),
                               final_allowed_date=final_allowed)
    for name in ("train", "val"):
        if not masks[name].any():
            raise ValueError(
                f"{config.variant}/{config.fold}: no {name} samples under the "
                "origin AND target split rule"
            )

    scalers = fit_scalers_on_train(samples, masks["train"],
                                    sequence_length=config.sequence_length)
    access_audit = {
        "store_root": str(store.root),
        "store_sha256": store.store_sha256,
        "store_last_feature_date": str(pd.Timestamp(store.stock["date"].max()).date()),
        "store_last_target_date": str(
            pd.Timestamp(store.targets["target_date"].max()).date()),
        "max_feature_date_consumed": _max_date(
            samples.frame.loc[masks["train"] | masks["val"], "origin_date"], "origin"),
        "max_origin_date_consumed": _max_date(
            samples.frame.loc[masks["train"] | masks["val"], "origin_date"], "origin"),
        "max_target_date_consumed": _max_date(
            samples.frame.loc[masks["train"] | masks["val"], "target_date"], "target"),
        "n_train_samples": int(masks["train"].sum()),
        "n_val_samples": int(masks["val"].sum()),
        "post_regime_rows_consumed": 0 if final_allowed is not None else None,
        "regime": config.experiment_regime,
        "final_allowed_date": final_allowed,
    }
    if final_allowed is not None:
        assert_pre_covid_dates(
            feature_dates=[access_audit["max_feature_date_consumed"]],
            origin_dates=[access_audit["max_origin_date_consumed"]],
            target_dates=[access_audit["max_target_date_consumed"]],
            final_allowed_date=final_allowed,
            where=f"{config.variant}/{config.fold} consumed data")
    return V2RunData(store=store, sector_map=sector_map, samples=samples, arrays=arrays,
                     masks=masks, scalers=scalers,
                     store_meta=store_fingerprints(config.processed_root),
                     access_audit=access_audit)


def _max_date(values: pd.Series, label: str) -> str | None:
    if len(values) == 0:
        return None
    return str(pd.Timestamp(values.max()).date())


def build_model_config(config: V2RunConfig, data: V2RunData) -> V2ModelConfig:
    """Translate the YAML model block plus the variant flags into a model config."""
    raw = dict(config.payload.get("model", {}))
    flags = config.flags
    arrays = data.arrays
    return V2ModelConfig(
        n_stock_features=arrays.n_stock_features,
        n_context_features=(arrays.n_context_features if flags["use_context"] else 0),
        n_regime_features=(arrays.n_regime_features if flags["use_regime"] else 0),
        n_tickers=max(len(arrays.ticker_vocab), 1),
        n_sectors=max(len(arrays.sector_vocab), 1),
        ticker_vocab=list(arrays.ticker_vocab),
        sector_vocab=list(arrays.sector_vocab),
        hidden_size=int(raw.get("hidden_size", 96)),
        lstm_layers=int(raw.get("lstm_layers", 2)),
        lstm_dropout=float(raw.get("lstm_dropout", 0.20)),
        context_dim=int(raw.get("context_dim", 32)),
        fusion_temporal_dim=int(raw.get("d_model", 64)),
        max_sequence_length=int(raw.get("max_sequence_length", 128)),
        d_model=int(raw.get("d_model", 64)),
        transformer_layers=int(raw.get("transformer_layers", 2)),
        n_heads=int(raw.get("n_heads", 4)),
        dim_feedforward=int(raw.get("dim_feedforward", 128)),
        transformer_dropout=float(raw.get("transformer_dropout", 0.10)),
        ticker_embedding_dim=int(raw.get("ticker_embedding_dim", 16)),
        sector_embedding_dim=int(raw.get("sector_embedding_dim", 8)),
        regime_embedding_dim=int(raw.get("regime_embedding_dim", 16)),
        fusion_hidden=int(raw.get("fusion_hidden", 128)),
        z_dim=int(raw.get("z_dim", 64)),
        fusion_dropout=float(raw.get("fusion_dropout", 0.10)),
        head_hidden=int(raw.get("head_hidden", 32)),
        **flags,
    )


def feature_schema(config: V2RunConfig, arrays: FeatureArrays) -> dict:
    """The schema recorded in the checkpoint and hashed into the manifest."""
    flags = config.flags
    return {
        "sequence_length": config.sequence_length,
        "stock_features": list(feat.STOCK_FEATURE_NAMES),
        "context_features": list(arrays.context_features),
        "context_features_used": bool(flags["use_context"]),
        "percentile_features": list(arrays.percentile_features),
        "percentile_features_scaled": False,
        "regime_features": list(arrays.regime_features),
        "targets": ["y_direction"] + (["y_return", "y_rank"] if flags["use_multitask"]
                                      else []),
        "scaler_policy": "global per feature group, fitted on TRAIN samples only",
        "prediction_time": "after market close on day t; predict day t+1",
        "stock_feature_count": arrays.n_stock_features,
        "context_feature_count": arrays.n_context_features,
        "regime_feature_count": arrays.n_regime_features,
        "market_context_columns": list(ctx.MARKET_FEATURES),
        "market_leave_one_out_columns": list(ctx.MARKET_LOO_FEATURES),
        "sector_context_columns": list(ctx.SECTOR_FEATURES),
        "relative_columns": list(ctx.RELATIVE_FEATURES),
        "targets_note": ("y_rank is a LABEL built from t+1 returns; it is never an input"),
    }


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model: ContextualLSTMTransformer, dataset: V2SequenceDataset, *,
            device: torch.device, batch_size: int = 512,
            collect_attention: bool = True,
            positions: np.ndarray | None = None) -> tuple[pd.DataFrame, dict]:
    """Predict a whole split; return the prediction frame and attention summary.

    ``positions`` maps each dataset item onto the row of the split frame it came
    from, so a per-security prediction pass can be merged back into one frame.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    pos = np.arange(len(dataset)) if positions is None else np.asarray(positions)
    rows: list[dict] = []
    attention_sum: np.ndarray | None = None
    attention_n = 0

    for batch in loader:
        outputs = model(batch["stock_sequence"].to(device), batch["ticker_id"].to(device),
                        context_sequence=batch["context_sequence"].to(device),
                        sector_id=batch["sector_id"].to(device),
                        regime_vector=batch["regime_vector"].to(device),
                        return_attention=collect_attention)
        index = batch["index"].numpy()
        frame = pd.DataFrame({
            "position": pos[index],
            "direction_logit": outputs["direction_logit"].detach().cpu().numpy(),
        })
        frame["p_up"] = 1.0 / (1.0 + np.exp(-frame["direction_logit"].to_numpy(float)))
        frame["confidence"] = np.abs(frame["p_up"].to_numpy(float) - 0.5)
        for key in ("return_prediction", "rank_prediction"):
            if key in outputs:
                frame[key] = outputs[key].detach().cpu().numpy()
        rows.append(frame)
        if collect_attention and "temporal_attention" in outputs:
            attention = outputs["temporal_attention"].detach().cpu().numpy()
            total = attention.sum(axis=0)
            attention_sum = total if attention_sum is None else attention_sum + total
            attention_n += attention.shape[0]

    predictions = pd.concat(rows, ignore_index=True).sort_values("position")
    summary: dict = {}
    if attention_sum is not None and attention_n:
        mean_attention = attention_sum / attention_n
        edge = max(1, len(mean_attention) // 10)
        summary = {
            "n_observations": int(attention_n),
            "sequence_length": int(mean_attention.shape[0]),
            "mean_attention_by_position": [float(v) for v in mean_attention],
            "attention_mass_last_10pct": float(mean_attention[-edge:].sum()),
            "attention_mass_origin": float(mean_attention[-1]),
            "attention_mass_first_10pct": float(mean_attention[:edge].sum()),
            "sums_to_one": bool(abs(float(mean_attention.sum()) - 1.0) < 1e-4),
            "note": ("attention over the causal window; the origin timestep is the "
                     "last position and carries the most recent information"),
        }
    return predictions, summary


def assemble_prediction_frame(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Join predictions onto their sample rows."""
    merged = frame.merge(predictions, on="position", how="left", validate="one_to_one")
    if merged["p_up"].isna().any():
        raise AssertionError("some validation samples received no prediction")
    return merged.rename(columns={"y_direction": "y_true"})


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def compute_metrics(frame: pd.DataFrame, *, train_labels: np.ndarray,
                    ece_bins: int = 10) -> tuple[dict, pd.DataFrame]:
    """Every reported metric for one validation split."""
    direction = direction_metrics(frame, ece_bins=ece_bins)
    baseline = train_majority_baseline(train_labels)
    if direction.get("n"):
        direction["train_majority_baseline"] = baseline
        direction["delta_vs_train_majority_macro"] = (
            direction["accuracy_macro_ticker"] - baseline)
        direction["delta_vs_train_majority_micro"] = (
            direction["accuracy_micro"] - baseline)
    p3 = precision_at_k(frame, k=3)
    curve = coverage_curve(frame)
    metrics = {
        "direction": direction,
        "selection": {
            "precision_at_3_up": p3["precision_at_3_up"],
            "precision_at_3_down": p3["precision_at_3_down"],
            "p_at_k_dates": p3["n_dates"],
        },
        "selective_accuracy": {
            f"max_coverage_at_{int(target * 100)}pct": max_coverage_at_accuracy(frame, target)
            for target in COVERAGE_TARGETS
        },
        "n_predictions": len(frame),
        "firewall": describe_firewall(),
        "test_2022_2023_evaluated": False,
        "interpretation_note": ("selective accuracy is NEVER overall accuracy; every "
                                "coverage point carries its own sample count"),
    }
    return metrics, curve


def per_ticker_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker, group in frame.groupby("ticker", sort=True):
        y = group["y_true"].to_numpy(int)
        p = group["p_up"].to_numpy(float)
        accuracy = float(((p >= 0.5).astype(int) == y).mean())
        majority = float(max(y.mean(), 1.0 - y.mean()))
        rows.append({
            "ticker": ticker,
            "n": len(group),
            "accuracy": accuracy,
            "majority_baseline": majority,
            "delta_vs_majority": accuracy - majority,
            "positive_rate": float(y.mean()),
            "mean_p_up": float(p.mean()),
            "mean_confidence": float(np.abs(p - 0.5).mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# artefacts
# ---------------------------------------------------------------------------

def write_outputs(out_dir: Path, config: V2RunConfig, data: V2RunData, *,
                  metrics: dict, curve: pd.DataFrame, predictions: pd.DataFrame,
                  training_history: dict, attention_summary: dict, complexity: dict,
                  multitask: dict | None = None, meta_summary: dict | None = None,
                  p3_rows: list[dict] | None = None) -> dict:
    """Write every per-experiment artefact and return the manifest."""
    ensure_dir(out_dir)
    (out_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config.payload, sort_keys=False), encoding="utf-8")
    atomic_json_dump(training_history, out_dir / "training_history.json")
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    per_ticker_frame(predictions).to_csv(out_dir / "ticker_metrics.csv", index=False)
    curve.to_csv(out_dir / "coverage_curve.csv", index=False)
    if p3_rows:
        pd.DataFrame(p3_rows).to_csv(out_dir / "p3_daily_selections.csv", index=False)
    atomic_json_dump(attention_summary or {}, out_dir / "attention_summary.json")
    atomic_json_dump(complexity, out_dir / "model_complexity.json")
    atomic_json_dump(metrics, out_dir / "aggregate_metrics.json")
    if multitask is not None:
        atomic_json_dump(multitask, out_dir / "multitask_metrics.json")
    if meta_summary is not None:
        atomic_json_dump(meta_summary, out_dir / "meta_episode_summary.json")

    manifest = {
        "experiment_id": config.payload["experiment_id"],
        "experiment_regime": config.experiment_regime,
        "pre_covid_mode": config.pre_covid_mode,
        "final_allowed_date": config.final_allowed_date,
        "component_flags": config.flags,
        "results_root": str(config.results_root),
        "runtime_root": str(config.runtime_root),
        "variant": config.variant,
        "variant_label": V2_VARIANT_LABELS[config.variant],
        "variant_description": V2_VARIANT_DESCRIPTIONS[config.variant],
        "is_original_paper_model": False,
        "classification": "NEW_EXPERIMENTAL_ARCHITECTURE",
        "fold": config.fold,
        "fold_window": dataclasses.asdict(config.window),
        "seed": config.seed,
        "device": str(complexity.get("device", config.device)),
        "config_sha256": config.sha256,
        "feature_store": data.store_meta,
        "sector_map_sha256": data.store.sector_map_sha256,
        "source_dataset_root": str(data.store.metadata.get("source_dataset_root")),
        "source_manifest_sha256": data.store.metadata.get("source_manifest_sha256"),
        "universe_id": data.store.universe_id,
        "data_variant": data.store.data_variant,
        "sequence_length": config.sequence_length,
        "samples": {
            "train": sample_metadata(data.samples.frame.loc[data.masks["train"]]),
            "val": sample_metadata(data.samples.frame.loc[data.masks["val"]]),
            "build": data.samples.diagnostics,
        },
        "scaler_fit": data.scalers.state(),
        "firewall": describe_firewall(),
        "precovid_firewall": (describe_precovid_firewall()
                              if config.pre_covid_mode else None),
        "data_access_audit": data.access_audit,
        "test_2022_2023_evaluated": False,
    }
    atomic_json_dump(manifest, out_dir / "manifest.json")
    return manifest


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def run_experiment(config: V2RunConfig, *, out_dir: Path | None = None,
                   experiment_id: str | None = None) -> dict:
    """Execute one V2 experiment end to end and return its summary."""
    setup_logging()
    experiment_id = experiment_id or new_experiment_id(config.variant.replace("-", ""))
    config.payload["experiment_id"] = experiment_id
    out_dir = (Path(out_dir) if out_dir is not None
               else experiment_dir(experiment_id, runtime_root=config.runtime_root))
    ensure_dir(out_dir)

    data = assemble_run_data(config)
    model_config = build_model_config(config, data)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model = ContextualLSTMTransformer(model_config)
    device = resolve_device(config.device)

    train_frame = data.split_frame("train")
    val_frame = data.split_frame("val")
    train_dataset = V2SequenceDataset(data.samples, train_frame.drop(columns=["position"]),
                                      scalers=data.scalers,
                                      sequence_length=config.sequence_length)
    val_dataset = V2SequenceDataset(data.samples, val_frame.drop(columns=["position"]),
                                    scalers=data.scalers,
                                    sequence_length=config.sequence_length)

    trainer = V2Trainer(model, config.train_config, loss_weights=config.loss_weights,
                        device=config.device)
    result = trainer.fit(train_dataset, val_dataset, train_frame=train_frame,
                         val_frame=val_frame)

    predictions, attention_summary = predict(
        model, val_dataset, device=device,
        collect_attention=bool(config.flags["use_transformer"]),
        positions=val_frame["position"].to_numpy())

    meta_summary = None
    meta_init = None
    if config.variant == "V2-F":
        predictions, meta_summary, meta_init = _run_meta_stage(
            config, data, model, device, predictions, val_frame)

    predictions = assemble_prediction_frame(val_frame, predictions)
    predictions["split"] = "val"
    if "return_prediction" in predictions.columns:
        predictions["predicted_next_return"] = (
            predictions["return_prediction"] * predictions["realized_vol_20"])

    train_labels = data.samples.frame.loc[data.masks["train"], "y_direction"].to_numpy(int)
    metrics, curve = compute_metrics(predictions, train_labels=train_labels,
                                     ece_bins=config.ece_bins)
    complexity = {
        **model.n_parameters(),
        "peak_gpu_memory_mb": result.peak_gpu_memory_mb,
        "training_seconds": result.train_seconds,
        "best_epoch": result.best_epoch,
        "epochs_run": result.epochs_run,
        "stopped_early": result.stopped_early,
        "sequence_length": config.sequence_length,
        "n_train_samples": len(train_frame),
        "n_val_samples": len(val_frame),
        "batch_size": config.train_config.batch_size,
        "max_epochs": config.train_config.max_epochs,
        "device": str(device),
    }

    multitask = None
    if config.flags["use_multitask"]:
        multitask = {
            "return": return_metrics(predictions, y_col="y_return",
                                     pred_col="return_prediction",
                                     raw_y_col="raw_next_return",
                                     raw_pred_col="predicted_next_return"),
            "rank": rank_metrics(predictions, pred_col="rank_prediction", y_col="y_rank"),
            "head_agreement": multi_head_agreement(predictions),
            "note": ("return and rank are auxiliary objectives; the headline metric "
                     "is direction accuracy"),
        }

    training_history = {
        "config": config.train_config.to_dict(),
        "best_epoch": result.best_epoch,
        "best_val_total_loss": result.best_val_total_loss,
        "epochs_run": result.epochs_run,
        "stopped_early": result.stopped_early,
        "train_seconds": result.train_seconds,
        "sampler": result.sampler,
        "history": result.history,
    }

    p3_rows = precision_at_k(predictions, k=3)["selections"]
    manifest = write_outputs(
        out_dir, config, data, metrics=metrics, curve=curve, predictions=predictions,
        training_history=training_history, attention_summary=attention_summary,
        complexity=complexity, multitask=multitask, meta_summary=meta_summary,
        p3_rows=p3_rows)

    save_checkpoint(
        out_dir / "checkpoint", model,
        scalers=data.scalers,
        feature_schema=feature_schema(config, data.arrays),
        sector_map_hash=data.store.sector_map_sha256,
        feature_store_hash=str(data.store_meta.get("store_sha256", "")),
        training_history=training_history,
        manifest=manifest,
        meta_config=(meta_summary or {}).get("config") if meta_summary else None,
        meta_initialization=meta_init,
        model_config=model_config,
    )

    summary = {
        "experiment_id": experiment_id,
        "out_dir": str(out_dir),
        "variant": config.variant,
        "fold": config.fold,
        "seed": config.seed,
        "metrics": metrics,
        "complexity": complexity,
        "manifest": manifest,
    }
    _append_ledger_row(config, data, summary, complexity)
    return summary


def _run_meta_stage(config: V2RunConfig, data: V2RunData, model, device,
                    base_predictions: pd.DataFrame, val_frame: pd.DataFrame):
    """Reptile meta-training, then per-security adaptation for validation.

    Adaptation for a security uses the last labelled TRAIN samples of that
    security, immediately preceding the validation window.  No validation label is
    ever read by the support loader.
    """
    from .meta import MetaConfig, ReptileMetaTrainer, validation_support_frames

    meta_config = MetaConfig(**dict(config.payload.get("meta", {})))
    trainer = ReptileMetaTrainer(model, meta_config, device=device,
                                 loss_weights=config.loss_weights,
                                 final_allowed_date=config.final_allowed_date)
    meta_result = trainer.fit(data.samples, data.masks["train"], seed=config.seed,
                              where=f"{config.variant}/{config.fold}/meta-train")

    adapted_frames: list[pd.DataFrame] = []
    audit: list[dict] = []
    for ticker in sorted(val_frame["ticker"].unique()):
        query_positions = val_frame.loc[val_frame["ticker"] == ticker, "position"]
        query = val_frame.loc[val_frame["ticker"] == ticker].reset_index(drop=True)
        support = validation_support_frames(
            data.samples, ticker=ticker, n_support=meta_config.validation_support_size,
            train_end=config.window.train_end, val_start=config.window.val_start)
        if support.empty or query.empty:
            continue
        audit.append(trainer.score_with_adaptation(
            data.samples, support_frame=support,
            query_frame=query.drop(columns=["position"]), ticker=ticker))
        adapted, _ = predict(
            trainer.model,
            V2SequenceDataset(data.samples, query.drop(columns=["position"]),
                              scalers=data.scalers,
                              sequence_length=config.sequence_length),
            device=device, collect_attention=False,
            positions=query_positions.to_numpy())
        adapted_frames.append(adapted)

    predictions = (pd.concat(adapted_frames, ignore_index=True) if adapted_frames
                   else base_predictions)
    meta_summary = {
        "label": meta_result["label"],
        "not_maml": True,
        "config": meta_result["config"],
        "n_episodes_available": meta_result["n_episodes_available"],
        "n_tickers_in_episodes": meta_result["n_tickers"],
        "history": meta_result["history"],
        "adaptation_audit": audit,
        "adaptation_delta_mean": float(np.mean([a["delta"] for a in audit]))
        if audit else None,
        "note": ("adaptation uses TRAIN support samples only; no validation label is "
                 "consumed by the support loader"),
    }
    return predictions, meta_summary, trainer.meta_init


def _ledger_root(config: V2RunConfig) -> Path:
    """Ledger directory for this track.

    The PRE-COVID ledger is never appended to the historical V2 ledger, and the
    historical V2 ledger is never appended to from a PRE-COVID run.
    """
    root = config.results_root
    root.mkdir(parents=True, exist_ok=True)
    return root


def _append_ledger_row(config: V2RunConfig, data: V2RunData, summary: dict,
                       complexity: dict) -> None:
    direction = summary["metrics"].get("direction", {})
    selection = summary["metrics"].get("selection", {})
    multitask_path = Path(summary["out_dir"]) / "multitask_metrics.json"
    multitask = json.loads(multitask_path.read_text()) if multitask_path.is_file() else {}
    return_block = multitask.get("return", {})
    rank_block = multitask.get("rank", {})
    append_experiment({
        "experiment_id": summary["experiment_id"],
        "experiment_dir": summary["out_dir"],
        "experiment_regime": config.experiment_regime,
        "variant": config.variant,
        "fold": config.fold,
        "seed": config.seed,
        "tickers": ",".join(config.supervised_tickers),
        "data_variant": data.store.data_variant,
        "universe_id": data.store.universe_id,
        "config_sha256": config.sha256,
        "feature_store_sha256": str(data.store_meta.get("store_sha256", "")),
        "sequence_length": config.sequence_length,
        "n_parameters": complexity["total_parameters"],
        "max_epochs": config.train_config.max_epochs,
        "best_epoch": complexity["best_epoch"],
        "validation_accuracy_micro": direction.get("accuracy_micro"),
        "validation_accuracy_macro": direction.get("accuracy_macro_ticker"),
        "validation_f1": direction.get("f1"),
        "validation_auc": direction.get("roc_auc"),
        "validation_brier": direction.get("brier"),
        "validation_ece": direction.get("ece"),
        "train_majority_baseline": direction.get("train_majority_baseline"),
        "precision_at_3_up": selection.get("precision_at_3_up"),
        "precision_at_3_down": selection.get("precision_at_3_down"),
        "return_mae": return_block.get("return_mae"),
        "return_spearman": return_block.get("return_spearman"),
        "rank_ic": rank_block.get("mean_rank_ic"),
        "test_2022_2023_evaluated": "false",
    }, results_root=_ledger_root(config))