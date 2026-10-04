"""The V2 checkpoint format.

V2 does NOT reuse the paper's ``FittedModel`` container.  That container holds
one binary logit, one scaler and one ticker's vocabulary; V2 has three heads, two
feature groups, two ID vocabularies, an optional FiLM module, an optional
adapter and, for the meta variant, a separate meta initialisation.  Loading a
V2 checkpoint must reproduce predictions exactly, so the format carries the
vocabularies, the feature schema and every hash needed to prove which data and
which sector map produced it.

Files written by :func:`save_checkpoint`::

    model.pt               state dict (+ meta initialisation for V2-F)
    model_config.json      V2ModelConfig, JSON-serialisable
    feature_schema.json    stock / context / rank / regime feature names
    stock_scaler.joblib    global train-only stock scaler
    context_scaler.joblib  global train-only context scaler
    ticker_vocab.json      ticker -> embedding index
    sector_vocab.json      sector -> embedding index (UNKNOWN always present)
    sector_map_hash.txt
    feature_store_hash.txt
    training_history.json
    manifest.json
    meta_config.json       V2-F only
    meta_initialization.pt V2-F only
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn

from agentic_forecaster.utils import atomic_json_dump, atomic_torch_save

from .dataset import GroupFeatureScaler, V2ScalerBundle
from .model import ContextualLSTMTransformer, V2ModelConfig
from .sectors import UNKNOWN_LABEL

logger = logging.getLogger("agentic_forecaster.v2.checkpoint")

CHECKPOINT_FILES: tuple[str, ...] = (
    "model.pt",
    "model_config.json",
    "feature_schema.json",
    "stock_scaler.joblib",
    "context_scaler.joblib",
    "ticker_vocab.json",
    "sector_vocab.json",
    "sector_map_hash.txt",
    "feature_store_hash.txt",
    "training_history.json",
    "manifest.json",
)

META_FILES: tuple[str, ...] = ("meta_config.json", "meta_initialization.pt")


class V2CheckpointError(RuntimeError):
    """Raised when a V2 checkpoint is missing or inconsistent."""


@dataclass
class V2Checkpoint:
    """A loaded V2 checkpoint."""

    root: Path
    model: ContextualLSTMTransformer
    config: V2ModelConfig
    stock_scaler: GroupFeatureScaler
    context_scaler: GroupFeatureScaler
    ticker_vocab: list[str]
    sector_vocab: list[str]
    feature_schema: dict
    manifest: dict
    training_history: dict
    meta_config: dict | None = None
    meta_initialization: dict[str, torch.Tensor] | None = None

    @property
    def scalers(self) -> V2ScalerBundle:
        return V2ScalerBundle(stock=self.stock_scaler, context=self.context_scaler)

    def to(self, device: torch.device | str) -> V2Checkpoint:
        self.model.to(device)
        return self

    def eval(self) -> V2Checkpoint:
        self.model.eval()
        return self


def _ordered_vocab(payload: dict) -> list[str]:
    """Rebuild an index-ordered vocabulary from its persisted form."""
    return [payload[str(i)] for i in range(len(payload))]


def _vocab_payload(vocab: list[str]) -> dict:
    return {
        "index_to_value": {str(i): v for i, v in enumerate(vocab)},
        "value_to_index": {v: i for i, v in enumerate(vocab)},
        "size": len(vocab),
        "unknown_slot": vocab.index(UNKNOWN_LABEL) if UNKNOWN_LABEL in vocab else 0,
    }


def save_checkpoint(root: str | Path, model: ContextualLSTMTransformer, *,
                    scalers: V2ScalerBundle, feature_schema: dict,
                    sector_map_hash: str, feature_store_hash: str,
                    training_history: dict, manifest: dict,
                    meta_config: dict | None = None,
                    meta_initialization: dict[str, torch.Tensor] | None = None,
                    model_config: V2ModelConfig | None = None) -> Path:
    """Persist a complete, reloadable V2 checkpoint."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config = model_config or model.config

    atomic_torch_save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                      root / "model.pt")
    atomic_json_dump(config.to_dict(), root / "model_config.json")
    atomic_json_dump(feature_schema, root / "feature_schema.json")
    joblib.dump(scalers.stock, root / "stock_scaler.joblib")
    joblib.dump(scalers.context, root / "context_scaler.joblib")
    atomic_json_dump(_vocab_payload(config.ticker_vocab), root / "ticker_vocab.json")
    atomic_json_dump(_vocab_payload(config.sector_vocab), root / "sector_vocab.json")
    (root / "sector_map_hash.txt").write_text(str(sector_map_hash) + "\n", encoding="utf-8")
    (root / "feature_store_hash.txt").write_text(str(feature_store_hash) + "\n",
                                                 encoding="utf-8")
    atomic_json_dump(training_history, root / "training_history.json")
    atomic_json_dump(manifest, root / "manifest.json")

    if meta_config is not None:
        atomic_json_dump(meta_config, root / "meta_config.json")
        initialisation = meta_initialization or {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if name in set(model.adaptable_parameter_names())
        }
        atomic_torch_save(initialisation, root / "meta_initialization.pt")
    return root


def load_checkpoint(root: str | Path, *, device: torch.device | str = "cpu",
                    load_meta: bool = True) -> V2Checkpoint:
    """Load a V2 checkpoint and rebuild the exact model it describes."""
    root = Path(root)
    missing = [name for name in CHECKPOINT_FILES if not (root / name).is_file()]
    if missing:
        raise V2CheckpointError(f"{root}: checkpoint is missing {missing}")

    config = V2ModelConfig.from_dict(
        json.loads((root / "model_config.json").read_text()))
    model = ContextualLSTMTransformer(config)
    state = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    stock_scaler = joblib.load(root / "stock_scaler.joblib")
    context_scaler = joblib.load(root / "context_scaler.joblib")
    ticker_vocab = _ordered_vocab(
        json.loads((root / "ticker_vocab.json").read_text())["index_to_value"])
    sector_vocab = _ordered_vocab(
        json.loads((root / "sector_vocab.json").read_text())["index_to_value"])

    meta_config = None
    meta_init = None
    if load_meta and (root / "meta_config.json").is_file():
        meta_config = json.loads((root / "meta_config.json").read_text())
        meta_init = torch.load(root / "meta_initialization.pt", map_location="cpu",
                               weights_only=True)

    checkpoint = V2Checkpoint(
        root=root,
        model=model,
        config=config,
        stock_scaler=stock_scaler,
        context_scaler=context_scaler,
        ticker_vocab=ticker_vocab,
        sector_vocab=sector_vocab,
        feature_schema=json.loads((root / "feature_schema.json").read_text()),
        manifest=json.loads((root / "manifest.json").read_text()),
        training_history=json.loads((root / "training_history.json").read_text()),
        meta_config=meta_config,
        meta_initialization=meta_init,
    )
    return checkpoint.to(device).eval()


def assert_predictions_match(model_a: nn.Module, model_b: nn.Module,
                             batch: dict[str, torch.Tensor], *,
                             atol: float = 1e-6) -> float:
    """Round-trip check: two models must produce identical logits.

    Returns the maximum absolute difference so a caller can record how exact the
    reload was rather than merely that it passed.
    """
    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        out_a = model_a(batch["stock_sequence"], batch["ticker_id"],
                        context_sequence=batch.get("context_sequence"),
                        sector_id=batch.get("sector_id"),
                        regime_vector=batch.get("regime_vector"))
        out_b = model_b(batch["stock_sequence"], batch["ticker_id"],
                        context_sequence=batch.get("context_sequence"),
                        sector_id=batch.get("sector_id"),
                        regime_vector=batch.get("regime_vector"))
    worst = 0.0
    for key, value in out_a.items():
        if not torch.is_tensor(value) or key not in out_b:
            continue
        worst = max(worst, float((value - out_b[key]).abs().max()))
    if worst > atol:
        raise V2CheckpointError(
            f"checkpoint round trip changed predictions by {worst:.3e} (atol {atol:.1e})")
    return worst


def vocabulary_frames(ticker_vocab: list[str], sector_vocab: list[str]
                      ) -> tuple[list[str], list[str]]:
    """Validate the vocabularies before they are written."""
    if not ticker_vocab:
        raise V2CheckpointError("ticker vocabulary is empty")
    if UNKNOWN_LABEL not in sector_vocab:
        raise V2CheckpointError(
            f"the sector vocabulary must contain {UNKNOWN_LABEL!r} so an unmapped "
            f"sector has a valid embedding index; got {sector_vocab}"
        )
    if len(set(ticker_vocab)) != len(ticker_vocab):
        raise V2CheckpointError("ticker vocabulary contains duplicates")
    return list(ticker_vocab), list(sector_vocab)


def checkpoint_hashes(root: str | Path) -> dict:
    """Hash the checkpoint's identity files for the ledger."""
    root = Path(root)
    out = {}
    for name in ("sector_map_hash.txt", "feature_store_hash.txt"):
        path = root / name
        out[name.replace(".txt", "")] = path.read_text().strip() if path.is_file() else None
    return out


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()