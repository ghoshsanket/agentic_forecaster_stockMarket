"""The V3 neural stage: a shared LSTM over the stock sequence FUSED WITH a small
encoder over the exogenous sequence.

WHY TWO ENCODERS (and not one wide input)
-----------------------------------------
The exogenous block is dozens of columns against 27 stock features.  Concatenating
them at every timestep would let the wider block dominate the input scale and the
model would effectively become an exogenous-only model.  Instead each stream gets
its own encoder over the SAME 60 origin dates, and the two representations are
fused:

    stock sequence      27 features -> Linear -> GELU -> LayerNorm
                      -> LSTM(96, 2 layers, dropout 0.20)      -> 96
    exogenous sequence  F_ex features over the same 60 dates
                      -> Linear(F_ex, 32) -> GELU -> LayerNorm
                      -> LSTM(32, 1 layer)                     -> 32
    ticker embedding    16
    fusion             96 + 32 + 16 = 144
                      -> Linear(144, 96) -> GELU -> Dropout(0.10)
                      -> Linear(96, 32) -> GELU -> Linear(32, 1)

The exogenous encoder is deliberately SMALL: the question is whether the external
INFORMATION carries signal, not whether a bigger encoder can extract more from it.

A Transformer variant is available only as a second ablation.  It must earn its
complexity: the PRE-COVID price-only study found the Transformer did not help.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

logger = logging.getLogger("agentic_forecaster.v3.neural")

#: Fixed architecture parameters (the unchanged V2-A temporal encoder).
STOCK_HIDDEN_SIZE = 96
STOCK_LSTM_LAYERS = 2
STOCK_LSTM_DROPOUT = 0.20
TICKER_EMBEDDING_DIM = 16
EXOGENOUS_HIDDEN_SIZE = 32
EXOGENOUS_LSTM_LAYERS = 1
FUSION_HIDDEN = 96
FUSION_Z = 32
FUSION_DROPOUT = 0.10


@dataclass(frozen=True)
class NeuralArchitecture:
    """Every architectural choice, recorded in the checkpoint."""

    n_stock_features: int
    n_exogenous_features: int
    n_tickers: int
    sequence_length: int = 60
    use_transformer: bool = False
    d_model: int = 64
    transformer_layers: int = 2
    n_heads: int = 4
    dim_feedforward: int = 128
    transformer_dropout: float = 0.10

    def to_dict(self) -> dict:
        return {
            "n_stock_features": self.n_stock_features,
            "n_exogenous_features": self.n_exogenous_features,
            "n_tickers": self.n_tickers,
            "sequence_length": self.sequence_length,
            "use_transformer": self.use_transformer,
            "stock_encoder": {"hidden": STOCK_HIDDEN_SIZE, "layers": STOCK_LSTM_LAYERS,
                              "dropout": STOCK_LSTM_DROPOUT, "unidirectional": True},
            "exogenous_encoder": {"hidden": EXOGENOUS_HIDDEN_SIZE,
                                  "layers": EXOGENOUS_LSTM_LAYERS},
            "ticker_embedding": TICKER_EMBEDDING_DIM,
            "fusion": {"input": STOCK_HIDDEN_SIZE + EXOGENOUS_HIDDEN_SIZE
                                     + TICKER_EMBEDDING_DIM,
                       "hidden": FUSION_HIDDEN, "z": FUSION_Z,
                       "dropout": FUSION_DROPOUT},
            "head": "binary direction, 1 logit",
        }


class DualSequenceDirectionModel(nn.Module):
    """Shared stock LSTM + exogenous encoder, fused into one direction logit."""

    def __init__(self, architecture: NeuralArchitecture) -> None:
        super().__init__()
        self.architecture = architecture
        self.architecture_record = architecture.to_dict

        self.stock_input = nn.Sequential(
            nn.Linear(architecture.n_stock_features, STOCK_HIDDEN_SIZE),
            nn.GELU(),
            nn.LayerNorm(STOCK_HIDDEN_SIZE),
        )
        self.stock_lstm = nn.LSTM(STOCK_HIDDEN_SIZE, STOCK_HIDDEN_SIZE,
                                  num_layers=STOCK_LSTM_LAYERS,
                                  dropout=STOCK_LSTM_DROPOUT,
                                  batch_first=True, bidirectional=False)

        self.exogenous_enabled = architecture.n_exogenous_features > 0
        if self.exogenous_enabled:
            self.exogenous_input = nn.Sequential(
                nn.Linear(architecture.n_exogenous_features, EXOGENOUS_HIDDEN_SIZE),
                nn.GELU(),
                nn.LayerNorm(EXOGENOUS_HIDDEN_SIZE),
            )
            self.exogenous_lstm = nn.LSTM(EXOGENOUS_HIDDEN_SIZE,
                                          EXOGENOUS_HIDDEN_SIZE,
                                          num_layers=EXOGENOUS_LSTM_LAYERS,
                                          batch_first=True, bidirectional=False)

        self.ticker_embedding = nn.Embedding(max(architecture.n_tickers, 1),
                                             TICKER_EMBEDDING_DIM)

        temporal_dim = STOCK_HIDDEN_SIZE
        if architecture.use_transformer:
            self.transformer_input = nn.Linear(STOCK_HIDDEN_SIZE, architecture.d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=architecture.d_model, nhead=architecture.n_heads,
                dim_feedforward=architecture.dim_feedforward,
                dropout=architecture.transformer_dropout, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer,
                                                     num_layers=architecture.transformer_layers)
            self.attention_pool = nn.Linear(architecture.d_model, 1)
            temporal_dim = architecture.d_model

        fusion_dim = temporal_dim + (EXOGENOUS_HIDDEN_SIZE if self.exogenous_enabled
                                     else 0) + TICKER_EMBEDDING_DIM
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, FUSION_HIDDEN),
            nn.GELU(),
            nn.Dropout(FUSION_DROPOUT),
            nn.Linear(FUSION_HIDDEN, FUSION_Z),
            nn.GELU(),
        )
        self.head = nn.Linear(FUSION_Z, 1)

    # ------------------------------------------------------------------ forward

    def _pool(self, outputs: torch.Tensor) -> torch.Tensor:
        """Last timestep of a unidirectional LSTM: the most recent state."""
        return outputs[:, -1, :]

    def stock_representation(self, stock_sequence: torch.Tensor) -> torch.Tensor:
        encoded = self.stock_input(stock_sequence)
        sequence, _ = self.stock_lstm(encoded)
        if self.architecture.use_transformer:
            projected = self.transformer_input(sequence)
            transformed = self.transformer(projected)
            weights = torch.softmax(self.attention_pool(transformed), dim=1)
            return (transformed * weights).sum(dim=1)
        return self._pool(sequence)

    def exogenous_representation(self, exogenous_sequence: torch.Tensor | None
                                 ) -> torch.Tensor | None:
        if not self.exogenous_enabled or exogenous_sequence is None:
            return None
        encoded = self.exogenous_input(exogenous_sequence)
        sequence, _ = self.exogenous_lstm(encoded)
        return self._pool(sequence)

    def fuse(self, stock: torch.Tensor, ticker_id: torch.Tensor,
             exogenous: torch.Tensor | None = None) -> torch.Tensor:
        parts = [stock, self.ticker_embedding(ticker_id)]
        external = self.exogenous_representation(exogenous)
        if external is not None:
            parts.insert(1, external)
        return self.fusion(torch.cat(parts, dim=1))

    def forward(self, stock_sequence: torch.Tensor, ticker_id: torch.Tensor, *,
                exogenous_sequence: torch.Tensor | None = None) -> torch.Tensor:
        stock = self.stock_representation(stock_sequence)
        fused = self.fuse(stock, ticker_id, exogenous_sequence)
        return self.head(fused).reshape(-1)


def build_model(architecture: NeuralArchitecture, *, seed: int = 42
                ) -> DualSequenceDirectionModel:
    """Deterministic construction."""
    torch.manual_seed(seed)
    return DualSequenceDirectionModel(architecture)


def model_parameter_count(model: nn.Module) -> dict:
    return {"total_parameters": int(sum(p.numel() for p in model.parameters())),
            "trainable_parameters": int(
                sum(p.numel() for p in model.parameters() if p.requires_grad))}


def predict_probabilities(model: nn.Module, loader, device: torch.device
                          ) -> np.ndarray:
    """Sigmoid of the direction logit, in loader order."""
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            exogenous = batch.get("exogenous_sequence")
            logits = model(
                batch["stock_sequence"].to(device),
                batch["ticker_id"].to(device),
                exogenous_sequence=(None if exogenous is None
                                    else exogenous.to(device)))
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def exogenous_window_is_finite(samples, panel: pd.DataFrame, columns, *,
                               sequence_length: int = 60) -> np.ndarray:
    """Per-sample check that EVERY exogenous value in the 60-day window exists.

    A neural sequence needs all 60 exogenous values to be FINITE, not merely present.
    Rows that fail are dropped from the neural fit rather than padded or filled with
    an invented value, which is what the missing-data policy requires.

    The check is vectorised per security: the panel is ROW-ALIGNED with
    ``samples.frame``, so a matrix row maps to a panel row through one array lookup.
    """
    frame = samples.frame
    total = len(frame)
    if not columns or panel is None:
        return np.ones(total, dtype=bool)
    values = panel.loc[:, list(columns)].to_numpy(dtype=np.float64)
    out = np.zeros(total, dtype=bool)
    for ticker, block in frame.groupby("ticker", sort=True):
        matrices = samples.arrays.matrices[str(ticker)]
        matrix_dates = pd.DatetimeIndex(matrices.dates)
        positions = block.sort_values("origin_date").index.to_numpy()
        sample_dates = pd.DatetimeIndex(pd.to_datetime(
            frame.loc[positions, "origin_date"]))
        # this security's matrix date -> offset into ``positions``
        offset = pd.Series(np.arange(len(positions)), index=sample_dates)
        matrix_rows = frame.loc[positions, "row"].to_numpy(int)
        for position, matrix_row in zip(positions, matrix_rows, strict=True):
            start = int(matrix_row) + 1 - int(sequence_length)
            if start < 0:
                continue
            window = matrix_dates[start:int(matrix_row) + 1]
            offsets = offset.reindex(window).to_numpy(dtype=float)
            if np.isnan(offsets).any():
                continue
            rows = positions[offsets.astype(int)]
            out[position] = bool(np.isfinite(values[rows]).all())
    return out


class DualSequenceDataset(torch.utils.data.Dataset):
    """Yields synchronised stock and exogenous sequences for one origin.

    Both sequences cover the SAME 60 origin dates and end at the origin row, so no
    exogenous value from ``t+1`` onwards can reach the input.
    """

    def __init__(self, samples, frame: pd.DataFrame, *, stock_scaler=None,
                 exogenous_scaler=None, sequence_length: int = 60,
                 exogenous_by_pair: pd.DataFrame | None = None,
                 exogenous_columns: tuple[str, ...] = ()) -> None:
        self.samples = samples
        self.frame = frame.reset_index(drop=True)
        self.stock_scaler = stock_scaler
        self.exogenous_scaler = exogenous_scaler
        self.sequence_length = int(sequence_length)
        self.exogenous_columns = tuple(exogenous_columns)
        # ``exogenous_by_pair`` is keyed by (ticker, origin_date): the date-keyed
        # source features broadcast per security, plus the security-specific relative
        # context.  A sequence is assembled by looking up the SAME 60 (security, date)
        # pairs the stock sequence covers, so the two streams are genuinely
        # synchronised and no date outside the window can enter either.
        self._exogenous_by_pair = None
        if exogenous_by_pair is not None and self.exogenous_columns:
            keys = pd.MultiIndex.from_arrays([
                samples.frame["ticker"].to_numpy(),
                pd.DatetimeIndex(pd.to_datetime(samples.frame["origin_date"]))])
            self._exogenous_by_pair = exogenous_by_pair.copy()
            self._exogenous_by_pair.index = keys

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        matrices = self.samples.arrays.matrices[str(row["ticker"])]
        end = int(row["row"]) + 1
        start = end - self.sequence_length
        stock = matrices.stock[start:end]
        if self.stock_scaler is not None and self.stock_scaler.columns:
            stock = self.stock_scaler.transform(stock)
        item = {
            "stock_sequence": torch.from_numpy(np.ascontiguousarray(stock,
                                                                    dtype=np.float32)),
            "ticker_id": torch.tensor(int(row["ticker_id"]), dtype=torch.long),
            "y_direction": torch.tensor(float(row["y_direction"]), dtype=torch.float32),
            "index": torch.tensor(index, dtype=torch.long),
        }
        if self._exogenous_by_pair is not None and self.exogenous_columns:
            dates = pd.DatetimeIndex(matrices.dates[start:end])
            ticker_key = str(row["ticker"])
            pairs = pd.MultiIndex.from_arrays(
                [np.repeat(ticker_key, len(dates)), dates])
            exogenous = (self._exogenous_by_pair.reindex(pairs)
                         .reindex(columns=list(self.exogenous_columns))
                         .to_numpy(dtype=np.float32))
            if self.exogenous_scaler is not None:
                exogenous = self.exogenous_scaler.transform(exogenous)
            item["exogenous_sequence"] = torch.from_numpy(
                np.ascontiguousarray(exogenous, dtype=np.float32))
        return item
