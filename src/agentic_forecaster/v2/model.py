"""``ContextualLSTMTransformer``: the V2 shared model.

The architecture is deliberately auditable rather than clever.  Every stage can
be switched off by configuration, because the V2 experiment is an ABLATION:
V2-A removes the Transformer and the context, V2-C adds them back, and the
difference between those runs is the only thing the comparison measures.

    stock sequence  ->  unidirectional LSTM (2 x 96)      -> H_lstm [B,T,96]
    context sequence->  Linear(F_ctx,32) + GELU + LayerNorm -> H_ctx [B,T,32]
    [H_lstm, H_ctx] ->  Linear(128,64) + LayerNorm + GELU
    + sinusoidal positional encoding
    ->  causal TransformerEncoder (2 x d_model 64, 4 heads, FF 128)
    ->  learned temporal attention pooling -> pooled [B,64]
    + Linear(96,64) applied to the LAST LSTM hidden state
    ->  LayerNorm  ->  temporal representation [B,64]
    + ticker embedding 16 | sector embedding 8 | regime embedding 16
    ->  fusion MLP -> z [B,64]
    ->  FiLM conditioning (V2-E) -> residual adapter (V2-F) -> task heads

Design notes
------------
* The FULL LSTM hidden-state sequence is returned, never only the final state.
* Attention is masked CAUSALLY even though the whole window is available at the
  origin: it keeps the temporal representation auditable and prevents later
  timesteps from influencing earlier ones inside the window.
* The LSTM residual is explicit so the architecture can fall back towards the
  LSTM representation if the Transformer contributes little.
* V2 is ONE shared model.  A security's identity enters only through a 16-d
  ticker embedding (plus a sector embedding and a regime embedding), never
  through a price level or a separate parameter set.
* FiLM is CONDITIONAL ADAPTATION, not meta-learning.  The Reptile-style
  meta-learning variant lives in ``meta.py`` and adapts only the adapter and the
  heads.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
from torch import nn

from .sectors import UNKNOWN_LABEL


@dataclass
class V2ModelConfig:
    """Every architectural choice of the shared V2 model."""

    n_stock_features: int
    n_context_features: int = 0
    n_regime_features: int = 0
    n_tickers: int = 1
    n_sectors: int = 1
    ticker_vocab: list[str] = field(default_factory=list)
    sector_vocab: list[str] = field(default_factory=list)

    # --- A. stock temporal encoder
    hidden_size: int = 96
    lstm_layers: int = 2
    lstm_dropout: float = 0.20

    # --- B. context projection
    context_dim: int = 32

    # --- C. temporal fusion
    fusion_temporal_dim: int = 64

    # --- D/E. positional encoding + transformer
    max_sequence_length: int = 128
    d_model: int = 64
    transformer_layers: int = 2
    n_heads: int = 4
    dim_feedforward: int = 128
    transformer_dropout: float = 0.10

    # --- F. attention pooling
    pooling_dim: int = 64

    # --- embeddings
    ticker_embedding_dim: int = 16
    sector_embedding_dim: int = 8
    regime_embedding_dim: int = 16
    regime_hidden: int = 32

    # --- fusion MLP
    fusion_hidden: int = 128
    z_dim: int = 64
    fusion_dropout: float = 0.10

    # --- heads
    head_hidden: int = 32

    # --- variant switches
    use_transformer: bool = True
    use_context: bool = False
    use_sector_embedding: bool = False
    use_regime: bool = False
    use_multitask: bool = False
    use_film: bool = False
    use_adapter: bool = False

    # --- FiLM
    film_scale: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> V2ModelConfig:
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in fields})

    @property
    def fusion_input_dim(self) -> int:
        """Temporal representation (d_model) plus every identity embedding."""
        return self.d_model + self.ticker_embedding_dim + (
            self.sector_embedding_dim if self.use_sector_embedding else 0
        ) + (self.regime_embedding_dim if self.use_regime else 0)

    @property
    def temporal_fusion_input(self) -> int:
        return self.hidden_size + (self.context_dim if self.use_context else 0)

    def unknown_sector_index(self) -> int:
        """Index of the UNKNOWN sector slot, which always exists by construction."""
        if UNKNOWN_LABEL in self.sector_vocab:
            return self.sector_vocab.index(UNKNOWN_LABEL)
        return 0


def sinusoidal_position_encoding(max_length: int, d_model: int) -> torch.Tensor:
    """Standard sinusoidal encoding, shape [max_length, d_model]."""
    position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
    half = d_model // 2
    div_term = torch.exp(
        torch.arange(half, dtype=torch.float32) * (-math.log(10000.0) / max(d_model, 1)))
    encoding = torch.zeros(max_length, d_model, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term)
    return encoding


def causal_mask(length: int, device: torch.device | None = None) -> torch.Tensor:
    """Boolean ``[T, T]`` mask, ``True`` where attention is FORBIDDEN.

    ``mask[i, j]`` is True for ``j > i``: position ``i`` may attend to itself and
    to the past only.
    """
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)


class FiLMConditioning(nn.Module):
    """Constrained feature-wise modulation conditioned on identity and regime.

    This is CONDITIONAL ADAPTATION: gamma and beta are a deterministic function
    of the ticker, sector and regime embeddings.  The modulation is bounded
    (``gamma = 1 + 0.1 tanh(.)``, ``beta = 0.1 tanh(.)``) so a badly conditioned
    sample cannot destroy the shared representation.
    """

    def __init__(self, condition_dim: int, feature_dim: int, scale: float = 0.1) -> None:
        super().__init__()
        self.scale = float(scale)
        self.produce = nn.Linear(condition_dim, 2 * feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        # Start as the identity so enabling FiLM cannot break a trained encoder.
        nn.init.zeros_(self.produce.weight)
        nn.init.zeros_(self.produce.bias)

    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, dict]:
        raw_gamma, raw_beta = self.produce(condition).chunk(2, dim=-1)
        gamma = 1.0 + self.scale * torch.tanh(raw_gamma)
        beta = self.scale * torch.tanh(raw_beta)
        adapted = self.norm(gamma * z + beta)
        return adapted, {"gamma": gamma, "beta": beta}


class ResidualAdapter(nn.Module):
    """``z_meta = z + adapter(z)`` -- the only block meta-learning adapts."""

    def __init__(self, feature_dim: int = 64, hidden_dim: int = 32) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        nn.init.zeros_(self.block[-1].weight)
        nn.init.zeros_(self.block[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.block(z)


class ContextualLSTMTransformer(nn.Module):
    """The shared V2 model.  One instance serves every security."""

    def __init__(self, config: V2ModelConfig) -> None:
        super().__init__()
        if config.fusion_temporal_dim != config.d_model:
            raise ValueError(
                f"fusion_temporal_dim ({config.fusion_temporal_dim}) must equal d_model "
                f"({config.d_model}): the fused sequence is fed straight into the "
                "Transformer and the attention pooling"
            )
        self.config = config

        # A. stock temporal encoder -----------------------------------------
        self.lstm = nn.LSTM(
            input_size=config.n_stock_features,
            hidden_size=config.hidden_size,
            num_layers=config.lstm_layers,
            dropout=config.lstm_dropout if config.lstm_layers > 1 else 0.0,
            bidirectional=False,
            batch_first=True,
        )

        # B. context projection ----------------------------------------------
        if config.use_context:
            self.context_encoder = nn.Sequential(
                nn.Linear(config.n_context_features, config.context_dim),
                nn.GELU(),
                nn.LayerNorm(config.context_dim),
            )
        else:
            self.context_encoder = nn.Identity()

        # C. temporal fusion --------------------------------------------------
        self.temporal_fusion = nn.Sequential(
            nn.Linear(config.temporal_fusion_input, config.fusion_temporal_dim),
            nn.LayerNorm(config.fusion_temporal_dim),
            nn.GELU(),
        )

        # D. positional encoding ---------------------------------------------
        self.register_buffer(
            "position_encoding",
            sinusoidal_position_encoding(config.max_sequence_length, config.d_model),
            persistent=False,
        )

        # E. causal transformer ------------------------------------------------
        if config.use_transformer:
            layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.dim_feedforward,
                dropout=config.transformer_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                layer, num_layers=config.transformer_layers, enable_nested_tensor=False)
        else:
            self.transformer = None

        # F. learned temporal attention pooling --------------------------------
        self.pool_score = nn.Linear(config.d_model, 1)
        self.pool_norm = nn.LayerNorm(config.d_model)

        # G. LSTM residual -----------------------------------------------------
        self.lstm_residual = nn.Linear(config.hidden_size, config.d_model)
        self.temporal_norm = nn.LayerNorm(config.d_model)

        # 17. static embeddings ------------------------------------------------
        self.ticker_embedding = nn.Embedding(config.n_tickers, config.ticker_embedding_dim)
        self.sector_embedding = (
            nn.Embedding(config.n_sectors, config.sector_embedding_dim)
            if config.use_sector_embedding else None
        )

        # 18. regime encoder ---------------------------------------------------
        if config.use_regime:
            self.regime_encoder = nn.Sequential(
                nn.Linear(config.n_regime_features, config.regime_hidden),
                nn.GELU(),
                nn.Linear(config.regime_hidden, config.regime_embedding_dim),
                nn.LayerNorm(config.regime_embedding_dim),
            )
        else:
            self.regime_encoder = None

        # 19. fusion representation -------------------------------------------
        self.fusion = nn.Sequential(
            nn.Linear(config.fusion_input_dim, config.fusion_hidden),
            nn.GELU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.fusion_hidden, config.z_dim),
            nn.GELU(),
            nn.LayerNorm(config.z_dim),
        )

        # 20. task heads -------------------------------------------------------
        self.direction_head = nn.Sequential(
            nn.Linear(config.z_dim, config.head_hidden),
            nn.GELU(),
            nn.Linear(config.head_hidden, 1),
        )
        if config.use_multitask:
            self.return_head = nn.Sequential(
                nn.Linear(config.z_dim, config.head_hidden),
                nn.GELU(),
                nn.Linear(config.head_hidden, 1),
            )
            self.rank_head = nn.Sequential(
                nn.Linear(config.z_dim, config.head_hidden),
                nn.GELU(),
                nn.Linear(config.head_hidden, 1),
                nn.Sigmoid(),
            )
        else:
            self.return_head = None
            self.rank_head = None

        # 22. FiLM -------------------------------------------------------------
        self.film = None
        if config.use_film:
            condition_dim = (config.ticker_embedding_dim
                             + (config.sector_embedding_dim if config.use_sector_embedding else 0)
                             + (config.regime_embedding_dim if config.use_regime else 0))
            self.film = FiLMConditioning(condition_dim, config.z_dim, config.film_scale)

        # 23. residual adapter --------------------------------------------------
        self.adapter = (ResidualAdapter(config.z_dim)
                        if config.use_adapter else None)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------ parts

    def encode_temporal(self, stock_sequence: torch.Tensor,
                        context_sequence: torch.Tensor | None,
                        *, return_attention: bool = False
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Temporal branch: LSTM + context fusion + Transformer + pooling.

        Returns ``(temporal_representation, attention_weights, lstm_last)``.
        """
        h_lstm, _ = self.lstm(stock_sequence)                       # [B, T, hidden]

        if self.config.use_context:
            if context_sequence is None or context_sequence.shape[-1] == 0:
                raise ValueError("this variant expects a context sequence")
            h_context = self.context_encoder(context_sequence)      # [B, T, 32]
        else:
            h_context = h_lstm.new_zeros((*h_lstm.shape[:2], 0))

        fused = self.temporal_fusion(torch.cat([h_lstm, h_context], dim=-1))

        length = fused.shape[1]
        if self.transformer is not None:
            if length > self.config.max_sequence_length:
                raise ValueError(
                    f"sequence length {length} exceeds max_sequence_length "
                    f"{self.config.max_sequence_length}"
                )
            positions = self.position_encoding[:length].to(fused.dtype)
            encoded = self.transformer(fused + positions, mask=causal_mask(length, fused.device))
        else:
            encoded = fused

        scores = self.pool_score(self.pool_norm(encoded)).squeeze(-1)   # [B, T]
        attention = torch.softmax(scores, dim=1)
        pooled = torch.bmm(attention.unsqueeze(1), encoded).squeeze(1)  # [B, d_model]

        lstm_last = h_lstm[:, -1, :]
        temporal = self.temporal_norm(pooled + self.lstm_residual(lstm_last))
        if return_attention:
            return temporal, attention, lstm_last
        return temporal, attention, lstm_last

    def condition_vector(self, ticker_id: torch.Tensor, sector_id: torch.Tensor,
                         regime_vector: torch.Tensor | None) -> tuple[torch.Tensor, dict]:
        """Ticker / sector / regime embeddings, concatenated."""
        parts = [self.ticker_embedding(ticker_id)]
        embeddings = {"ticker": parts[0]}
        if self.sector_embedding is not None:
            embeddings["sector"] = self.sector_embedding(sector_id)
            parts.append(embeddings["sector"])
        if self.regime_encoder is not None:
            if regime_vector is None:
                raise ValueError("this variant expects a regime vector")
            embeddings["regime"] = self.regime_encoder(regime_vector)
            parts.append(embeddings["regime"])
        return torch.cat(parts, dim=-1), embeddings

    # ---------------------------------------------------------------- forward

    def forward(self, stock_sequence: torch.Tensor, ticker_id: torch.Tensor, *,
                context_sequence: torch.Tensor | None = None,
                sector_id: torch.Tensor | None = None,
                regime_vector: torch.Tensor | None = None,
                return_attention: bool = False) -> dict[str, torch.Tensor]:
        """Run the shared model.

        Returns a dict with ``direction_logit`` plus, for a multi-task variant,
        ``return_prediction`` (volatility-normalised) and ``rank_prediction`` in
        [0, 1], together with ``z``, the adapted representation and (on request)
        the temporal attention weights.
        """
        if sector_id is None:
            sector_id = torch.zeros_like(ticker_id)

        temporal, attention, _ = self.encode_temporal(
            stock_sequence, context_sequence, return_attention=return_attention)

        condition, _ = self.condition_vector(ticker_id, sector_id, regime_vector)
        z = self.fusion(torch.cat([temporal, condition], dim=-1))

        outputs: dict[str, torch.Tensor] = {"z": z}
        film_stats: dict[str, torch.Tensor] = {}
        if self.film is not None:
            z, film_stats = self.film(z, condition)
        outputs["z_adapted"] = z

        if self.adapter is not None:
            z = self.adapter(z)
        outputs["z_meta"] = z

        outputs["direction_logit"] = self.direction_head(z).squeeze(-1)
        if self.return_head is not None:
            outputs["return_prediction"] = self.return_head(z).squeeze(-1)
        if self.rank_head is not None:
            outputs["rank_prediction"] = self.rank_head(z).squeeze(-1)
        if film_stats:
            outputs["film_gamma"] = film_stats["gamma"]
            outputs["film_beta"] = film_stats["beta"]
        if return_attention:
            outputs["temporal_attention"] = attention
        return outputs

    # ------------------------------------------------------------- meta hooks

    def adaptable_parameter_names(self) -> list[str]:
        """Parameter names meta-learning is allowed to adapt.

        The shared encoder (LSTM, Transformer, context projection, embeddings)
        is deliberately EXCLUDED: only the residual adapter and the task heads
        are meta-adapted.
        """
        names: list[str] = []
        prefixes = ("adapter.", "direction_head.", "return_head.", "rank_head.")
        for name, _ in self.named_parameters():
            if name.startswith(prefixes):
                names.append(name)
        return names

    def shared_parameter_names(self) -> list[str]:
        adaptable = set(self.adaptable_parameter_names())
        return [n for n, _ in self.named_parameters() if n not in adaptable]

    def set_encoder_frozen(self, frozen: bool = True) -> None:
        """Freeze or unfreeze the shared encoder (everything but adapter/heads)."""
        for name, parameter in self.named_parameters():
            if name not in set(self.adaptable_parameter_names()):
                parameter.requires_grad_(not frozen)

    def n_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "adaptable_parameters": int(sum(
                p.numel() for n, p in self.named_parameters()
                if n in set(self.adaptable_parameter_names()))),
        }


def build_model(config: V2ModelConfig, *, seed: int | None = None) -> ContextualLSTMTransformer:
    """Deterministic construction: the same seed gives the same initialisation."""
    if seed is not None:
        torch.manual_seed(seed)
    return ContextualLSTMTransformer(config)