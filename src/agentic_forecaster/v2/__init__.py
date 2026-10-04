"""MODEL V2 -- a NEW experimental architecture family.

MODEL V2 IS NOT THE ORIGINAL PAPER MODEL.
=========================================

This package is a deliberately separate namespace.  Nothing here is imported by
the faithful paper reconstruction (``models/attention_lstm.py``,
``agents/model_agent.py``, ``data/agent.py``, ``configs/paper.yaml``), and
nothing here writes into ``results/reproduction_recovery/``.

What V2 changes relative to the paper reconstruction
---------------------------------------------------
* ONE SHARED model across many securities instead of one independent model per
  stock.
* STATIONARY, per-security features (returns, ranges, normalised indicators)
  instead of raw absolute OHLCV price levels.
* CONTEXT derived from the cross-section itself (leave-one-out market and
  sector statistics, cross-sectional ranks), because absolute levels and
  raw price history are not comparable across securities.
* A single sequence is split into a stock-only stream and a context stream that
  are fused per timestep.
* Transformer attention with a causal mask over the whole historical window,
  attention pooling and an explicit LSTM residual.
* Multi-task supervision (direction, volatility-normalised return,
  cross-sectional next-day rank).
* Optional conditional (FiLM) adaptation and an optional Reptile-style
  lightweight meta-learning stage that adapts only a residual adapter and the
  task heads.

See ``docs/MODEL_V2.md`` for the scientific status of this system.
"""

from __future__ import annotations

__all__ = [
    "IS_ORIGINAL_PAPER_MODEL",
    "MODEL_V2_CLASSIFICATION",
]

#: Machine-readable classification recorded in manifests, checkpoints and the
#: development report.  V2 must never be presented as the paper's model.
MODEL_V2_CLASSIFICATION = "NEW_EXPERIMENTAL_ARCHITECTURE"

#: Permanent, machine-checkable reminder of the paper firewall: V2 is a new
#: system.  There is no import path from this package to ``paper_reference``.
IS_ORIGINAL_PAPER_MODEL = False