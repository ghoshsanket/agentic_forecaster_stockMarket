# Implementation Assumptions

Every reconstruction decision is classified with one of three labels:

| Label | Meaning |
|---|---|
| **PAPER-DEFINED** | Explicitly present in the publication |
| **AUTHOR-CONFIRMED** | Confirmed by the authors but not stated verbatim in the publication |
| **RECONSTRUCTION-ASSUMED** | Chosen by the reconstruction team where the paper is silent or ambiguous |

## Data

| # | Item | Label | Choice / rationale |
|---|---|---|---|
| 1 | Stock universe | PAPER-DEFINED | 50 NIFTY-50 stocks. The publication uses NIFTY-50, not "50 NIFTY-100 stocks". |
| 2 | Constituent as-of date | RECONSTRUCTION-ASSUMED | NIFTY-50 membership is time-varying and the publication states no as-of date. We fix a concrete list in `configs/nifty50.yaml` and record unresolved members as unavailable. |
| 3 | Symbol aliases | RECONSTRUCTION-ASSUMED | Only genuine symbol-format spellings of the SAME security are aliased (e.g. `M&M` → `MM`). A different company is never substituted (e.g. `TATAMOTORS` is NOT `TATACOMM`). |
| 4 | Intraday → daily resampling | PAPER-DEFINED | open=first, high=max, low=min, close=last, volume=sum per trading day. |
| 5 | Target | PAPER-DEFINED | `y_t = 1 if Close[t+1] > Close[t] else 0` (next-DAY direction). |
| 6 | Lookback | AUTHOR-CONFIRMED | 30-day sequence. |
| 7 | Python | AUTHOR-CONFIRMED | Python 3.11 is the project runtime policy. |
| 8 | PyTorch | AUTHOR-CONFIRMED | PyTorch is the deep-learning framework. |
| 9 | StandardScaler | RECONSTRUCTION-ASSUMED | One scaler per stock, per fold, fit on training rows only. The paper does not state the scaling convention. |
| 10 | Missing values | RECONSTRUCTION-ASSUMED | Rows with NaN after indicator computation are dropped. |
| 11 | Split assignment | RECONSTRUCTION-ASSUMED | A sample belongs to a window only when BOTH origin date and target date fall inside it (prevents boundary label leakage). |

## Features

| # | Item | Label | Choice / rationale |
|---|---|---|---|
| 12 | Feature subset | RECONSTRUCTION-ASSUMED | Where the paper is ambiguous about the exact subset, we use OHLCV + log-return + 20-day realised volatility + RSI-14 + MACD (line/signal/histogram) + ATR-14. |
| 13 | log_return | PAPER-DEFINED | `log(close[t] / close[t-1])`. |
| 14 | realised volatility | PAPER-DEFINED | Rolling standard deviation of LOG returns. |
| 15 | Indicator parameters | RECONSTRUCTION-ASSUMED | RSI-14 (Wilder), MACD 12/26/9, ATR-14 (Wilder) — industry defaults. |
| 16 | Extra indicators | N/A | SMA/EMA/Bollinger/OBV/multi-period returns remain implemented but are NOT enabled in `configs/paper.yaml`. |

## Model

| # | Item | Label | Choice / rationale |
|---|---|---|---|
| 17 | One model per stock | AUTHOR-CONFIRMED | One independent model per stock; no pooled model. |
| 18 | Attention formulation | PAPER-DEFINED | Additive (Bahdanau-style) attention over the LSTM time steps. |
| 19 | Binary head | PAPER-DEFINED | `Linear(hidden_size, 1)` → `BCEWithLogitsLoss` → `p_up = sigmoid(logit)`. |
| 20 | LSTM layers | RECONSTRUCTION-ASSUMED | 2 layers. |
| 21 | Hidden size | RECONSTRUCTION-ASSUMED | 64 units. |
| 22 | Dropout | RECONSTRUCTION-ASSUMED | 0.2. |
| 23 | Baselines | PAPER-DEFINED | Plain LSTM, Random Forest, Logistic Regression, Majority Class. |

## Training

| # | Item | Label | Choice / rationale |
|---|---|---|---|
| 24 | Epochs | AUTHOR-CONFIRMED | Maximum 3 epochs. **Not** paper-defined. |
| 25 | Optimizer | PAPER-DEFINED | Adam, lr = 0.001, weight decay = 0.0001. |
| 26 | Adam betas | RECONSTRUCTION-ASSUMED | (0.9, 0.999). |
| 27 | Batch size | PAPER-DEFINED | 64. |
| 28 | Patience | PAPER-DEFINED | 10. |
| 29 | Gradient clipping | RECONSTRUCTION-ASSUMED | `gradient_clip_norm = 1.0`. |
| 30 | Seeds | RECONSTRUCTION-ASSUMED | Global seed 42. |

## Calibration

| # | Item | Label | Choice / rationale |
|---|---|---|---|
| 31 | Method | RECONSTRUCTION-ASSUMED | Temperature scaling (single scalar `T`) fit on validation. The paper mentions calibration but the exact method is not explicitly recoverable. |
| 32 | CalibrationFactor | RECONSTRUCTION-ASSUMED | Treated as directional confidence. |

## Explainability

| # | Item | Label | Choice / rationale |
|---|---|---|---|
| 33 | SHAP implementation | RECONSTRUCTION-ASSUMED | `shap.GradientExplainer` over a binary-logit wrapper; Integrated Gradients fallback, always labelled. |
| 34 | Attention evidence | PAPER-DEFINED | Top attended days within the 30-day window. |
| 35 | LLM explanation usage | AUTHOR-CONFIRMED | An LLM narrates the structured evidence. `openai` is imported lazily and only when the provider is enabled. |

## Risk (Phase-1)

| # | Item | Label | Choice / rationale |
|---|---|---|---|
| 36 | ATR risk equations | PAPER-DEFINED | `SL = close ∓ lambda_SL·ATR`, `TP = close ± lambda_TP·ATR` by direction. |
| 37 | Confidence multipliers | PAPER-DEFINED | HIGH (>0.80): 0.8 / 1.5; MEDIUM (>0.60): 1.0 / 1.0; LOW: 1.2 / 0.8. |
| 38 | Bearish symmetric policy | RECONSTRUCTION-ASSUMED | The DOWN branch mirrors the UP branch. |
| 39 | RRR / RiskScore | PAPER-DEFINED | `RRR = abs(TP-close)/abs(close-SL)`; `RiskScore = confidence / max(ATR, 1e-8)`. |
| 40 | Position sizing | **REMOVED** | Half-Kelly / volatility targeting are NOT part of Phase-1 and have been deleted. |

## Evaluation

| # | Item | Label | Choice / rationale |
|---|---|---|---|
| 41 | Walk-forward folds | PAPER-DEFINED | Exactly two folds: FOLD 0 (train 2016-2020, val 2021, test 2022) and FOLD 1 (train 2016-2021, val 2022, test 2023). |
| 42 | Precision@3 | PAPER-DEFINED | Cross-sectional by date; UP and DOWN reported separately. |
| 43 | ECE bin count | RECONSTRUCTION-ASSUMED | 10 equal-width bins. |
| 44 | Metrics | PAPER-DEFINED | Accuracy, F1, Brier, ECE, Precision@3, ROC-AUC. |
| 45 | Raw vs calibrated | PAPER-DEFINED | Brier/ECE are reported for both uncalibrated and calibrated probabilities. |
| 46 | Aggregate policy | RECONSTRUCTION-ASSUMED | Unweighted mean of per-ticker test-split metrics. |
