# Implementation Assumptions

This document records every reconstruction assumption made when translating
the paper into code. Each item is classified as:

| Label | Meaning |
|---|---|
| PAPER-DEFINED | Explicitly stated in the paper |
| RECONSTRUCTION-ASSUMED | Chosen by the reconstruction team where the paper is ambiguous |

## Data

### 1. Stock universe — PAPER-DEFINED
- **Choice:** 50 selected NIFTY-50 stocks listed in `configs/nifty50.yaml`.
- **Note:** The publication uses NIFTY-50, NOT "50 NIFTY-100 stocks".

### 2. Intraday → daily resampling — PAPER-DEFINED
- **Choice:** 1-minute OHLCV resampled to daily: open=first, high=max, low=min, close=last, volume=sum.

### 3. Target definition — PAPER-DEFINED
- **Choice:** `y_t = 1 if Close[t+1] > Close[t] else 0`. Next-DAY direction.

### 4. Lookback window — PAPER-DEFINED
- **Choice:** 30 trading days. Input = `feature[t-29]...feature[t]`, target = direction `Close[t] → Close[t+1]`.

### 5. Scaler fit scope — PAPER-DEFINED
- **Choice:** One `StandardScaler` per ticker per fold, fit on TRAIN rows only.

### 6. Missing values — RECONSTRUCTION-ASSUMED
- **Choice:** Rows with NaN after indicator computation are dropped.

## Features

### 7. Phase-1 feature set — PAPER-DEFINED
- **Choice:** open, high, low, close, volume, log_return, realized_volatility_20, rsi_14, macd, macd_signal, macd_histogram, atr_14.
- **Note:** Extra indicators (SMA, EMA, Bollinger, OBV, multi-period returns) remain implemented but are NOT enabled in `configs/paper.yaml`.

### 8. Log return — PAPER-DEFINED
- **Choice:** `log_return[t] = log(close[t] / close[t-1])`.

### 9. Realized volatility — PAPER-DEFINED
- **Choice:** Rolling standard deviation of LOG returns (20-day).

### 10. RSI / MACD / ATR parameters — RECONSTRUCTION-ASSUMED
- **Choice:** RSI-14 (Wilder's), MACD 12/26/9, ATR-14 (Wilder's). Standard defaults; paper names indicators without full parameters.

## Models

### 11. One-model-per-stock — PAPER-DEFINED
- **Choice:** One independent Attention-LSTM per stock. Never a pooled "ALL" model.

### 12. LSTM hidden size / layers / dropout — RECONSTRUCTION-ASSUMED
- **Choice:** 64 hidden units, 2 layers, dropout 0.2. Not explicitly stated in the paper.

### 13. Attention mechanism variant — RECONSTRUCTION-ASSUMED
- **Choice:** Additive (Bahdanau-style) attention. Paper says "attention" without specifying the variant.

### 14. Binary logit output — PAPER-DEFINED
- **Choice:** `Linear(hidden_size, 1)`, `BCEWithLogitsLoss`, `p_up = sigmoid(logit)`.

## Training

### 15. Epochs — PAPER-DEFINED
- **Choice:** Maximum 3 epochs.

### 16. Optimizer — PAPER-DEFINED / RECONSTRUCTION-ASSUMED
- **Choice:** Adam, lr=0.001, betas=(0.9, 0.999), weight_decay=1e-4. Betas are RECONSTRUCTION-ASSUMED (Adam defaults).

### 17. Early stopping — RECONSTRUCTION-ASSUMED
- **Choice:** Patience 10 on validation loss.

### 18. Gradient clipping — RECONSTRUCTION-ASSUMED
- **Choice:** `gradient_clip_norm: 1.0`.

### 19. Seeds — RECONSTRUCTION-ASSUMED
- **Choice:** Global seed 42.

## Calibration

### 20. Calibration method — RECONSTRUCTION-ASSUMED
- **Choice:** Temperature scaling (single scalar T), fit on validation. The paper mentions calibration but the exact method is not explicitly recoverable.

## Explainability

### 21. SHAP variant — RECONSTRUCTION-ASSUMED
- **Choice:** `GradientExplainer` primary; permutation fallback explicitly reported. Paper says "SHAP" without specifying the variant.

### 22. Attention evidence — PAPER-DEFINED
- **Choice:** Top-5 most-attended days per prediction.

### 23. LLM narration — RECONSTRUCTION-ASSUMED
- **Choice:** Optional OpenAI-compatible endpoint with grounding instructions; deterministic fallback is the default.

## Risk — PAPER-DEFINED

### 24. ATR-based stop-loss / take-profit — PAPER-DEFINED
- **Choice:** Confidence bands (HIGH >0.80, MEDIUM >0.60, LOW otherwise) with lambda_SL/lambda_TP multipliers applied to ATR. SL/TP are PRICES, not percentages.
- **Note:** Half-Kelly sizing and fixed 5%/10% stops are NOT paper-defined and have been removed.

### 25. RRR and RiskScore — PAPER-DEFINED
- **Choice:** `RRR = abs(TP-close)/abs(close-SL)`, `RiskScore = confidence / max(ATR, 1e-8)`.

## Evaluation

### 26. Walk-forward folds — PAPER-DEFINED
- **Choice:** Exactly two folds: Fold 0 (train 2016-2020, val 2021, test 2022) and Fold 1 (train 2016-2021, val 2022, test 2023). Retrain every ticker between folds.
- **Note:** Generic 21-day walk-forward is NOT the paper scheme.

### 27. Precision@3 — PAPER-DEFINED
- **Choice:** Cross-sectional by date. UP P@3 and DOWN P@3 computed separately.

### 28. Metrics — PAPER-DEFINED
- **Choice:** Accuracy, Brier, ECE (10 bins), Precision@3, F1, ROC-AUC. Both raw and calibrated Brier/ECE reported.

## Reporting

### 29. Report formats — PAPER-DEFINED
- **Choice:** HTML and PDF per-ticker reports.

### 30. Report content — RECONSTRUCTION-ASSUMED
- **Choice:** Run ID, ticker, origin/target dates, close, raw+calibrated p(up), direction, confidence, indicators, SL/TP prices, RRR, RiskScore, SHAP, attention, reason codes, narrative, model/config IDs, disclaimer.
