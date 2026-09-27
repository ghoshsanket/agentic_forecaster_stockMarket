# Implementation Assumptions

This document records every reconstruction assumption made when translating
the paper into code. Where the paper is ambiguous, the chosen default is
stated explicitly.

## Classification legend

| Label | Meaning |
|---|---|
| PAPER-DEFINED | Explicitly stated in the paper |
| AUTHOR-CONFIRMED | Confirmed by the paper authors but not verbatim in the publication |
| RECONSTRUCTION-ASSUMED | Chosen by the reconstruction team where the paper is ambiguous |

## Data

### 1. Stock universe

- **Classification:** PAPER-DEFINED
- **Choice:** 50 NIFTY 100 constituents listed in `configs/nifty50.yaml`.
- **Rationale:** The paper specifies 50 NIFTY 100 stocks.
- **Alternatives considered:** NIFTY 50 (too few), S&P 500 (wrong market).

### 2. Target definition

- **Classification:** RECONSTRUCTION-ASSUMED
- **Choice:** Next-day close-to-close direction: `1` if `close[t+1] > close[t]` else `0`.
- **Rationale:** The paper says "next-day direction" without specifying intraday vs close-to-close.
- **Alternatives considered:** Open-to-open (different trading session boundary); intraday high/low (requires different label construction).

### 3. Lookback window

- **Classification:** PAPER-DEFINED
- **Choice:** 30 trading days, configurable via `data.sequence_length`.
- **Rationale:** The paper explicitly states 30-day lookback.
- **Alternatives considered:** None.

### 4. Scaler fit scope

- **Classification:** RECONSTRUCTION-ASSUMED
- **Choice:** `StandardScaler` fit on the training split only, then applied to val/test.
- **Rationale:** Prevents look-ahead bias from using future data statistics.
- **Alternatives considered:** Fit on full dataset (leaks future information); fit per-ticker (valid but more complex).

### 5. Missing values

- **Classification:** RECONSTRUCTION-ASSUMED
- **Choice:** Rows with NaN after indicator computation are dropped (`features.drop_na: true`).
- **Rationale:** LSTM cannot handle NaN; dropping is the simplest correct approach.
- **Alternatives considered:** Forward-fill (introduces stale data); interpolation (introduces synthetic data).

### 6. Split strategy

- **Classification:** PAPER-DEFINED
- **Choice:** Chronological date-based splits via `train_start/train_end/val_start/val_end/test_start/test_end` in config.
- **Rationale:** The paper uses an 85/15 train/test split chronologically.
- **Alternatives considered:** Random split (leaks future); fractional fallback (70/15/15) available when dates not specified.

## Features

### 7. Indicator set

- **Classification:** RECONSTRUCTION-ASSUMED
- **Choice:** 17 indicators: RSI-14, MACD line/signal/histogram, ATR-14, 20-day volatility, SMA-20/50, EMA-12/26, Bollinger upper/lower/width, OBV, 1/5/10-day returns, log-volume.
- **Rationale:** The paper mentions RSI, MACD, ATR and volatility explicitly; remaining indicators are standard additions covered by the paper's "technical indicators" phrasing.
- **Alternatives considered:** Minimal set (only paper-named indicators); extended set (add Stochastic, CCI, Williams %R).

### 8. RSI smoothing

- **Classification:** RECONSTRUCTION-ASSUMED
- **Choice:** Wilder's smoothing (EMA with `alpha = 1/period`).
- **Rationale:** Standard for RSI; the paper says "RSI" without specifying the smoothing variant.
- **Alternatives considered:** Simple moving average (less standard); Cutler's RSI (SMA-based).

### 9. MACD parameters

- **Classification:** RECONSTRUCTION-ASSUMED
- **Choice:** Standard 12/26/9 EMA parameterisation.
- **Rationale:** Industry default; paper says "MACD" without specifying periods.
- **Alternatives considered:** 8/21/9 (faster); adaptive MACD (over-engineered).

### 10. ATR smoothing

- **Classification:** RECONSTRUCTION-ASSUMED
- **Choice:** Wilder's smoothing of the true range.
- **Rationale:** Standard for ATR.
- **Alternatives considered:** Simple moving average (less standard).

## Models

### 11. Attention mechanism variant

- **Classification:** RECONSTRUCTION-ASSUMED
- **Choice:** Additive (Bahdanau-style) attention over the 30 LSTM output time steps.
- **Rationale:** The paper says "attention" without specifying the variant; additive attention is the most common default and produces interpretable per-timestep weights.
- **Alternatives considered:** Dot-product attention (no interpretable alignment); multi-head attention (paper specifies single-head); Luong attention (multiplicative, less common for this architecture).

### 12. LSTM hidden size and layers

- **Classification:** PAPER-DEFINED
- **Choice:** 64 hidden units, 2 layers, dropout 0.2.
- **Rationale:** Paper's stated hyperparameters.
- **Alternatives considered:** None.

### 13. Class imbalance handling

- **Classification:** RECONSTRUCTION-ASSUMED
- **Choice:** `class_weight: balanced` for LSTM, RF and LR.
- **Rationale:** Financial direction targets are often imbalanced; balanced weights prevent majority-class collapse.
- **Alternatives considered:** Oversampling (duplicates data); SMOTE (synthetic samples); no weighting (risks bias).

### 14. Baseline hyperparameters

- **Classification:** RECONSTRUCTION-ASSUMED
- **Choice:** Random Forest (300 trees, max_depth 8, min_samples_leaf 5), Logistic Regression (C=1.0, max_iter=1000), Majority.
- **Rationale:** Standard defaults for fair comparison; paper mentions baselines without full hyperparameters.
- **Alternatives considered:** Grid search per baseline (computationally expensive); different depths/C values.

## Training

### 15. Optimizer

- **Classification:** PAPER-DEFINED
- **Choice:** Adam, learning rate 0.001, weight decay 1e-4.
- **Rationale:** Paper's stated optimizer.
- **Alternatives considered:** SGD (no adaptive learning rate); AdamW (decoupled weight decay).

### 16. Early stopping

- **Classification:** PAPER-DEFINED
- **Choice:** Patience 15 epochs on validation loss; best checkpoint restored.
- **Rationale:** Paper's stated patience.
- **Alternatives considered:** No early stopping (overfitting risk); patience 5 (underfitting risk).

### 17. Seeds

- **Classification:** RECONSTRUCTION-ASSUMED
- **Choice:** Global seed 42 for Python, NumPy and PyTorch.
- **Rationale:** Reproducibility; the paper does not specify a seed.
- **Alternatives considered:** Multiple seed averaging (more robust but expensive); no seeding (irreproducible).

### 18. Loss function

- **Classification:** PAPER-DEFINED
- **Choice:** Cross-entropy loss (class-weighted).
- **Rationale:** Standard for binary classification; paper's stated objective.
- **Alternatives considered:** Focal loss (handles imbalance differently); MSE (not standard for classification).

## Calibration

### 19. Calibration method

- **Classification:** PAPER-DEFINED
- **Choice:** Temperature scaling (single scalar `T`), fit on the validation split by minimising NLL with LBFGS.
- **Rationale:** Paper's stated calibration approach.
- **Alternatives considered:** Platt scaling (more parameters); isotonic regression (non-parametric, needs more data); no calibration (ECE/Brier degrade).

### 20. Calibration effect

- **Classification:** PAPER-DEFINED
- **Choice:** A scalar `T` preserves ranking, so accuracy and Precision@3 are unchanged; only Brier and ECE improve.
- **Rationale:** Mathematical property of scalar temperature scaling.
- **Alternatives considered:** Per-class temperature (breaks ranking invariance).

## Explainability

### 21. SHAP variant

- **Classification:** RECONSTRUCTION-ASSUMED
- **Choice:** `DeepExplainer` for the torch model, with a permutation-based fallback when SHAP is unavailable or fails.
- **Rationale:** The paper says "SHAP" without specifying the explainer variant; DeepExplainer is the standard for deep models.
- **Alternatives considered:** KernelExplainer (model-agnostic but slow); GradientExplainer (torch-specific, similar to DeepExplainer); TreeExplainer (only for tree models).

### 22. Attention evidence extraction

- **Classification:** PAPER-DEFINED
- **Choice:** The top-5 most-attended days per prediction are extracted from the attention weight vector.
- **Rationale:** Paper's stated attention evidence approach.
- **Alternatives considered:** All days (uninterpretable); threshold-based (loses magnitude information).

### 23. LLM narration

- **Classification:** PAPER-DEFINED
- **Choice:** Optional OpenAI-compatible endpoint; when disabled (default) a deterministic template is used.
- **Rationale:** The deterministic fallback is the only explanation mode exercised in CI.
- **Alternatives considered:** Always-on LLM (non-reproducible); no LLM (paper requires it).

## Risk

### 24. Position sizing

- **Classification:** PAPER-DEFINED
- **Choice:** Half-Kelly: `f* = 0.25 * (p*b - (1-p)) / b` with `b = 1` (even odds). Capped by `max_position_pct` (10%) and by volatility targeting (`vol_target / realised_vol`).
- **Rationale:** Paper's stated risk equations.
- **Alternatives considered:** Full Kelly (too aggressive); fixed fractional (ignores conviction); no vol targeting (ignores regime).

### 25. Stop loss and take profit

- **Classification:** PAPER-DEFINED
- **Choice:** 5% adverse move stop loss; take-profit at 2x stop loss (10%).
- **Rationale:** Paper's stated risk parameters.
- **Alternatives considered:** Trailing stop (more complex); ATR-based stop (adaptive but not paper-defined).

### 26. Conviction thresholds

- **Classification:** PAPER-DEFINED
- **Choice:** Low < 55%, medium 55-70%, high > 85%.
- **Rationale:** Paper's stated conviction bands.
- **Alternatives considered:** Equal thirds (arbitrary); quartiles (data-dependent).

## Evaluation

### 27. Walk-forward scheme

- **Classification:** PAPER-DEFINED
- **Choice:** Anchored scheme, step 21 trading days (~1 month), minimum 3 years of training data.
- **Rationale:** Paper's stated walk-forward approach.
- **Alternatives considered:** Rolling window (loses old data); expanding with different step (changes fold count).

### 28. Metrics

- **Classification:** PAPER-DEFINED
- **Choice:** Accuracy, Brier, ECE (10 equal-width bins), Precision@3, F1, ROC-AUC.
- **Rationale:** Paper's stated evaluation metrics.
- **Alternatives considered:** Log loss (redundant with Brier); Matthews correlation coefficient (not paper-defined).

### 29. Ablations

- **Classification:** PAPER-DEFINED
- **Choice:** Model families can be toggled via `models.<name>.enabled` in the config.
- **Rationale:** Paper's ablation study design.
- **Alternatives considered:** Separate config files (more files to manage); command-line flags (less reproducible).

## Reporting

### 30. Report formats

- **Classification:** PAPER-DEFINED
- **Choice:** HTML (always) and PDF (optional, via reportlab).
- **Rationale:** Paper's stated output formats.
- **Alternatives considered:** Markdown only (no visual layout); Jupyter notebook (not standalone).

### 31. Report content

- **Classification:** RECONSTRUCTION-ASSUMED
- **Choice:** Per-ticker reports bundle forecast, explanation, risk decision and metrics.
- **Rationale:** The paper describes per-stock reports but does not enumerate exact sections.
- **Alternatives considered:** Aggregate-only report (loses per-stock detail); raw JSON only (not human-readable).
