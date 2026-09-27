# Paper-to-Code Traceability

**Paper:** Explanation-First Agentic Forecaster for Stock Market
**DOI:** 10.1109/IEMENTECH202669403.2026.11434302

This document maps each paper section/equation to the implementing code, tests, and configuration.

## Classification legend

| Label | Meaning |
|---|---|
| PAPER-DEFINED | Explicitly stated in the paper |
| AUTHOR-CONFIRMED | Confirmed by the paper authors but not verbatim in the publication |
| RECONSTRUCTION-ASSUMED | Chosen by the reconstruction team where the paper is ambiguous |

## Equation-to-code map

| Paper Section/Equation | Requirement | Classification | Implementation File/Symbol | Test | Config | Status | Notes |
|---|---|---|---|---|---|---|---|
| Eq. 1: OHLCV | Raw OHLCV input per ticker | PAPER-DEFINED | `features/engineer.py:build_feature_frame` (lines 84-87); `data/agent.py:DataAgent.run` (line 135) | `tests/unit/test_features.py` | `paper.yaml: features.use_ohlcv: true` | Implemented | Dataset provides date/open/high/low/close/volume |
| Eq. 2: log return | Log price return feature | PAPER-DEFINED | `features/engineer.py` returns_1/5/10 computed via `close.pct_change(n)` (lines 103-105) | `tests/unit/test_features.py` | `paper.yaml: features.indicators: [returns_1, returns_5, returns_10]` | Implemented | Simple return used; log variant available via log-price transform |
| Eq. 3: realized volatility | Rolling realized volatility | PAPER-DEFINED | `features/engineer.py` line 94: `close.pct_change().rolling(20).std()` | `tests/unit/test_features.py` | `paper.yaml: features.indicators: [volatility_20]` | Implemented | 20-day rolling std of returns |
| Eq. 4-6: RSI | Relative Strength Index (14-day, Wilder's smoothing) | PAPER-DEFINED | `features/engineer.py:rsi` (lines 13-22) | `tests/unit/test_features.py` | `paper.yaml: features.indicators: [rsi_14]` | Implemented | EMA with alpha=1/period; fillna(50.0) for warmup |
| Eq. 7: next-day target | Binary direction: 1 if close[t+1] > close[t] | PAPER-DEFINED | `features/engineer.py` lines 116-117; `data/agent.py` target construction | `tests/unit/test_sequencing.py`, `tests/integration/test_data_agent.py` | `paper.yaml: data.target_type: direction, data.target_horizon: 1` | Implemented | Last row target set to NaN and dropped |
| Eq. 8-9: 85/15 split | Train/test chronological split (85% train, 15% test) | PAPER-DEFINED | `data/agent.py:DataAgent.run` (lines 145-208) | `tests/unit/test_splits.py`, `tests/integration/test_data_agent.py` | `paper.yaml: data.train_start/train_end/val_start/val_end/test_start/test_end` | Implemented | Date-based split via paper.yaml; fractional fallback (70/15/15) available |
| System architecture: five agents | Data, Model, Explainer, Risk, Report agents | PAPER-DEFINED | `agents/model_agent.py`, `agents/explainer_agent.py`, `risk/risk_agent.py`, `agents/report_agent.py`, `data/agent.py` | `tests/integration/test_pipeline.py`, `tests/integration/test_packaging.py` | `paper.yaml: orchestration.agents: [data, model, explainer, risk, report]` | Implemented | Orchestrated by `orchestration/pipeline.py` |
| Eq. 11: attention score | Additive attention: v^T tanh(W_h h_t + b) | RECONSTRUCTION-ASSUMED | `models/attention_lstm.py:AttentionLSTM.forward` lines 35-38 | `tests/unit/test_attention.py`, `tests/unit/test_models.py` | `paper.yaml: models.attention_lstm.hidden_size: 64` | Implemented | Bahdanau-style; variant not specified in paper |
| Eq. 12: attention normalization | Softmax over time-step scores | PAPER-DEFINED | `models/attention_lstm.py` line 48: `torch.softmax(scores, dim=1)` | `tests/unit/test_attention.py` | — | Implemented | Ensures weights sum to 1 |
| Eq. 13: context vector | Weighted sum of LSTM outputs: c = sum(alpha_t * h_t) | PAPER-DEFINED | `models/attention_lstm.py` line 49: `torch.bmm(weights.unsqueeze(1), lstm_out)` | `tests/unit/test_attention.py` | — | Implemented | Shape (B, H) |
| Eq. 14: sigmoid prediction | Sigmoid output for up/down probability | PAPER-DEFINED | `models/attention_lstm.py:classifier` (linear head); softmax applied in training loop; sigmoid via softmax equivalence | `tests/unit/test_models.py` | — | Implemented | Two-class softmax equivalent to sigmoid |
| Eq. 15: training objective | Cross-entropy loss with Adam | PAPER-DEFINED | `training/trainer.py:Trainer.fit` lines 79-99; Adam optimizer line 80 | `tests/unit/test_models.py`, `tests/integration/test_pipeline.py` | `paper.yaml: models.attention_lstm.learning_rate: 0.001, weight_decay: 0.0001` | Implemented | Class-weighted CrossEntropyLoss |
| Eq. 16: stop loss | 5% adverse-move stop loss | PAPER-DEFINED | `risk/risk_agent.py:RiskAgent.decide` line 84: `stop_loss=self.stop_loss_pct` | `tests/unit/test_risk.py`, `tests/unit/test_risk_extended.py` | `paper.yaml: risk.stop_loss_pct: 0.05` | Implemented | |
| Eq. 17: take profit | 2x stop-loss take profit | PAPER-DEFINED | `risk/risk_agent.py:RiskAgent.decide` line 85: `take_profit=2.0 * self.stop_loss_pct` | `tests/unit/test_risk.py` | Derived from stop_loss_pct | Implemented | 10% take-profit at default settings |
| Eq. 18: conviction parameters | Low < 55%, medium 55-70%, high > 85% | PAPER-DEFINED | `risk/risk_agent.py:risk_level` thresholds (lines 69-76); `agents/report_agent.py` | `tests/unit/test_risk_extended.py` | `paper.yaml: reporting.conviction_thresholds: {low: 0.55, medium: 0.70, high: 0.85}` | Implemented | Also: no_trade below 1% position |
| Eq. 19: RRR | Risk-reward ratio | PAPER-DEFINED | `risk/risk_agent.py` (implicit: take_profit / stop_loss = 2.0) | `tests/unit/test_risk.py` | `paper.yaml: risk.stop_loss_pct: 0.05` | Implemented | RRR = 2:1 via take_profit = 2x stop_loss |
| Eq. 20: RiskScore | Half-Kelly with volatility targeting | PAPER-DEFINED | `risk/risk_agent.py:RiskAgent.kelly_size` lines 46-51; `vol_size` lines 53-56; `decide` lines 58-86 | `tests/unit/test_risk.py`, `tests/unit/test_risk_extended.py` | `paper.yaml: risk.kelly_fraction: 0.25, risk.max_position_pct: 0.10, risk.volatility_target: 0.15` | Implemented | f* = 0.25 * (p*b - (1-p))/b; capped by max_position_pct |
| Eq. 21: walk-forward folds | Anchored expanding-window folds | PAPER-DEFINED | `evaluation/walk_forward.py:walk_forward_folds` (lines 31-55); `paper_walk_forward_folds` (lines 58-69) | `tests/unit/test_walk_forward.py`, `tests/integration/test_pipeline.py` | `paper.yaml: evaluation.walk_forward.scheme: anchored, step: 21` | Implemented | Two paper-specific folds (2016-2022, 2016-2023) |
| Eq. 22: accuracy | Classification accuracy | PAPER-DEFINED | `evaluation/metrics.py:compute_metrics` line 52 | `tests/unit/test_metrics.py` | `paper.yaml: evaluation.metrics: [accuracy, ...]` | Implemented | threshold = 0.5 |
| Eq. 23: F1 | F1 score | PAPER-DEFINED | `evaluation/metrics.py` line 56: `f1_score` | `tests/unit/test_metrics.py` | `paper.yaml: evaluation.metrics: [f1, ...]` | Implemented | zero_division=0 |
| Eq. 24: Precision@K | Precision at top-K (K=3) | PAPER-DEFINED | `evaluation/metrics.py:precision_at_k` (lines 32-39) | `tests/unit/test_metrics.py` | `paper.yaml: evaluation.metrics: [precision_at_3]` | Implemented | K=3 configurable |
| Eq. 25: Brier | Brier score | PAPER-DEFINED | `evaluation/metrics.py` line 53: `brier_score_loss` | `tests/unit/test_metrics.py` | `paper.yaml: evaluation.metrics: [brier]` | Implemented | Binary Brier |
| Eq. 26: ECE | Expected Calibration Error (10 equal-width bins) | PAPER-DEFINED | `evaluation/metrics.py:expected_calibration_error` (lines 15-29) | `tests/unit/test_metrics.py`, `tests/unit/test_calibration_extended.py` | `paper.yaml: evaluation.ece_bins: 10` | Implemented | Equal-width bins |

## Non-equation components

| Paper component | Requirement | Classification | Implementation File/Symbol | Test | Config | Status | Notes |
|---|---|---|---|---|---|---|---|
| SHAP explanations | Feature attribution per prediction | PAPER-DEFINED | `explainability/shap_explainer.py:ShapExplainer` (DeepExplainer + permutation fallback) | `tests/unit/test_models.py`, `tests/integration/test_pipeline.py` | `paper.yaml: explainability.method: shap, shap_explainer: deep` | Implemented | DeepExplainer for torch; permutation fallback |
| LLM explanations | Optional natural-language narration | PAPER-DEFINED | `agents/explainer_agent.py:_llm_narrate` | `tests/integration/test_pipeline.py` | `paper.yaml: reporting.llm.enabled: false` | Implemented | OpenAI-compatible endpoint; disabled by default |
| Deterministic explanation fallback | Template-based explanation when LLM disabled | RECONSTRUCTION-ASSUMED | `agents/explainer_agent.py:_deterministic_narrate` | `tests/integration/test_pipeline.py` | `paper.yaml: reporting.llm.fallback: deterministic` | Implemented | Only mode exercised in CI |
| PDF reports | Per-ticker PDF report | PAPER-DEFINED | `agents/report_agent.py:_render_pdf`; `reporting/pdf_report.py` | `tests/unit/test_reporting.py` | `paper.yaml: reporting.formats: [html, pdf]` | Implemented | reportlab-based |
| Streamlit | Interactive dashboard | PAPER-DEFINED | `app/streamlit_app.py` | — | — | Implemented | Demo app |
| Calibration | Temperature scaling on validation split | PAPER-DEFINED | `calibration/temperature.py:TemperatureCalibrator` | `tests/unit/test_calibration.py`, `tests/unit/test_calibration_extended.py` | `paper.yaml: calibration.method: temperature, fit_on: val` | Implemented | LBFGS optimizer; scalar T |
| Baselines | Plain LSTM, Random Forest, Logistic Regression, Majority | PAPER-DEFINED | `models/lstm.py`, `models/baselines.py` | `tests/unit/test_models.py` | `paper.yaml: models.lstm/random_forest/logistic_regression/majority.enabled: true` | Implemented | All baselines receive flattened (30 x F) input |
| Ablations | Toggle model families | PAPER-DEFINED | `models/__init__.py`; `orchestration/pipeline.py` | `tests/integration/test_pipeline.py` | `paper.yaml: models.<name>.enabled` | Implemented | Config-driven toggles |
| 50 per-stock models | One model per ticker (50 NIFTY 100 constituents) | PAPER-DEFINED | `scripts/train_all.py`; `orchestration/pipeline.py` | `tests/integration/test_pipeline.py` | `paper.yaml: data.tickers: configs/nifty50.yaml` | Implemented | 50 individual checkpoints |
| 30-day lookback | Sequence length of 30 trading days | PAPER-DEFINED | `data/agent.py` line 100; `models/attention_lstm.py` | `tests/unit/test_sequencing.py` | `paper.yaml: data.sequence_length: 30` | Implemented | Configurable |
| Attention evidence | Top-5 most-attended days per prediction | PAPER-DEFINED | `explainability/attention.py:extract_attention_evidence` | `tests/unit/test_attention.py` | `paper.yaml: explainability.attention_evidence: true` | Implemented | Returns day_offset + attention_weight |

## Five-agent workflow

| Paper agent | Classification | Entry point |
|---|---|---|
| Data Agent | PAPER-DEFINED | `DataAgent.run()` |
| Model Agent | PAPER-DEFINED | `ModelAgent.run()` |
| Explainer Agent | PAPER-DEFINED | `ExplainerAgent.run()` |
| Risk Agent | PAPER-DEFINED | `RiskAgent.decide()` |
| Report Agent | PAPER-DEFINED | `ReportAgent.run()` |

## Configuration map

| Paper hyperparameter | Config key | Value |
|---|---|---|
| Lookback window | `data.sequence_length` | 30 |
| LSTM hidden size | `models.attention_lstm.hidden_size` | 64 |
| LSTM layers | `models.attention_lstm.num_layers` | 2 |
| Dropout | `models.attention_lstm.dropout` | 0.2 |
| Learning rate | `models.attention_lstm.learning_rate` | 0.001 |
| Batch size | `models.attention_lstm.batch_size` | 64 |
| Early stopping patience | `models.attention_lstm.patience` | 15 |
| ECE bins | `evaluation.ece_bins` | 10 |
| Kelly fraction | `risk.kelly_fraction` | 0.25 |
| Max position | `risk.max_position_pct` | 0.10 |
| Stop loss | `risk.stop_loss_pct` | 0.05 |
