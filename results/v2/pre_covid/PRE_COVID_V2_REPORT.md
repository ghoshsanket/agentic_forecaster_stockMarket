# PRE-COVID V2 REPORT

**Regime: `PRE_COVID_EXPERIMENTAL_REGIME`. MODEL V2 IS NOT THE ORIGINAL PAPER MODEL** (classification `NEW_EXPERIMENTAL_ARCHITECTURE`).

- generated: `2026-10-04T06:48:18.999132+00:00`
- survivorship: `SURVIVORSHIP_BIASED_FIXED_UNIVERSE_RESEARCH_TRACK` — the universe is a fixed, later-reconstructed constituent list projected backwards; every number refers to available securities from the reconstructed fixed universe, never to the historical NIFTY-50 membership
- final allowed date: **2019-12-31** (nothing from 2020 onward was consumed)
- 2022/2023 labels accessed: **NO**

This experiment establishes ONE thing: whether a model trained and evaluated entirely before 2020 behaves differently. It does **not** claim that COVID caused any earlier failure — that would be a causal claim this design cannot support.

## A/B. Test and lint results

| check | result |
|---|---|
| ruff (`ruff check .`) | **pass** |
| pytest (`pytest -p no:warnings -q`) | **pass** (exit 0) |
| PRE-COVID store `--verify` | **pass** |
| supervised universe `--verify` | **pass** |

recorded at `2026-10-04T06:45:25.875426+00:00`

## C. PRE-COVID context store

| field | value |
|---|---|
| store root | `/home/iemiedc2026/Documents/Sanket/Research/dataset/agentic-forecaster/processed/v2/pre_covid/context_store` |
| store SHA256 | `9ef053533ab05cfe27c919316503f80b4239e7b97a98acda603d98d29d97ce32` |
| sector-map SHA256 | `e1c7567c4770e4efc30a02f85e45f8896eeff270563f8b8887d9e0e7602895b3` |
| source-manifest SHA256 | `9c8f66bd5db5b7bc6751747f5d6a091a20bd9fc2c7dbe50b1ce4e5290ed9cd25` |
| last feature date | 2019-12-31 |
| last target date | 2019-12-31 |

## D/E/F. Supervised universe (derived, frozen before training)

- eligible securities: **41** of 47 candidates
- excluded: ADANIPORTS, HDFCLIFE, INDIGO, LT, SBILIFE, WIPRO

```
ADANIENT, APOLLOHOSP, ASIANPAINT, AXISBANK, BAJAJ-AUTO, BAJAJFINSV, BAJFINANCE, BEL, BHARTIARTL, CIPLA, COALINDIA, DRREDDY, EICHERMOT, GRASIM, HCLTECH, HDFCBANK, HINDALCO, HINDUNILVR, ICICIBANK, INFY, ITC, JSWSTEEL, KOTAKBANK, M&M, MARUTI, NESTLEIND, NTPC, ONGC, POWERGRID, RELIANCE, SBIN, SHRIRAMFIN, SUNPHARMA, TATACONSUM, TATASTEEL, TCS, TECHM, TITAN, TMPV, TRENT, ULTRACEMCO
```

Exclusion reasons are recorded in `results/v2/pre_covid/pre_covid_supervised_universe.csv`.

## G–J. Development results (2017 and 2018 ONLY)

| model | fold | macro | micro | F1 | balanced | AUC | Brier | ECE | baseline | P@3 up | P@3 down |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LOGISTIC (ceiling check) | PRECOVID_DEV_A | 0.5117 | 0.5117 | 0.5267 | 0.5113 | 0.5198 | 0.2496 | 0.0077 | 0.5036 | 0.5560 | 0.5412 |
| LOGISTIC (ceiling check) | PRECOVID_DEV_B | 0.5111 | 0.5111 | 0.4933 | 0.5108 | 0.5178 | 0.2498 | 0.0069 | 0.5040 | 0.5306 | 0.5415 |
| **LOGISTIC** | MEAN | 0.5114 | 0.5114 | 0.5100 | 0.5110 | 0.5188 | 0.2497 | 0.0073 | 0.5038 | 0.5433 | 0.5413 |
| V2-A | PRECOVID_DEV_A | 0.5134 | 0.5134 | 0.6607 | 0.5059 | 0.5202 | 0.2497 | 0.0034 | 0.5035 | 0.5466 | 0.5115 |
| V2-A | PRECOVID_DEV_B | 0.5166 | 0.5166 | 0.2737 | 0.5132 | 0.5323 | 0.2496 | 0.0126 | 0.5040 | 0.5061 | 0.5374 |
| **V2-A** | MEAN | 0.5150 | 0.5150 | 0.4672 | 0.5095 | 0.5263 | 0.2496 | 0.0080 | 0.5038 | 0.5263 | 0.5244 |
| V2-B | PRECOVID_DEV_A | 0.5130 | 0.5130 | 0.5719 | 0.5107 | 0.5179 | 0.2497 | 0.0038 | 0.5035 | 0.5412 | 0.5101 |
| V2-B | PRECOVID_DEV_B | 0.5053 | 0.5053 | 0.3151 | 0.5025 | 0.5061 | 0.2500 | 0.0053 | 0.5040 | 0.4844 | 0.5184 |
| **V2-B** | MEAN | 0.5092 | 0.5092 | 0.4435 | 0.5066 | 0.5120 | 0.2499 | 0.0046 | 0.5038 | 0.5128 | 0.5142 |
| V2-C | PRECOVID_DEV_A | 0.5112 | 0.5112 | 0.5277 | 0.5108 | 0.5192 | 0.2499 | 0.0147 | 0.5036 | 0.5479 | 0.5223 |
| V2-C | PRECOVID_DEV_B | 0.5088 | 0.5088 | 0.5477 | 0.5097 | 0.5086 | 0.2502 | 0.0106 | 0.5040 | 0.5361 | 0.4898 |
| **V2-C** | MEAN | 0.5100 | 0.5100 | 0.5377 | 0.5102 | 0.5139 | 0.2501 | 0.0126 | 0.5038 | 0.5420 | 0.5060 |

## K/L. Component deltas

| contribution | macro | micro | ROC-AUC | Brier | F1 |
|---|---|---|---|---|---|
| transformer_B_minus_A | -0.0058 | -0.0058 | -0.0143 | 0.0002 | -0.0237 |
| context_C_minus_B | 0.0009 | 0.0009 | 0.0019 | 0.0002 | 0.0942 |

A Brier delta is an improvement when it is negative.

## M. Per-ticker breadth and N. Signal gate

- **V2-A**: 13/41 eligible tickers beat their baseline (31.7%); median ticker accuracy 0.5144, p25 0.4979, p75 0.5303
- **V2-B**: 10/41 eligible tickers beat their baseline (24.4%); median ticker accuracy 0.5121, p25 0.4938, p75 0.5244
- **V2-C**: 13/41 eligible tickers beat their baseline (31.7%); median ticker accuracy 0.5080, p25 0.4958, p75 0.5285

**PRE-COVID SIGNAL GATE: FAIL**

- **V2-B**: mean_macro_accuracy FAIL (value 0.5092, required >= 0.5400); mean_roc_auc FAIL (value 0.5120, required >= 0.5300); beats_baseline_both_years PASS (value True, required >= True); roc_auc_above_0.50_both_years PASS (value True, required >= 0.5000); min_auc_per_fold PASS (value 0.5000, required >= 0.5000); ticker_fraction_beating_baseline FAIL (value 0.2439, required >= 0.5500)
- **V2-C**: mean_macro_accuracy FAIL (value 0.5100, required >= 0.5400); mean_roc_auc FAIL (value 0.5139, required >= 0.5300); beats_baseline_both_years PASS (value True, required >= True); roc_auc_above_0.50_both_years PASS (value True, required >= 0.5000); min_auc_per_fold PASS (value 0.5000, required >= 0.5000); ticker_fraction_beating_baseline FAIL (value 0.3171, required >= 0.5500)

the PRE-COVID signal gate FAILED: stop successfully and do NOT open 2019

## O–Q. Continuation stages (D / E / F)

- **V2-D**: NOT EXECUTED.
- **V2-E**: NOT EXECUTED.
- **V2-F**: NOT EXECUTED.


## R/S/T. Selection, seed stability and freeze

- winning base: **V2-C** (higher mean macro-ticker accuracy across 2017 and 2018; context is NOT favoured by construction)
- frozen selection: NOT WRITTEN (the programme stopped earlier)

## U–X. 2019 LOCKBOX

**NOT OPENED.** the PRE-COVID signal gate failed, so the staged programme stopped before winner selection and the architecture lockbox was never opened

## Y. Data-access audit

| field | value |
|---|---|
| store_last_feature_date | 2019-12-31 |
| store_last_target_date | 2019-12-31 |
| max_feature_date_consumed | 2018-12-28 |
| max_origin_date_consumed | 2018-12-28 |
| max_target_date_consumed | 2018-12-31 |
| rows_from_2020_plus_present_in_source_files | 72900 |
| rows_from_2020_plus_loaded_into_pre_covid_store | 0 |
| rows_from_2020_plus_consumed_by_any_model | 0 |

## Z. Output paths

- report: `/home/iemiedc2026/Documents/Sanket/Research/projects/agentic-forecaster/results/v2/pre_covid/PRE_COVID_V2_REPORT.md`
- summary: `/home/iemiedc2026/Documents/Sanket/Research/projects/agentic-forecaster/results/v2/pre_covid/pre_covid_v2_summary.json`
- ledger: `/home/iemiedc2026/Documents/Sanket/Research/projects/agentic-forecaster/results/v2/pre_covid/experiment_ledger.csv`
- universe: `/home/iemiedc2026/Documents/Sanket/Research/projects/agentic-forecaster/results/v2/pre_covid/pre_covid_supervised_universe.csv`
- selection: `/home/iemiedc2026/Documents/Sanket/Research/projects/agentic-forecaster/results/v2/pre_covid/pre_covid_dev_selection.json`  (NOT WRITTEN -- the programme stopped before winner selection, so no selection was frozen)
- audit: `/home/iemiedc2026/Documents/Sanket/Research/projects/agentic-forecaster/results/v2/pre_covid/data_access_audit.json`
- runtime experiments: `/home/iemiedc2026/Documents/Sanket/Research/output/agentic-forecaster/v2/pre_covid`
- processed store: `/home/iemiedc2026/Documents/Sanket/Research/dataset/agentic-forecaster/processed/v2/pre_covid`

## AA. Recommended next step

Do NOT open 2019. The PRE-COVID signal gate failed, which means the available features do not support a directional model in this regime either. The next useful step is a target/objective question rather than an architecture question: measure how much signal exists at all (the logistic ceiling check above is the reference), and re-target the evaluation towards ranking / volatility metrics if directional accuracy stays near the majority baseline. The 2020+ regime-aware track remains a separate future task.

