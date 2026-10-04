# PRE-COVID MULTI-HORIZON DIRECTION FORECASTING -- REPORT

**MODEL V2 IS NOT THE ORIGINAL PAPER MODEL** (classification `NEW_EXPERIMENTAL_ARCHITECTURE`).

- track: `MULTI_HORIZON` / `PRE_COVID_MULTI_HORIZON_DIRECTION_TRACK`
- generated: `2026-10-04T08:27:46.151828+00:00`
- question: does the existing PRE-COVID price-derived dataset contain more predictable directional information at 3-, 5- or 10-trading-day horizons than at the one-day horizon?
- only what changed: **FORECAST_HORIZON**
- survivorship: `SURVIVORSHIP_BIASED_FIXED_UNIVERSE_RESEARCH_TRACK` -- available securities from the reconstructed fixed universe
- final allowed target date: **2019-12-31**
- 2022/2023 labels accessed: **NO**

A 5-day result is never called next-day accuracy: `5-trading-day directional accuracy` means the model correctly predicts whether Close[t+5] is above or below Close[t].

## 1-2. Test and lint results

| check | result |
|---|---|
| ruff (`ruff check .`) | **pass** |
| pytest (`pytest -p no:warnings -q`) | **pass** (exit 0) |

## 3. Supervised ticker count and list

- eligible securities: **42**
- frozen universe SHA256: `5b10bd8f5401542c86490efd114a4a4c4adeb68ee1e127813eb520388b426a05`
- frozen before training: **True**

ADANIENT, ADANIPORTS, APOLLOHOSP, ASIANPAINT, AXISBANK, BAJAJ-AUTO, BAJAJFINSV, BAJFINANCE, BEL, BHARTIARTL, CIPLA, DRREDDY, EICHERMOT, GRASIM, HCLTECH, HDFCBANK, HINDALCO, HINDUNILVR, ICICIBANK, INFY, ITC, JSWSTEEL, KOTAKBANK, LT, M&M, MARUTI, NTPC, ONGC, POWERGRID, RELIANCE, SBIN, SHRIRAMFIN, SUNPHARMA, TATACONSUM, TATASTEEL, TCS, TECHM, TITAN, TMPV, TRENT, ULTRACEMCO, WIPRO

Excluded (with reasons in `supervised_universe.csv`):
- `COALINDIA`: only 650 valid H=10 training samples in MH_DEV_2014 (need 1000)
- `HDFCLIFE`: only 0 valid H=10 training samples in MH_DEV_2014 (need 1000); only 157 valid validation samples in the weakest development year at H=1 (need 180); only 155 valid validation samples in the weakest development year at H=3 (need 180); only 153 valid validation samples in the weakest development year at H=5 (need 180); only 148 valid validation samples in the weakest development year at H=10 (need 180)
- `INDIGO`: only 0 valid H=10 training samples in MH_DEV_2014 (need 1000)
- `NESTLEIND`: only 916 valid H=10 training samples in MH_DEV_2014 (need 1000)
- `SBILIFE`: only 0 valid H=10 training samples in MH_DEV_2014 (need 1000)

## 4. Exact data range consumed

| field | value |
|---|---|
| source store (READ-ONLY) | `/home/iemiedc2026/Documents/Sanket/Research/dataset/agentic-forecaster/processed/v2/pre_covid` |
| source store SHA256 | `9ef053533ab05cfe27c919316503f80b4239e7b97a98acda603d98d29d97ce32` |
| store last feature date | 2019-12-31 |
| store last target date | 2019-12-31 |
| horizon target cache | `/home/iemiedc2026/Documents/Sanket/Research/dataset/agentic-forecaster/processed/v2/multi_horizon/horizon_targets` |
| target schema SHA256 | `8256cca11e82983747dbc2955ad5b45ae5ec20cf4f3d915980da73805dbf3668` |
| cache max target_end_date | 2019-12-31 |
| max feature date consumed | 2019-12-31 |
| max origin date consumed | 2018-12-28 |
| max target_end_date consumed | 2018-12-31 |

## 5-6. Regime confirmation

- 2019 remained SEALED during development: **True** (lockbox fold scored: False)
- 2019 labels consumed during development: **0**
- 2020+ observations consumed: **0**
- 2020+ rows present in the raw source files (never loaded): 72900
- development folds scored: MH_DEV_2014, MH_DEV_2015, MH_DEV_2016, MH_DEV_2017, MH_DEV_2018

## 7-14. Screen results by horizon and model

`accuracy`, `balanced accuracy`, `ROC-AUC`, `Brier` and the train-majority baseline, per development year. The horizon column states the exact semantics.

### M1 LogisticRegression

| objective | fold | n | accuracy | balanced acc | ROC-AUC | Brier | train-majority baseline | delta |
|---|---|---|---|---|---|---|---|---|
| `ABS_DIR_1D_CONTROL` (1-trading-day directional accuracy) | MH_DEV_2014 | 10206 | 0.5191 | 0.5139 | 0.5217 | 0.2496 | 0.5021 | 0.0170 |
| `ABS_DIR_1D_CONTROL` (1-trading-day directional accuracy) | MH_DEV_2015 | 10290 | 0.4985 | 0.4986 | 0.5030 | 0.2501 | 0.5036 | -0.0051 |
| `ABS_DIR_1D_CONTROL` (1-trading-day directional accuracy) | MH_DEV_2016 | 10290 | 0.5054 | 0.5032 | 0.5051 | 0.2500 | 0.5033 | 0.0022 |
| `ABS_DIR_1D_CONTROL` (1-trading-day directional accuracy) | MH_DEV_2017 | 10374 | 0.5101 | 0.5040 | 0.5144 | 0.2497 | 0.5034 | 0.0068 |
| `ABS_DIR_1D_CONTROL` (1-trading-day directional accuracy) | MH_DEV_2018 | 10290 | 0.5084 | 0.5120 | 0.5197 | 0.2500 | 0.5041 | 0.0042 |
| `ABS_DIR_3D` (3-trading-day directional accuracy) | MH_DEV_2014 | 10122 | 0.5434 | 0.5045 | 0.4972 | 0.2486 | 0.5221 | 0.0213 |
| `ABS_DIR_3D` (3-trading-day directional accuracy) | MH_DEV_2015 | 10206 | 0.5106 | 0.5018 | 0.4978 | 0.2506 | 0.5239 | -0.0133 |
| `ABS_DIR_3D` (3-trading-day directional accuracy) | MH_DEV_2016 | 10206 | 0.5172 | 0.5009 | 0.5165 | 0.2496 | 0.5228 | -0.0056 |
| `ABS_DIR_3D` (3-trading-day directional accuracy) | MH_DEV_2017 | 10290 | 0.5521 | 0.5031 | 0.5130 | 0.2477 | 0.5223 | 0.0298 |
| `ABS_DIR_3D` (3-trading-day directional accuracy) | MH_DEV_2018 | 10206 | 0.5063 | 0.5025 | 0.5424 | 0.2498 | 0.5249 | -0.0186 |
| `ABS_DIR_5D` (5-trading-day directional accuracy) | MH_DEV_2014 | 10038 | 0.5596 | 0.5046 | 0.5116 | 0.2468 | 0.5364 | 0.0231 |
| `ABS_DIR_5D` (5-trading-day directional accuracy) | MH_DEV_2015 | 10122 | 0.5163 | 0.5038 | 0.4936 | 0.2512 | 0.5382 | -0.0219 |
| `ABS_DIR_5D` (5-trading-day directional accuracy) | MH_DEV_2016 | 10122 | 0.5267 | 0.5022 | 0.5150 | 0.2493 | 0.5360 | -0.0093 |
| `ABS_DIR_5D` (5-trading-day directional accuracy) | MH_DEV_2017 | 10206 | 0.5729 | 0.4996 | 0.5091 | 0.2457 | 0.5347 | 0.0382 |
| `ABS_DIR_5D` (5-trading-day directional accuracy) | MH_DEV_2018 | 10122 | 0.5039 | 0.5035 | 0.5482 | 0.2506 | 0.5384 | -0.0345 |
| `ABS_DIR_10D` (10-trading-day directional accuracy) | MH_DEV_2014 | 9828 | 0.5885 | 0.5001 | 0.5217 | 0.2425 | 0.5569 | 0.0316 |
| `ABS_DIR_10D` (10-trading-day directional accuracy) | MH_DEV_2015 | 9912 | 0.5156 | 0.4995 | 0.4722 | 0.2533 | 0.5596 | -0.0440 |
| `ABS_DIR_10D` (10-trading-day directional accuracy) | MH_DEV_2016 | 9912 | 0.5456 | 0.5000 | 0.4944 | 0.2485 | 0.5559 | -0.0103 |
| `ABS_DIR_10D` (10-trading-day directional accuracy) | MH_DEV_2017 | 9996 | 0.6061 | 0.5004 | 0.4895 | 0.2416 | 0.5543 | 0.0518 |
| `ABS_DIR_10D` (10-trading-day directional accuracy) | MH_DEV_2018 | 9912 | 0.4942 | 0.5001 | 0.5488 | 0.2534 | 0.5593 | -0.0650 |

### M2 HistGradientBoostingClassifier

| objective | fold | n | accuracy | balanced acc | ROC-AUC | Brier | train-majority baseline | delta |
|---|---|---|---|---|---|---|---|---|
| `ABS_DIR_1D_CONTROL` (1-trading-day directional accuracy) | MH_DEV_2014 | 10206 | 0.5159 | 0.5145 | 0.5212 | 0.2505 | 0.5021 | 0.0138 |
| `ABS_DIR_1D_CONTROL` (1-trading-day directional accuracy) | MH_DEV_2015 | 10290 | 0.5072 | 0.5072 | 0.5147 | 0.2501 | 0.5036 | 0.0036 |
| `ABS_DIR_1D_CONTROL` (1-trading-day directional accuracy) | MH_DEV_2016 | 10290 | 0.5140 | 0.5132 | 0.5199 | 0.2498 | 0.5033 | 0.0107 |
| `ABS_DIR_1D_CONTROL` (1-trading-day directional accuracy) | MH_DEV_2017 | 10374 | 0.5095 | 0.5073 | 0.5089 | 0.2503 | 0.5034 | 0.0062 |
| `ABS_DIR_1D_CONTROL` (1-trading-day directional accuracy) | MH_DEV_2018 | 10290 | 0.5077 | 0.5091 | 0.5146 | 0.2506 | 0.5041 | 0.0035 |
| `ABS_DIR_3D` (3-trading-day directional accuracy) | MH_DEV_2014 | 10122 | 0.5347 | 0.5097 | 0.5202 | 0.2487 | 0.5221 | 0.0126 |
| `ABS_DIR_3D` (3-trading-day directional accuracy) | MH_DEV_2015 | 10206 | 0.5053 | 0.4988 | 0.5036 | 0.2515 | 0.5239 | -0.0186 |
| `ABS_DIR_3D` (3-trading-day directional accuracy) | MH_DEV_2016 | 10206 | 0.5191 | 0.5078 | 0.5293 | 0.2494 | 0.5228 | -0.0037 |
| `ABS_DIR_3D` (3-trading-day directional accuracy) | MH_DEV_2017 | 10290 | 0.5290 | 0.4961 | 0.5142 | 0.2482 | 0.5223 | 0.0067 |
| `ABS_DIR_3D` (3-trading-day directional accuracy) | MH_DEV_2018 | 10206 | 0.5135 | 0.5109 | 0.5268 | 0.2505 | 0.5249 | -0.0114 |
| `ABS_DIR_5D` (5-trading-day directional accuracy) | MH_DEV_2014 | 10038 | 0.5493 | 0.5091 | 0.5270 | 0.2473 | 0.5364 | 0.0129 |
| `ABS_DIR_5D` (5-trading-day directional accuracy) | MH_DEV_2015 | 10122 | 0.5116 | 0.5013 | 0.5035 | 0.2521 | 0.5382 | -0.0266 |
| `ABS_DIR_5D` (5-trading-day directional accuracy) | MH_DEV_2016 | 10122 | 0.5229 | 0.5039 | 0.5205 | 0.2498 | 0.5360 | -0.0131 |
| `ABS_DIR_5D` (5-trading-day directional accuracy) | MH_DEV_2017 | 10206 | 0.5572 | 0.5019 | 0.5079 | 0.2464 | 0.5347 | 0.0225 |
| `ABS_DIR_5D` (5-trading-day directional accuracy) | MH_DEV_2018 | 10122 | 0.5055 | 0.5052 | 0.5259 | 0.2518 | 0.5384 | -0.0328 |
| `ABS_DIR_10D` (10-trading-day directional accuracy) | MH_DEV_2014 | 9828 | 0.5796 | 0.5089 | 0.5332 | 0.2430 | 0.5569 | 0.0227 |
| `ABS_DIR_10D` (10-trading-day directional accuracy) | MH_DEV_2015 | 9912 | 0.5135 | 0.4993 | 0.5037 | 0.2541 | 0.5596 | -0.0461 |
| `ABS_DIR_10D` (10-trading-day directional accuracy) | MH_DEV_2016 | 9912 | 0.5375 | 0.4981 | 0.5144 | 0.2486 | 0.5559 | -0.0184 |
| `ABS_DIR_10D` (10-trading-day directional accuracy) | MH_DEV_2017 | 9996 | 0.5764 | 0.4908 | 0.5072 | 0.2433 | 0.5543 | 0.0221 |
| `ABS_DIR_10D` (10-trading-day directional accuracy) | MH_DEV_2018 | 9912 | 0.4956 | 0.5009 | 0.5069 | 0.2554 | 0.5593 | -0.0637 |

## 15-16. Common-origin and non-overlapping comparisons

| model | objective | view | mean accuracy | mean balanced acc | mean ROC-AUC | mean Brier | mean delta | worst-year AUC | years AUC>0.50 | years beating baseline |
|---|---|---|---|---|---|---|---|---|---|---|
| LOGISTIC | `ABS_DIR_1D_CONTROL` | natural | 0.5083 | 0.5063 | 0.5128 | 0.2499 | 0.0050 | 0.5030 | 5 | 4 |
| LOGISTIC | `ABS_DIR_1D_CONTROL` | common_origin | 0.5076 | 0.5063 | 0.5133 | 0.2499 | 0.0050 | 0.5057 | 5 | 4 |
| LOGISTIC | `ABS_DIR_1D_CONTROL` | non_overlapping | 0.5083 | 0.5063 | 0.5128 | 0.2499 | 0.0050 | 0.5030 | 5 | 4 |
| LOGISTIC | `ABS_DIR_3D` | natural | 0.5259 | 0.5026 | 0.5134 | 0.2493 | 0.0027 | 0.4972 | 3 | 2 |
| LOGISTIC | `ABS_DIR_3D` | common_origin | 0.5240 | 0.5026 | 0.5142 | 0.2493 | 0.0027 | 0.4961 | 3 | 2 |
| LOGISTIC | `ABS_DIR_3D` | non_overlapping | 0.5205 | 0.5026 | 0.5115 | 0.2493 | 0.0027 | 0.4934 | 4 | 2 |
| LOGISTIC | `ABS_DIR_5D` | natural | 0.5359 | 0.5027 | 0.5155 | 0.2487 | -0.0009 | 0.4936 | 4 | 2 |
| LOGISTIC | `ABS_DIR_5D` | common_origin | 0.5348 | 0.5027 | 0.5159 | 0.2487 | -0.0009 | 0.4943 | 4 | 2 |
| LOGISTIC | `ABS_DIR_5D` | non_overlapping | 0.5368 | 0.5027 | 0.5232 | 0.2487 | -0.0009 | 0.4894 | 4 | 2 |
| LOGISTIC | `ABS_DIR_10D` | natural | 0.5500 | 0.5000 | 0.5053 | 0.2479 | -0.0072 | 0.4722 | 2 | 2 |
| LOGISTIC | `ABS_DIR_10D` | common_origin | 0.5500 | 0.5000 | 0.5053 | 0.2479 | -0.0072 | 0.4722 | 2 | 2 |
| LOGISTIC | `ABS_DIR_10D` | non_overlapping | 0.5498 | 0.5000 | 0.4984 | 0.2479 | -0.0072 | 0.4628 | 2 | 2 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_1D_CONTROL` | natural | 0.5109 | 0.5103 | 0.5159 | 0.2503 | 0.0076 | 0.5089 | 5 | 5 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_1D_CONTROL` | common_origin | 0.5101 | 0.5103 | 0.5156 | 0.2503 | 0.0076 | 0.5085 | 5 | 5 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_1D_CONTROL` | non_overlapping | 0.5109 | 0.5103 | 0.5159 | 0.2503 | 0.0076 | 0.5089 | 5 | 5 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_3D` | natural | 0.5203 | 0.5047 | 0.5188 | 0.2497 | -0.0029 | 0.5036 | 5 | 2 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_3D` | common_origin | 0.5191 | 0.5047 | 0.5196 | 0.2497 | -0.0029 | 0.5038 | 5 | 2 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_3D` | non_overlapping | 0.5140 | 0.5047 | 0.5111 | 0.2497 | -0.0029 | 0.4994 | 4 | 2 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_5D` | natural | 0.5293 | 0.5043 | 0.5170 | 0.2495 | -0.0074 | 0.5035 | 5 | 2 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_5D` | common_origin | 0.5285 | 0.5043 | 0.5171 | 0.2495 | -0.0074 | 0.5032 | 5 | 2 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_5D` | non_overlapping | 0.5332 | 0.5043 | 0.5259 | 0.2495 | -0.0074 | 0.4990 | 4 | 2 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_10D` | natural | 0.5405 | 0.4996 | 0.5131 | 0.2489 | -0.0167 | 0.5037 | 5 | 2 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_10D` | common_origin | 0.5405 | 0.4996 | 0.5131 | 0.2489 | -0.0167 | 0.5037 | 5 | 2 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_10D` | non_overlapping | 0.5377 | 0.4996 | 0.5050 | 0.2489 | -0.0167 | 0.4895 | 1 | 2 |

## 17. Confidence intervals (date-block bootstrap)

| model | objective | fold | accuracy 95% CI | ROC-AUC 95% CI | method |
|---|---|---|---|---|---|
| LOGISTIC | `ABS_DIR_1D_CONTROL` | MH_DEV_2014 | [0.5019, 0.5369] | [0.5046, 0.5385] | moving block, 21 dates |
| LOGISTIC | `ABS_DIR_1D_CONTROL` | MH_DEV_2015 | [0.4817, 0.5114] | [0.4887, 0.5173] | moving block, 21 dates |
| LOGISTIC | `ABS_DIR_1D_CONTROL` | MH_DEV_2016 | [0.4908, 0.5310] | [0.4879, 0.5291] | moving block, 21 dates |
| LOGISTIC | `ABS_DIR_1D_CONTROL` | MH_DEV_2017 | [0.4925, 0.5208] | [0.5034, 0.5244] | moving block, 21 dates |
| LOGISTIC | `ABS_DIR_1D_CONTROL` | MH_DEV_2018 | [0.4866, 0.5197] | [0.5034, 0.5306] | moving block, 21 dates |
| LOGISTIC | `ABS_DIR_3D` | MH_DEV_2014 | [0.5101, 0.5872] | [0.4809, 0.5207] | moving block, 21 dates |
| LOGISTIC | `ABS_DIR_3D` | MH_DEV_2015 | [0.4679, 0.5366] | [0.4644, 0.5244] | moving block, 21 dates |
| LOGISTIC | `ABS_DIR_3D` | MH_DEV_2016 | [0.4855, 0.5698] | [0.5077, 0.5438] | moving block, 21 dates |
| LOGISTIC | `ABS_DIR_3D` | MH_DEV_2017 | [0.5146, 0.5719] | [0.4975, 0.5385] | moving block, 21 dates |
| LOGISTIC | `ABS_DIR_3D` | MH_DEV_2018 | [0.4557, 0.5423] | [0.5246, 0.5613] | moving block, 21 dates |
| LOGISTIC | `ABS_DIR_5D` | MH_DEV_2014 | [0.5176, 0.6170] | [0.4953, 0.5421] | moving block, 21 dates |
| LOGISTIC | `ABS_DIR_5D` | MH_DEV_2015 | [0.4553, 0.5484] | [0.4517, 0.5330] | moving block, 21 dates |
| LOGISTIC | `ABS_DIR_5D` | MH_DEV_2016 | [0.4892, 0.5973] | [0.5041, 0.5485] | moving block, 21 dates |
| LOGISTIC | `ABS_DIR_5D` | MH_DEV_2017 | [0.5273, 0.5987] | [0.4897, 0.5464] | moving block, 21 dates |
| LOGISTIC | `ABS_DIR_5D` | MH_DEV_2018 | [0.4392, 0.5530] | [0.5200, 0.5792] | moving block, 21 dates |
| LOGISTIC | `ABS_DIR_10D` | MH_DEV_2014 | [0.5295, 0.6742] | [0.4899, 0.5512] | moving block, 21 dates |
| LOGISTIC | `ABS_DIR_10D` | MH_DEV_2015 | [0.4325, 0.5702] | [0.4258, 0.5092] | moving block, 21 dates |
| LOGISTIC | `ABS_DIR_10D` | MH_DEV_2016 | [0.4909, 0.6502] | [0.4779, 0.5280] | moving block, 21 dates |
| LOGISTIC | `ABS_DIR_10D` | MH_DEV_2017 | [0.5465, 0.6404] | [0.4476, 0.5370] | moving block, 21 dates |
| LOGISTIC | `ABS_DIR_10D` | MH_DEV_2018 | [0.4063, 0.5716] | [0.5252, 0.5844] | moving block, 21 dates |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_1D_CONTROL` | MH_DEV_2014 | [0.5026, 0.5301] | [0.5050, 0.5385] | moving block, 21 dates |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_1D_CONTROL` | MH_DEV_2015 | [0.4927, 0.5154] | [0.4997, 0.5254] | moving block, 21 dates |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_1D_CONTROL` | MH_DEV_2016 | [0.5036, 0.5278] | [0.5070, 0.5351] | moving block, 21 dates |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_1D_CONTROL` | MH_DEV_2017 | [0.4966, 0.5178] | [0.4975, 0.5185] | moving block, 21 dates |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_1D_CONTROL` | MH_DEV_2018 | [0.4952, 0.5168] | [0.5021, 0.5228] | moving block, 21 dates |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_3D` | MH_DEV_2014 | [0.5126, 0.5661] | [0.5093, 0.5388] | moving block, 21 dates |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_3D` | MH_DEV_2015 | [0.4727, 0.5229] | [0.4862, 0.5152] | moving block, 21 dates |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_3D` | MH_DEV_2016 | [0.4973, 0.5570] | [0.5193, 0.5468] | moving block, 21 dates |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_3D` | MH_DEV_2017 | [0.5049, 0.5419] | [0.5032, 0.5287] | moving block, 21 dates |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_3D` | MH_DEV_2018 | [0.4746, 0.5374] | [0.5104, 0.5436] | moving block, 21 dates |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_5D` | MH_DEV_2014 | [0.5175, 0.5905] | [0.5092, 0.5520] | moving block, 21 dates |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_5D` | MH_DEV_2015 | [0.4591, 0.5409] | [0.4775, 0.5210] | moving block, 21 dates |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_5D` | MH_DEV_2016 | [0.4911, 0.5795] | [0.5072, 0.5433] | moving block, 21 dates |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_5D` | MH_DEV_2017 | [0.5236, 0.5771] | [0.4945, 0.5280] | moving block, 21 dates |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_5D` | MH_DEV_2018 | [0.4495, 0.5452] | [0.5014, 0.5500] | moving block, 21 dates |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_10D` | MH_DEV_2014 | [0.5364, 0.6476] | [0.5185, 0.5582] | moving block, 21 dates |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_10D` | MH_DEV_2015 | [0.4408, 0.5642] | [0.4692, 0.5305] | moving block, 21 dates |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_10D` | MH_DEV_2016 | [0.4888, 0.6289] | [0.4993, 0.5517] | moving block, 21 dates |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_10D` | MH_DEV_2017 | [0.5309, 0.6005] | [0.4791, 0.5349] | moving block, 21 dates |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_10D` | MH_DEV_2018 | [0.4140, 0.5659] | [0.4899, 0.5346] | moving block, 21 dates |

## 18. Horizon ranking, 19. SCREEN-PASS, 20. STRONGLY PROMISING

_No candidate horizon reached the screening gate, so no ranking is produced._

- SCREEN-PASS horizons: NONE
- STRONGLY PROMISING horizons: NONE
- horizons selected for the neural stage: NONE (hard stop)

- LOGISTIC `ABS_DIR_1D_CONTROL`: **SCREEN_FAIL** -- failed: mean_roc_auc, mean_balanced_accuracy
- LOGISTIC `ABS_DIR_3D`: **SCREEN_FAIL** -- failed: mean_roc_auc, mean_balanced_accuracy, mean_accuracy_delta_vs_train_majority, years_roc_auc_above_50
- LOGISTIC `ABS_DIR_5D`: **SCREEN_FAIL** -- failed: mean_roc_auc, mean_balanced_accuracy, mean_accuracy_delta_vs_train_majority, common_origin_directionally_positive
- LOGISTIC `ABS_DIR_10D`: **SCREEN_FAIL** -- failed: mean_roc_auc, mean_balanced_accuracy, mean_accuracy_delta_vs_train_majority, years_roc_auc_above_50, common_origin_directionally_positive, non_overlap_not_chance
- HIST_GRADIENT_BOOSTING `ABS_DIR_1D_CONTROL`: **SCREEN_FAIL** -- failed: mean_roc_auc, mean_balanced_accuracy
- HIST_GRADIENT_BOOSTING `ABS_DIR_3D`: **SCREEN_FAIL** -- failed: mean_roc_auc, mean_balanced_accuracy, mean_accuracy_delta_vs_train_majority, common_origin_directionally_positive
- HIST_GRADIENT_BOOSTING `ABS_DIR_5D`: **SCREEN_FAIL** -- failed: mean_roc_auc, mean_balanced_accuracy, mean_accuracy_delta_vs_train_majority, common_origin_directionally_positive
- HIST_GRADIENT_BOOSTING `ABS_DIR_10D`: **SCREEN_FAIL** -- failed: mean_roc_auc, mean_balanced_accuracy, mean_accuracy_delta_vs_train_majority, common_origin_directionally_positive

## 21-25. Neural results and Transformer delta

_No neural fit was run: the screening gate did not pass any horizon, so section 20 requires a successful stop before any LSTM is trained._

## 26-27. Neural gate, selected model, 28. seed stability

- horizon/model frozen before 2019: **NO**

_Seed stability was not reached._

## 29-32. 2019 lockbox

**NOT OPENED.** The staged programme stopped before the lockbox, so no 2019 label was consumed at any point.

## 33. Horizon return magnitude diagnostic (analysis only)

| objective | median abs future return | mean abs future return | std of future return | class balance (up) |
|---|---|---|---|---|
| `ABS_DIR_1D_CONTROL` | 0.00987 | 0.01337 | 0.01879 | 0.5056 |
| `ABS_DIR_3D` | 0.01767 | 0.02370 | 0.03261 | 0.5255 |
| `ABS_DIR_5D` | 0.02336 | 0.03089 | 0.04199 | 0.5344 |
| `ABS_DIR_10D` | 0.03426 | 0.04390 | 0.05837 | 0.5503 |

Samples are NEVER filtered on the size of the future move.

## 34. Model confidence vs future move magnitude (analysis only)

| model | objective | fold | n | Pearson | Spearman |
|---|---|---|---|---|---|
| LOGISTIC | `ABS_DIR_1D_CONTROL` | MH_DEV_2014 | 10206 | 0.0841 | 0.0655 |
| LOGISTIC | `ABS_DIR_1D_CONTROL` | MH_DEV_2015 | 10290 | 0.0937 | 0.0378 |
| LOGISTIC | `ABS_DIR_1D_CONTROL` | MH_DEV_2016 | 10290 | 0.0463 | 0.0261 |
| LOGISTIC | `ABS_DIR_1D_CONTROL` | MH_DEV_2017 | 10374 | 0.0863 | 0.0632 |
| LOGISTIC | `ABS_DIR_1D_CONTROL` | MH_DEV_2018 | 10290 | 0.0927 | 0.0544 |
| LOGISTIC | `ABS_DIR_3D` | MH_DEV_2014 | 10122 | -0.0247 | -0.0283 |
| LOGISTIC | `ABS_DIR_3D` | MH_DEV_2015 | 10206 | 0.0372 | -0.0045 |
| LOGISTIC | `ABS_DIR_3D` | MH_DEV_2016 | 10206 | -0.0701 | -0.0604 |
| LOGISTIC | `ABS_DIR_3D` | MH_DEV_2017 | 10290 | -0.0072 | -0.0161 |
| LOGISTIC | `ABS_DIR_3D` | MH_DEV_2018 | 10206 | -0.0073 | -0.0302 |
| LOGISTIC | `ABS_DIR_5D` | MH_DEV_2014 | 10038 | -0.0289 | -0.0316 |
| LOGISTIC | `ABS_DIR_5D` | MH_DEV_2015 | 10122 | 0.0175 | -0.0077 |
| LOGISTIC | `ABS_DIR_5D` | MH_DEV_2016 | 10122 | -0.0475 | -0.0275 |
| LOGISTIC | `ABS_DIR_5D` | MH_DEV_2017 | 10206 | -0.0060 | 0.0051 |
| LOGISTIC | `ABS_DIR_5D` | MH_DEV_2018 | 10122 | -0.0339 | -0.0326 |
| LOGISTIC | `ABS_DIR_10D` | MH_DEV_2014 | 9828 | -0.0090 | 0.0006 |
| LOGISTIC | `ABS_DIR_10D` | MH_DEV_2015 | 9912 | 0.0190 | 0.0093 |
| LOGISTIC | `ABS_DIR_10D` | MH_DEV_2016 | 9912 | -0.0433 | -0.0431 |
| LOGISTIC | `ABS_DIR_10D` | MH_DEV_2017 | 9996 | -0.0208 | -0.0126 |
| LOGISTIC | `ABS_DIR_10D` | MH_DEV_2018 | 9912 | -0.0629 | -0.0213 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_1D_CONTROL` | MH_DEV_2014 | 10206 | 0.0760 | 0.0389 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_1D_CONTROL` | MH_DEV_2015 | 10290 | 0.0363 | 0.0152 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_1D_CONTROL` | MH_DEV_2016 | 10290 | 0.0481 | 0.0072 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_1D_CONTROL` | MH_DEV_2017 | 10374 | 0.0219 | -0.0001 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_1D_CONTROL` | MH_DEV_2018 | 10290 | 0.0608 | 0.0231 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_3D` | MH_DEV_2014 | 10122 | 0.0120 | 0.0009 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_3D` | MH_DEV_2015 | 10206 | 0.0216 | 0.0122 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_3D` | MH_DEV_2016 | 10206 | 0.0246 | 0.0192 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_3D` | MH_DEV_2017 | 10290 | -0.0064 | -0.0223 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_3D` | MH_DEV_2018 | 10206 | -0.0050 | -0.0194 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_5D` | MH_DEV_2014 | 10038 | 0.0156 | 0.0093 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_5D` | MH_DEV_2015 | 10122 | 0.0283 | 0.0147 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_5D` | MH_DEV_2016 | 10122 | 0.0400 | 0.0194 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_5D` | MH_DEV_2017 | 10206 | 0.0117 | 0.0062 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_5D` | MH_DEV_2018 | 10122 | 0.0030 | -0.0073 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_10D` | MH_DEV_2014 | 9828 | 0.0567 | 0.0452 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_10D` | MH_DEV_2015 | 9912 | 0.0524 | 0.0444 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_10D` | MH_DEV_2016 | 9912 | 0.0469 | 0.0280 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_10D` | MH_DEV_2017 | 9996 | 0.0429 | 0.0422 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_10D` | MH_DEV_2018 | 9912 | 0.0072 | 0.0082 |

This never selects or filters a prediction; it only asks whether the model naturally becomes more confident on larger subsequent moves.

## 35-36. Regime audit

- maximum consumed target date: **2018-12-31**
- 2020+ rows consumed: **0** (MUST BE ZERO)
- 2020+ rows in the raw source files, never loaded: 72900
- rejection happens at access time: `PostCovidDataAccessError` / `MultiHorizonLockboxError`

Full audit: `results/v2/multi_horizon/data_access_audit.json`.

## 37. Recommended next action

**ADD_EXOGENOUS_INFORMATION**

This action is REPORTED, never executed automatically.

## Interpretation limits

- classification accuracy is the primary metric in this phase; overlapping multi-day forecasts may represent overlapping holding periods, so they are NOT compounded into portfolio returns without a separate portfolio simulator
- the universe is a fixed, later-reconstructed constituent list projected backwards
- 1D is a CONTROL and is never selected as a horizon

MULTI_HORIZON_SIGNAL: NONE
