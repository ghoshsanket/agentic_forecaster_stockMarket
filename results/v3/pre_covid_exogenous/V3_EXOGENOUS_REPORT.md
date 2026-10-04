# V3 PRE-COVID EXOGENOUS MARKET-INFORMATION REPORT

**MODEL V2/V3 ARE NOT THE ORIGINAL PAPER MODEL** (classification `NEW_EXPERIMENTAL_ARCHITECTURE`).

- track: `V3_EXOGENOUS_PRECOVID`
- generated: `2026-10-04T12:14:43.961033+00:00`
- question: does point-in-time market, volatility, global-risk, currency and commodity information add genuine directional signal beyond the stock's own price-derived features?
- only what changed: **INFORMATION_SET** (the information set, not the architecture)
- survivorship: `SURVIVORSHIP_BIASED_FIXED_UNIVERSE_RESEARCH_TRACK`
- final allowed date: **2019-12-31**

A multi-day result is never called next-day accuracy: `5-trading-day directional accuracy` means the model predicts whether Close[t+5] is above or below Close[t].

## 1-2. Test and lint

| check | result |
|---|---|
| ruff (`ruff check .`) | **pass** |
| pytest | **pass** (exit 0) |

## 3. Source probe table

- declared: **27** | accepted: **18** | rejected: **9**

| source_id | provider | identifier | available | first_actual_date | last_actual_date | row_count | missing_fraction | availability_class | accepted | exclusion_reason |
|---|---|---|---|---|---|---|---|---|---|---|
| NIFTY50 | yfinance | ^NSEI | True | 2007-09-17 | 2019-12-31 | 3001 | 0.0323 | INDIA_SAME_CLOSE | True |  |
| NIFTY_BANK | yfinance | ^NSEBANK | True | 2007-09-17 | 2019-12-31 | 3017 | 0.0242 | INDIA_SAME_CLOSE | True |  |
| INDIA_VIX | yfinance | ^INDIAVIX | True | 2008-03-03 | 2019-12-31 | 2899 | 0.0242 | INDIA_SAME_CLOSE | True |  |
| NIFTY_IT | yfinance | ^CNXIT | True | 2007-09-17 | 2019-12-31 | 3017 | 0.0242 | INDIA_SAME_CLOSE | True |  |
| NIFTY_AUTO | yfinance | ^CNXAUTO | False | n/a | n/a | 0 | n/a | INDIA_SAME_CLOSE | False | provider returned no rows |
| NIFTY_FMCG | yfinance | ^CNXFMCG | False | n/a | n/a | 0 | n/a | INDIA_SAME_CLOSE | False | provider returned no rows |
| NIFTY_PHARMA | yfinance | ^CNXPHARMA | True | 2011-01-31 | 2019-12-31 | 2190 | 0.0242 | INDIA_SAME_CLOSE | True |  |
| NIFTY_METAL | yfinance | ^CNXMETAL | False | n/a | n/a | 0 | n/a | INDIA_SAME_CLOSE | False | provider returned no rows |
| NIFTY_ENERGY | yfinance | ^CNXENERGY | False | n/a | n/a | 0 | n/a | INDIA_SAME_CLOSE | False | provider returned no rows |
| NIFTY_FIN_SERVICE | yfinance | ^NIFTY_FIN_SERVICE | False | n/a | n/a | 0 | n/a | INDIA_SAME_CLOSE | False | provider returned no rows |
| NIFTY_REALTY | yfinance | ^CNXREALTY | False | n/a | n/a | 0 | n/a | INDIA_SAME_CLOSE | False | provider returned no rows |
| NIFTY_PSU_BANK | yfinance | ^CNXPSUBANK | False | n/a | n/a | 0 | n/a | INDIA_SAME_CLOSE | False | provider returned no rows |
| SP500 | yfinance | ^GSPC | True | 2005-01-03 | 2019-12-31 | 3775 | 0.0040 | EXTERNAL_CONSERVATIVE_LAG1 | True |  |
| NASDAQ_COMPOSITE | yfinance | ^IXIC | True | 2005-01-03 | 2019-12-31 | 3775 | 0.0040 | EXTERNAL_CONSERVATIVE_LAG1 | True |  |
| DOW_JONES | yfinance | ^DJI | True | 2005-01-03 | 2019-12-31 | 3775 | 0.0040 | EXTERNAL_CONSERVATIVE_LAG1 | True |  |
| US_VIX | yfinance | ^VIX | True | 2005-01-03 | 2019-12-31 | 3775 | 0.0040 | EXTERNAL_CONSERVATIVE_LAG1 | True |  |
| NIKKEI_225 | yfinance | ^N225 | True | 2005-01-04 | 2019-12-30 | 3673 | 0.0121 | EXTERNAL_CONSERVATIVE_LAG1 | True |  |
| HANG_SENG | yfinance | ^HSI | True | 2005-01-03 | 2019-12-31 | 3696 | 0.0081 | EXTERNAL_CONSERVATIVE_LAG1 | True |  |
| EURO_STOXX_50 | yfinance | ^STOXX50E | True | 2007-03-30 | 2019-12-30 | 3191 | 0.0276 | EXTERNAL_CONSERVATIVE_LAG1 | True |  |
| USDINR | yfinance | INR=X | True | 2005-01-03 | 2019-12-31 | 3882 | 0.0115 | EXTERNAL_CONSERVATIVE_LAG1 | True |  |
| BRENT_CRUDE | yfinance | BZ=F | True | 2007-07-30 | 2019-12-31 | 3074 | 0.0079 | EXTERNAL_CONSERVATIVE_LAG1 | True |  |
| WTI_CRUDE | yfinance | CL=F | True | 2005-01-03 | 2019-12-31 | 3772 | 0.0079 | EXTERNAL_CONSERVATIVE_LAG1 | True |  |
| GOLD | yfinance | GC=F | True | 2005-01-03 | 2019-12-31 | 3768 | 0.0079 | EXTERNAL_CONSERVATIVE_LAG1 | True |  |
| US_DOLLAR_INDEX | yfinance | DX-Y.NYB | True | 2005-01-03 | 2019-12-31 | 3779 | 0.0040 | EXTERNAL_CONSERVATIVE_LAG1 | True |  |
| US_10Y_YIELD | yfinance | ^TNX | True | 2005-01-03 | 2019-12-31 | 3770 | 0.0040 | EXTERNAL_CONSERVATIVE_LAG1 | True |  |
| FII_NET_FLOW | none_available | NOT_PROVIDED | False | n/a | n/a | 0 | n/a | EXTERNAL_CONSERVATIVE_LAG1 | False | unavailable |
| DII_NET_FLOW | none_available | NOT_PROVIDED | False | n/a | n/a | 0 | n/a | EXTERNAL_CONSERVATIVE_LAG1 | False | unavailable |

## 4-5. Accepted sources, and 6. their exact lag rule

| source_id | identifier | availability_class | final_lag_rule | SHA256 |
|---|---|---|---|---|
| NIFTY50 | ^NSEI | INDIA_SAME_CLOSE | same-session close (INDIA_SAME_CLOSE) | n/a |
| NIFTY_BANK | ^NSEBANK | INDIA_SAME_CLOSE | same-session close (INDIA_SAME_CLOSE) | n/a |
| INDIA_VIX | ^INDIAVIX | INDIA_SAME_CLOSE | same-session close (INDIA_SAME_CLOSE) | n/a |
| NIFTY_IT | ^CNXIT | INDIA_SAME_CLOSE | same-session close (INDIA_SAME_CLOSE) | n/a |
| NIFTY_PHARMA | ^CNXPHARMA | INDIA_SAME_CLOSE | same-session close (INDIA_SAME_CLOSE) | n/a |
| SP500 | ^GSPC | EXTERNAL_CONSERVATIVE_LAG1 | conservative lag1: most recent source observation strictly before the NSE prediction timestamp | n/a |
| NASDAQ_COMPOSITE | ^IXIC | EXTERNAL_CONSERVATIVE_LAG1 | conservative lag1: most recent source observation strictly before the NSE prediction timestamp | n/a |
| DOW_JONES | ^DJI | EXTERNAL_CONSERVATIVE_LAG1 | conservative lag1: most recent source observation strictly before the NSE prediction timestamp | n/a |
| US_VIX | ^VIX | EXTERNAL_CONSERVATIVE_LAG1 | conservative lag1: most recent source observation strictly before the NSE prediction timestamp | n/a |
| NIKKEI_225 | ^N225 | EXTERNAL_CONSERVATIVE_LAG1 | conservative lag1: most recent source observation strictly before the NSE prediction timestamp | n/a |
| HANG_SENG | ^HSI | EXTERNAL_CONSERVATIVE_LAG1 | conservative lag1: most recent source observation strictly before the NSE prediction timestamp | n/a |
| EURO_STOXX_50 | ^STOXX50E | EXTERNAL_CONSERVATIVE_LAG1 | conservative lag1: most recent source observation strictly before the NSE prediction timestamp | n/a |
| USDINR | INR=X | EXTERNAL_CONSERVATIVE_LAG1 | conservative lag1: most recent source observation strictly before the NSE prediction timestamp | n/a |
| BRENT_CRUDE | BZ=F | EXTERNAL_CONSERVATIVE_LAG1 | conservative lag1: most recent source observation strictly before the NSE prediction timestamp | n/a |
| WTI_CRUDE | CL=F | EXTERNAL_CONSERVATIVE_LAG1 | conservative lag1: most recent source observation strictly before the NSE prediction timestamp | n/a |
| GOLD | GC=F | EXTERNAL_CONSERVATIVE_LAG1 | conservative lag1: most recent source observation strictly before the NSE prediction timestamp | n/a |
| US_DOLLAR_INDEX | DX-Y.NYB | EXTERNAL_CONSERVATIVE_LAG1 | conservative lag1: most recent source observation strictly before the NSE prediction timestamp | n/a |
| US_10Y_YIELD | ^TNX | EXTERNAL_CONSERVATIVE_LAG1 | conservative lag1: most recent source observation strictly before the NSE prediction timestamp | n/a |

## 7. Source hashes

- source manifest SHA256: `2efeed90db8eb9a45aaf2df2645cb98ccb3a359d6064cac757636e77a02e7e51`
- processed exogenous store SHA256: `7181300519bc20613b75b1957704267ff1a7ace002911dfe32d410ce48afc6bd`

## 8-10. Regime confirmation

- maximum development date: **2019-12-31**
- 2019 sealed during selection: **True**
- 2020+ stock rows consumed: **0**
- 2020+ exogenous rows consumed: **0**
- max target_end_date consumed: **2018-12-31**

## 11-15. Screen results by information family and horizon

### X0_STOCK_ONLY

| model | objective | mean AUC | mean bal. acc | mean accuracy | mean Brier | train-majority | delta | years AUC>0.50 | ticker breadth | common-sample AUC | incremental AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| HIST_GRADIENT_BOOSTING | `ABS_DIR_1D_CONTROL` | 0.5159 | 0.5103 | 0.5109 | 0.2503 | n/a | 0.0076 | 5 | n/a | n/a | n/a |
| LOGISTIC | `ABS_DIR_1D_CONTROL` | 0.5128 | 0.5063 | 0.5083 | 0.2499 | n/a | 0.0050 | 5 | n/a | n/a | n/a |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_3D` | 0.5188 | 0.5047 | 0.5203 | 0.2497 | n/a | -0.0029 | 5 | n/a | n/a | n/a |
| LOGISTIC | `ABS_DIR_3D` | 0.5134 | 0.5026 | 0.5259 | 0.2493 | n/a | 0.0027 | 3 | n/a | n/a | n/a |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_5D` | 0.5170 | 0.5043 | 0.5293 | 0.2495 | n/a | -0.0074 | 5 | n/a | n/a | n/a |
| LOGISTIC | `ABS_DIR_5D` | 0.5155 | 0.5027 | 0.5359 | 0.2487 | n/a | -0.0009 | 4 | n/a | n/a | n/a |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_10D` | 0.5131 | 0.4996 | 0.5405 | 0.2489 | n/a | -0.0167 | 5 | n/a | n/a | n/a |
| LOGISTIC | `ABS_DIR_10D` | 0.5053 | 0.5000 | 0.5500 | 0.2479 | n/a | -0.0072 | 2 | n/a | n/a | n/a |

### X1_INDIA_MARKET

| model | objective | mean AUC | mean bal. acc | mean accuracy | mean Brier | train-majority | delta | years AUC>0.50 | ticker breadth | common-sample AUC | incremental AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| HIST_GRADIENT_BOOSTING | `ABS_DIR_1D_CONTROL` | 0.5070 | 0.5017 | 0.4997 | 0.2539 | n/a | -0.0022 | 3 | n/a | 0.5070 | -0.0060 |
| LOGISTIC | `ABS_DIR_1D_CONTROL` | 0.5271 | 0.5205 | 0.5192 | 0.2498 | n/a | 0.0173 | 5 | n/a | 0.5271 | 0.0158 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_3D` | 0.5282 | 0.5194 | 0.5160 | 0.2552 | n/a | -0.0025 | 5 | n/a | 0.5282 | 0.0068 |
| LOGISTIC | `ABS_DIR_3D` | 0.5410 | 0.5204 | 0.5240 | 0.2492 | n/a | 0.0055 | 5 | n/a | 0.5410 | 0.0236 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_5D` | 0.5267 | 0.5145 | 0.5125 | 0.2553 | n/a | -0.0175 | 5 | n/a | 0.5267 | 0.0094 |
| LOGISTIC | `ABS_DIR_5D` | 0.5455 | 0.5223 | 0.5317 | 0.2489 | n/a | 0.0017 | 5 | n/a | 0.5455 | 0.0298 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_10D` | 0.5179 | 0.5055 | 0.5004 | 0.2649 | n/a | -0.0456 | 4 | n/a | 0.5179 | -0.0007 |
| LOGISTIC | `ABS_DIR_10D` | 0.5472 | 0.5209 | 0.5422 | 0.2482 | n/a | -0.0039 | 5 | n/a | 0.5472 | 0.0370 |

### X2_GLOBAL_RISK

| model | objective | mean AUC | mean bal. acc | mean accuracy | mean Brier | train-majority | delta | years AUC>0.50 | ticker breadth | common-sample AUC | incremental AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| HIST_GRADIENT_BOOSTING | `ABS_DIR_1D_CONTROL` | 0.5026 | 0.5033 | 0.5029 | 0.2544 | n/a | 0.0012 | 2 | n/a | 0.5042 | -0.0088 |
| LOGISTIC | `ABS_DIR_1D_CONTROL` | 0.5016 | 0.5036 | 0.5017 | 0.2530 | n/a | -0.0001 | 2 | n/a | 0.5018 | -0.0096 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_3D` | 0.4947 | 0.4953 | 0.4995 | 0.2583 | n/a | -0.0174 | 2 | n/a | 0.5081 | -0.0133 |
| LOGISTIC | `ABS_DIR_3D` | 0.5069 | 0.5067 | 0.5017 | 0.2563 | n/a | -0.0153 | 3 | n/a | 0.5095 | -0.0079 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_5D` | 0.5091 | 0.5036 | 0.5115 | 0.2560 | n/a | -0.0168 | 3 | n/a | 0.5091 | -0.0083 |
| LOGISTIC | `ABS_DIR_5D` | 0.4996 | 0.5041 | 0.4955 | 0.2607 | n/a | -0.0327 | 3 | n/a | 0.5015 | -0.0142 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_10D` | 0.5055 | 0.5017 | 0.5122 | 0.2593 | n/a | -0.0311 | 1 | n/a | 0.4993 | -0.0193 |
| LOGISTIC | `ABS_DIR_10D` | 0.5042 | 0.5122 | 0.5065 | 0.2663 | n/a | -0.0368 | 2 | n/a | 0.5060 | -0.0043 |

### X3_MACRO_COMMODITY

| model | objective | mean AUC | mean bal. acc | mean accuracy | mean Brier | train-majority | delta | years AUC>0.50 | ticker breadth | common-sample AUC | incremental AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| HIST_GRADIENT_BOOSTING | `ABS_DIR_1D_CONTROL` | 0.4986 | 0.4985 | 0.4968 | 0.2544 | n/a | -0.0045 | 2 | n/a | 0.5030 | -0.0100 |
| LOGISTIC | `ABS_DIR_1D_CONTROL` | 0.5124 | 0.5049 | 0.5041 | 0.2515 | n/a | 0.0028 | 4 | n/a | 0.5147 | 0.0033 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_3D` | 0.4980 | 0.4984 | 0.4982 | 0.2574 | n/a | -0.0176 | 3 | n/a | 0.5133 | -0.0081 |
| LOGISTIC | `ABS_DIR_3D` | 0.5179 | 0.5017 | 0.5065 | 0.2526 | n/a | -0.0092 | 3 | n/a | 0.5215 | 0.0041 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_5D` | 0.5049 | 0.5009 | 0.4976 | 0.2593 | n/a | -0.0291 | 3 | n/a | 0.5095 | -0.0078 |
| LOGISTIC | `ABS_DIR_5D` | 0.5307 | 0.5174 | 0.5239 | 0.2532 | n/a | -0.0028 | 4 | n/a | 0.5342 | 0.0185 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_10D` | 0.5222 | 0.5161 | 0.5132 | 0.2596 | n/a | -0.0285 | 4 | n/a | 0.5282 | 0.0096 |
| LOGISTIC | `ABS_DIR_10D` | 0.5356 | 0.5140 | 0.5246 | 0.2565 | n/a | -0.0170 | 4 | n/a | 0.5367 | 0.0265 |

### X4_ALL_EXOGENOUS

| model | objective | mean AUC | mean bal. acc | mean accuracy | mean Brier | train-majority | delta | years AUC>0.50 | ticker breadth | common-sample AUC | incremental AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| HIST_GRADIENT_BOOSTING | `ABS_DIR_1D_CONTROL` | 0.5048 | 0.5052 | 0.5035 | 0.2544 | n/a | 0.0015 | 3 | n/a | 0.5048 | -0.0082 |
| LOGISTIC | `ABS_DIR_1D_CONTROL` | 0.5102 | 0.5090 | 0.5059 | 0.2615 | n/a | 0.0039 | 4 | n/a | 0.5102 | -0.0012 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_3D` | 0.5309 | 0.5222 | 0.5172 | 0.2551 | n/a | -0.0013 | 5 | n/a | 0.5309 | 0.0095 |
| LOGISTIC | `ABS_DIR_3D` | 0.5179 | 0.5128 | 0.5026 | 0.2729 | n/a | -0.0159 | 5 | n/a | 0.5179 | 0.0005 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_5D` | 0.5315 | 0.5176 | 0.5072 | 0.2590 | n/a | -0.0228 | 5 | n/a | 0.5315 | 0.0142 |
| LOGISTIC | `ABS_DIR_5D` | 0.5232 | 0.5126 | 0.4999 | 0.2798 | n/a | -0.0301 | 5 | n/a | 0.5232 | 0.0075 |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_10D` | 0.5256 | 0.5151 | 0.4991 | 0.2641 | n/a | -0.0470 | 4 | n/a | 0.5256 | 0.0070 |
| LOGISTIC | `ABS_DIR_10D` | 0.5245 | 0.5046 | 0.4929 | 0.2889 | n/a | -0.0532 | 4 | n/a | 0.5245 | 0.0143 |

## 16. Common-sample incremental AUC table

| model | objective | incremental AUC | positive years | incremental balanced accuracy | incremental Brier |
|---|---|---|---|---|---|
| HIST_GRADIENT_BOOSTING | `ABS_DIR_1D_CONTROL` | -0.0060 | 3 | n/a | n/a |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_10D` | -0.0007 | 3 | n/a | n/a |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_3D` | 0.0068 | 2 | n/a | n/a |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_5D` | 0.0094 | 3 | n/a | n/a |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_1D_CONTROL` | -0.0088 | 2 | n/a | n/a |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_10D` | -0.0193 | 1 | n/a | n/a |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_3D` | -0.0133 | 2 | n/a | n/a |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_5D` | -0.0083 | 1 | n/a | n/a |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_1D_CONTROL` | -0.0100 | 1 | n/a | n/a |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_10D` | 0.0096 | 3 | n/a | n/a |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_3D` | -0.0081 | 1 | n/a | n/a |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_5D` | -0.0078 | 3 | n/a | n/a |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_1D_CONTROL` | -0.0082 | 1 | n/a | n/a |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_10D` | 0.0070 | 2 | n/a | n/a |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_3D` | 0.0095 | 4 | n/a | n/a |
| HIST_GRADIENT_BOOSTING | `ABS_DIR_5D` | 0.0142 | 4 | n/a | n/a |
| LOGISTIC | `ABS_DIR_1D_CONTROL` | 0.0158 | 5 | n/a | n/a |
| LOGISTIC | `ABS_DIR_10D` | 0.0370 | 4 | n/a | n/a |
| LOGISTIC | `ABS_DIR_3D` | 0.0236 | 4 | n/a | n/a |
| LOGISTIC | `ABS_DIR_5D` | 0.0298 | 4 | n/a | n/a |
| LOGISTIC | `ABS_DIR_1D_CONTROL` | -0.0096 | 0 | n/a | n/a |
| LOGISTIC | `ABS_DIR_10D` | -0.0043 | 1 | n/a | n/a |
| LOGISTIC | `ABS_DIR_3D` | -0.0079 | 2 | n/a | n/a |
| LOGISTIC | `ABS_DIR_5D` | -0.0142 | 1 | n/a | n/a |
| LOGISTIC | `ABS_DIR_1D_CONTROL` | 0.0033 | 3 | n/a | n/a |
| LOGISTIC | `ABS_DIR_10D` | 0.0265 | 3 | n/a | n/a |
| LOGISTIC | `ABS_DIR_3D` | 0.0041 | 2 | n/a | n/a |
| LOGISTIC | `ABS_DIR_5D` | 0.0185 | 4 | n/a | n/a |
| LOGISTIC | `ABS_DIR_1D_CONTROL` | -0.0012 | 3 | n/a | n/a |
| LOGISTIC | `ABS_DIR_10D` | 0.0143 | 2 | n/a | n/a |
| LOGISTIC | `ABS_DIR_3D` | 0.0005 | 3 | n/a | n/a |
| LOGISTIC | `ABS_DIR_5D` | 0.0075 | 4 | n/a | n/a |

## 19. Source / feature-family contribution ranking

Ranked by mean common-sample incremental ROC-AUC -- the attribution this programme exists to produce.

| rank | family | mean_common_incremental_auc | mean_common_roc_auc | mean_incremental_balanced_accuracy | mean_incremental_brier | positive_incremental_auc_years |
|---|---|---|---|---|---|---|
| 1 | X1_INDIA_MARKET | 0.0265 | 0.5402 | n/a | n/a | 17 |
| 2 | X3_MACRO_COMMODITY | 0.0131 | 0.5268 | n/a | n/a | 12 |
| 3 | X4_ALL_EXOGENOUS | 0.0053 | 0.5190 | n/a | n/a | 12 |
| 4 | X2_GLOBAL_RISK | -0.0090 | 0.5047 | n/a | n/a | 4 |

- Did actual Indian market information help? -> see X1_INDIA_MARKET above
- Did global-risk information help? -> see X2_GLOBAL_RISK above
- Did FX/commodity/rate information help? -> see X3_MACRO_COMMODITY above
- Did the combination help? -> see X4_ALL_EXOGENOUS above

## 20-21. Feature importance (screening only)

| family | model | method | feature | importance |
|---|---|---|---|---|
| X1_INDIA_MARKET | LOGISTIC | standardised coefficient magnitude | NIFTY50_realized_vol_10 | 0.35089 |
| X1_INDIA_MARKET | LOGISTIC | standardised coefficient magnitude | NIFTY50_drawdown_20 | 0.32723 |
| X1_INDIA_MARKET | LOGISTIC | standardised coefficient magnitude | NIFTY_IT_distance_from_ma20 | 0.31810 |
| X1_INDIA_MARKET | LOGISTIC | standardised coefficient magnitude | NIFTY_IT_drawdown_20 | -0.22260 |
| X1_INDIA_MARKET | LOGISTIC | standardised coefficient magnitude | NIFTY_BANK_realized_vol_10 | -0.20510 |
| X1_INDIA_MARKET | LOGISTIC | standardised coefficient magnitude | NIFTY_IT_drawdown_60 | 0.20005 |
| X1_INDIA_MARKET | LOGISTIC | standardised coefficient magnitude | NIFTY_BANK_drawdown_20 | -0.19910 |
| X1_INDIA_MARKET | LOGISTIC | standardised coefficient magnitude | NIFTY50_drawdown_60 | -0.19357 |
| X1_INDIA_MARKET | LOGISTIC | standardised coefficient magnitude | NIFTY_BANK_return_10 | 0.18013 |
| X1_INDIA_MARKET | LOGISTIC | standardised coefficient magnitude | NIFTY50_return_10 | -0.17218 |
| X1_INDIA_MARKET | LOGISTIC | standardised coefficient magnitude | NIFTY_BANK_return_20 | 0.15259 |
| X1_INDIA_MARKET | LOGISTIC | standardised coefficient magnitude | NIFTY50_distance_from_ma60 | 0.13901 |
| X1_INDIA_MARKET | LOGISTIC | standardised coefficient magnitude | NIFTY_IT_distance_from_ma60 | -0.11439 |
| X1_INDIA_MARKET | LOGISTIC | standardised coefficient magnitude | NIFTY_IT_return_5 | -0.11154 |
| X1_INDIA_MARKET | LOGISTIC | standardised coefficient magnitude | NIFTY_BANK_return_3 | 0.10183 |
| X1_INDIA_MARKET | LOGISTIC | standardised coefficient magnitude | NIFTY_IT_return_20 | -0.10145 |
| X1_INDIA_MARKET | LOGISTIC | standardised coefficient magnitude | INDIA_VIX_log_level | 0.09014 |
| X1_INDIA_MARKET | LOGISTIC | standardised coefficient magnitude | NIFTY_BANK_return_5 | -0.08736 |
| X1_INDIA_MARKET | LOGISTIC | standardised coefficient magnitude | log_return_3 | -0.08013 |
| X1_INDIA_MARKET | LOGISTIC | standardised coefficient magnitude | NIFTY50_realized_vol_20 | -0.08003 |

Importance is a validation-only interpretation aid. It is never used in this run to engineer further features, and it is not evidence of causation.

## 22-23. Pairs that SIGNAL-PASS / are STRONGLY PROMISING

- SIGNAL-PASS: **NONE**
- STRONGLY PROMISING: **NONE**

## 24-28. Neural stage, seed stability and freeze

_No neural fit was run: the screening gate did not pass an information family, so the programme stops before any sequence model._

- seed stability: not reached
- frozen before 2019: **NO**

## 29-31. 2019 lockbox

**NOT OPENED.** No information family passed the screening gate, so the programme stopped before any model was selected and 2019 was never read.

## 32-34. Maximum meaningful selective coverage

A >=65% selective result counts only at coverage >= 20% AND n >= 500; otherwise the status is `SELECTIVE_65_NOT_ESTABLISHED`. Selective accuracy is never called overall accuracy.

## 35-36. Regime audit (both counts MUST be zero)

- 2020+ STOCK observations consumed: **0**
- 2020+ EXOGENOUS observations consumed: **0**
- 2020+ rows present in the raw snapshots (capped at download): 0

## 37. Recommended next action

**BUILD_HISTORICAL_SENTIMENT_EVENT_DATASET**

Reported, never executed.

## Interpretation limits

- the universe is a fixed, later-reconstructed constituent list projected backwards
- external data reduces the sample set, so every increment is reported on a COMMON sample as well as on each family's natural sample
- classification accuracy is the primary metric; overlapping multi-day forecasts are not compounded into portfolio returns
- no news or sentiment was used: standard external market state must prove itself first

V3_EXOGENOUS_SIGNAL: NONE
