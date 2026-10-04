# V3 EXOGENOUS AVAILABILITY AUDIT

- sampled stock origin dates: **100** (deterministic, seed 42)
- prediction timestamp: after the NSE close on the stock trading date
- causality violations: **0**
- verdict: **PASS**

For each accepted source: the earliest and latest source observation used, the realised lag in source trading observations, and how many sampled origins used a same-date observation. A conservative-lag1 source must show zero same-date uses.

| source | class | checked | min source obs | max source obs | min lag | max lag | same-date uses | violations |
|---|---|---|---|---|---|---|---|---|
| `BRENT_CRUDE` | EXTERNAL_CONSERVATIVE_LAG1 | 67 | 2007-09-21 | 2019-03-06 | 0 | 1 | 0 | 0 |
| `DOW_JONES` | EXTERNAL_CONSERVATIVE_LAG1 | 78 | 2005-04-01 | 2019-03-06 | 0 | 1 | 0 | 0 |
| `EURO_STOXX_50` | EXTERNAL_CONSERVATIVE_LAG1 | 68 | 2007-07-16 | 2019-03-06 | 0 | 1 | 0 | 0 |
| `GOLD` | EXTERNAL_CONSERVATIVE_LAG1 | 78 | 2005-04-01 | 2019-03-06 | 0 | 1 | 0 | 0 |
| `HANG_SENG` | EXTERNAL_CONSERVATIVE_LAG1 | 78 | 2005-04-01 | 2019-03-06 | 0 | 1 | 0 | 0 |
| `INDIA_VIX` | INDIA_SAME_CLOSE | 65 | 2008-04-21 | 2019-03-07 | 0 | 0 | 65 | 0 |
| `NASDAQ_COMPOSITE` | EXTERNAL_CONSERVATIVE_LAG1 | 78 | 2005-04-01 | 2019-03-06 | 0 | 1 | 0 | 0 |
| `NIFTY50` | INDIA_SAME_CLOSE | 67 | 2007-09-24 | 2019-03-07 | 0 | 0 | 67 | 0 |
| `NIFTY_BANK` | INDIA_SAME_CLOSE | 67 | 2007-09-24 | 2019-03-07 | 0 | 0 | 67 | 0 |
| `NIFTY_IT` | INDIA_SAME_CLOSE | 67 | 2007-09-24 | 2019-03-07 | 0 | 0 | 67 | 0 |
| `NIFTY_PHARMA` | INDIA_SAME_CLOSE | 46 | 2011-02-02 | 2019-03-07 | 0 | 0 | 46 | 0 |
| `NIKKEI_225` | EXTERNAL_CONSERVATIVE_LAG1 | 78 | 2005-04-01 | 2019-03-06 | 0 | 1 | 0 | 0 |
| `SP500` | EXTERNAL_CONSERVATIVE_LAG1 | 78 | 2005-04-01 | 2019-03-06 | 0 | 1 | 0 | 0 |
| `USDINR` | EXTERNAL_CONSERVATIVE_LAG1 | 78 | 2005-04-01 | 2019-03-06 | 0 | 1 | 0 | 0 |
| `US_10Y_YIELD` | EXTERNAL_CONSERVATIVE_LAG1 | 78 | 2005-04-01 | 2019-03-06 | 0 | 1 | 0 | 0 |
| `US_DOLLAR_INDEX` | EXTERNAL_CONSERVATIVE_LAG1 | 78 | 2005-04-01 | 2019-03-06 | 0 | 1 | 0 | 0 |
| `US_VIX` | EXTERNAL_CONSERVATIVE_LAG1 | 78 | 2005-04-01 | 2019-03-06 | 0 | 1 | 0 | 0 |
| `WTI_CRUDE` | EXTERNAL_CONSERVATIVE_LAG1 | 78 | 2005-04-01 | 2019-03-06 | 0 | 1 | 0 | 0 |

Rule enforced: every exogenous value used at origin t must come from a source observation no later than t, and a conservative-lag1 source must never use a same-date observation

