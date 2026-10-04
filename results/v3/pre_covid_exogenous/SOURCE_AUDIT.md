# V3 EXOGENOUS SOURCE AUDIT

Every external source attempted, whether or not it worked. A rejected source is excluded, never imputed, proxied or silently replaced.

- absolute final allowed date: `2019-12-31`
- download window end (exclusive): `2020-01-01`
- declared: **27** | accepted: **18** | rejected: **9**

| source_id | provider | identifier | available | first_actual_date | last_actual_date | row_count | missing_fraction | duplicate_dates | suspicious_jump_count | availability_class | accepted | exclusion_reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NIFTY50 | yfinance | ^NSEI | True | 2007-09-17 | 2019-12-31 | 3001 | 0.032258064516129004 | 0 | 0 | INDIA_SAME_CLOSE | True |  |
| NIFTY_BANK | yfinance | ^NSEBANK | True | 2007-09-17 | 2019-12-31 | 3017 | 0.024193548387096753 | 0 | 0 | INDIA_SAME_CLOSE | True |  |
| INDIA_VIX | yfinance | ^INDIAVIX | True | 2008-03-03 | 2019-12-31 | 2899 | 0.024193548387096753 | 0 | 8 | INDIA_SAME_CLOSE | True |  |
| NIFTY_IT | yfinance | ^CNXIT | True | 2007-09-17 | 2019-12-31 | 3017 | 0.024193548387096753 | 0 | 0 | INDIA_SAME_CLOSE | True |  |
| NIFTY_AUTO | yfinance | ^CNXAUTO | False | None | None | 0 | None | 0 | 0 | INDIA_SAME_CLOSE | False | provider returned no rows |
| NIFTY_FMCG | yfinance | ^CNXFMCG | False | None | None | 0 | None | 0 | 0 | INDIA_SAME_CLOSE | False | provider returned no rows |
| NIFTY_PHARMA | yfinance | ^CNXPHARMA | True | 2011-01-31 | 2019-12-31 | 2190 | 0.024193548387096753 | 0 | 0 | INDIA_SAME_CLOSE | True |  |
| NIFTY_METAL | yfinance | ^CNXMETAL | False | None | None | 0 | None | 0 | 0 | INDIA_SAME_CLOSE | False | provider returned no rows |
| NIFTY_ENERGY | yfinance | ^CNXENERGY | False | None | None | 0 | None | 0 | 0 | INDIA_SAME_CLOSE | False | provider returned no rows |
| NIFTY_FIN_SERVICE | yfinance | ^NIFTY_FIN_SERVICE | False | None | None | 0 | None | 0 | 0 | INDIA_SAME_CLOSE | False | provider returned no rows |
| NIFTY_REALTY | yfinance | ^CNXREALTY | False | None | None | 0 | None | 0 | 0 | INDIA_SAME_CLOSE | False | provider returned no rows |
| NIFTY_PSU_BANK | yfinance | ^CNXPSUBANK | False | None | None | 0 | None | 0 | 0 | INDIA_SAME_CLOSE | False | provider returned no rows |
| SP500 | yfinance | ^GSPC | True | 2005-01-03 | 2019-12-31 | 3775 | 0.003968253968253954 | 0 | 0 | EXTERNAL_CONSERVATIVE_LAG1 | True |  |
| NASDAQ_COMPOSITE | yfinance | ^IXIC | True | 2005-01-03 | 2019-12-31 | 3775 | 0.003968253968253954 | 0 | 0 | EXTERNAL_CONSERVATIVE_LAG1 | True |  |
| DOW_JONES | yfinance | ^DJI | True | 2005-01-03 | 2019-12-31 | 3775 | 0.003968253968253954 | 0 | 0 | EXTERNAL_CONSERVATIVE_LAG1 | True |  |
| US_VIX | yfinance | ^VIX | True | 2005-01-03 | 2019-12-31 | 3775 | 0.003968253968253954 | 0 | 11 | EXTERNAL_CONSERVATIVE_LAG1 | True |  |
| NIKKEI_225 | yfinance | ^N225 | True | 2005-01-04 | 2019-12-30 | 3673 | 0.012145748987854255 | 0 | 0 | EXTERNAL_CONSERVATIVE_LAG1 | True |  |
| HANG_SENG | yfinance | ^HSI | True | 2005-01-03 | 2019-12-31 | 3696 | 0.0080971659919028 | 0 | 0 | EXTERNAL_CONSERVATIVE_LAG1 | True |  |
| EURO_STOXX_50 | yfinance | ^STOXX50E | True | 2007-03-30 | 2019-12-30 | 3191 | 0.027559055118110187 | 0 | 0 | EXTERNAL_CONSERVATIVE_LAG1 | True |  |
| USDINR | yfinance | INR=X | True | 2005-01-03 | 2019-12-31 | 3882 | 0.011494252873563204 | 0 | 0 | EXTERNAL_CONSERVATIVE_LAG1 | True |  |
| BRENT_CRUDE | yfinance | BZ=F | True | 2007-07-30 | 2019-12-31 | 3074 | 0.007936507936507908 | 0 | 0 | EXTERNAL_CONSERVATIVE_LAG1 | True |  |
| WTI_CRUDE | yfinance | CL=F | True | 2005-01-03 | 2019-12-31 | 3772 | 0.007936507936507908 | 0 | 0 | EXTERNAL_CONSERVATIVE_LAG1 | True |  |
| GOLD | yfinance | GC=F | True | 2005-01-03 | 2019-12-31 | 3768 | 0.007936507936507908 | 0 | 0 | EXTERNAL_CONSERVATIVE_LAG1 | True |  |
| US_DOLLAR_INDEX | yfinance | DX-Y.NYB | True | 2005-01-03 | 2019-12-31 | 3779 | 0.003968253968253954 | 0 | 0 | EXTERNAL_CONSERVATIVE_LAG1 | True |  |
| US_10Y_YIELD | yfinance | ^TNX | True | 2005-01-03 | 2019-12-31 | 3770 | 0.003968253968253954 | 0 | 0 | EXTERNAL_CONSERVATIVE_LAG1 | True |  |
| FII_NET_FLOW | none_available | NOT_PROVIDED | False | None | None | 0 | None | 0 | 0 | EXTERNAL_CONSERVATIVE_LAG1 | False | unavailable |
| DII_NET_FLOW | none_available | NOT_PROVIDED | False | None | None | 0 | None | 0 | 0 | EXTERNAL_CONSERVATIVE_LAG1 | False | unavailable |

## Accepted sources and their final lag rule

| source_id | availability_class | final_lag_rule | SHA256 |
|---|---|---|---|
| NIFTY50 | INDIA_SAME_CLOSE | same-session close (INDIA_SAME_CLOSE) | 181647ebb31cc072... |
| NIFTY_BANK | INDIA_SAME_CLOSE | same-session close (INDIA_SAME_CLOSE) | 48dde02d7ca17dc1... |
| INDIA_VIX | INDIA_SAME_CLOSE | same-session close (INDIA_SAME_CLOSE) | d64534a6ac525d96... |
| NIFTY_IT | INDIA_SAME_CLOSE | same-session close (INDIA_SAME_CLOSE) | 52d8033a5cad4055... |
| NIFTY_PHARMA | INDIA_SAME_CLOSE | same-session close (INDIA_SAME_CLOSE) | b2b9c3b86fd66b74... |
| SP500 | EXTERNAL_CONSERVATIVE_LAG1 | conservative lag1: most recent source observation strictly before the NSE prediction timestamp | 77e9beb8794d7f40... |
| NASDAQ_COMPOSITE | EXTERNAL_CONSERVATIVE_LAG1 | conservative lag1: most recent source observation strictly before the NSE prediction timestamp | c4f9bf03c5b6d7a2... |
| DOW_JONES | EXTERNAL_CONSERVATIVE_LAG1 | conservative lag1: most recent source observation strictly before the NSE prediction timestamp | 3f11a2732af92650... |
| US_VIX | EXTERNAL_CONSERVATIVE_LAG1 | conservative lag1: most recent source observation strictly before the NSE prediction timestamp | 9be02a8e03a18239... |
| NIKKEI_225 | EXTERNAL_CONSERVATIVE_LAG1 | conservative lag1: most recent source observation strictly before the NSE prediction timestamp | ea782474f944dd75... |
| HANG_SENG | EXTERNAL_CONSERVATIVE_LAG1 | conservative lag1: most recent source observation strictly before the NSE prediction timestamp | cf05ca010e8925dd... |
| EURO_STOXX_50 | EXTERNAL_CONSERVATIVE_LAG1 | conservative lag1: most recent source observation strictly before the NSE prediction timestamp | 96cb512c4fe59640... |
| USDINR | EXTERNAL_CONSERVATIVE_LAG1 | conservative lag1: most recent source observation strictly before the NSE prediction timestamp | cabcbdfe3cf04ed6... |
| BRENT_CRUDE | EXTERNAL_CONSERVATIVE_LAG1 | conservative lag1: most recent source observation strictly before the NSE prediction timestamp | 326b76fb6fa4832b... |
| WTI_CRUDE | EXTERNAL_CONSERVATIVE_LAG1 | conservative lag1: most recent source observation strictly before the NSE prediction timestamp | a3df3d255a41e8c6... |
| GOLD | EXTERNAL_CONSERVATIVE_LAG1 | conservative lag1: most recent source observation strictly before the NSE prediction timestamp | e7124254a921b931... |
| US_DOLLAR_INDEX | EXTERNAL_CONSERVATIVE_LAG1 | conservative lag1: most recent source observation strictly before the NSE prediction timestamp | 3d719fd51e0eb77d... |
| US_10Y_YIELD | EXTERNAL_CONSERVATIVE_LAG1 | conservative lag1: most recent source observation strictly before the NSE prediction timestamp | 8ec0c22e7ddf48c2... |

## Rejected sources and why

| source_id | identifier | available | exclusion_reason |
|---|---|---|---|
| NIFTY_AUTO | ^CNXAUTO | False | provider returned no rows |
| NIFTY_FMCG | ^CNXFMCG | False | provider returned no rows |
| NIFTY_METAL | ^CNXMETAL | False | provider returned no rows |
| NIFTY_ENERGY | ^CNXENERGY | False | provider returned no rows |
| NIFTY_FIN_SERVICE | ^NIFTY_FIN_SERVICE | False | provider returned no rows |
| NIFTY_REALTY | ^CNXREALTY | False | provider returned no rows |
| NIFTY_PSU_BANK | ^CNXPSUBANK | False | provider returned no rows |
| FII_NET_FLOW | NOT_PROVIDED | False | unavailable |
| DII_NET_FLOW | NOT_PROVIDED | False | unavailable |

## Source-quality rules applied

- mandatory valid coverage across every development year 2014-2018
- missing fraction in the development years below 5 %
- no unexplained gap longer than 45 calendar days
- business-day granularity where the session expects it
- every non-Indian source uses conservative lag1 unless a documented earlier session close is recorded in its availability class
- the reconstructed equal-weight stock proxy is NOT used and is NOT called an index

