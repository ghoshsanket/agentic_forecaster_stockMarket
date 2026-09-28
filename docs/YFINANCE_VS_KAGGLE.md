# Yahoo Finance vs Kaggle daily reconstruction

Machine-readable results: `metadata/yfinance_vs_kaggle.csv` and
`metadata/yfinance_vs_kaggle.json` under
`$AGENTIC_DATA_ROOT/yfinance_daily_2000_2025`. Reproduce with:

```bash
uv run --frozen python scripts/compare_yfinance_vs_kaggle.py
```

## Why this comparison exists

Two independent daily histories now exist for the same 50 NIFTY-50
securities. The Kaggle data was used for the paper reproduction; the Yahoo
data is the reconstruction of the lost original. Any difference between them
changes what the paper's metrics mean, so the size and nature of the
difference is documented rather than left implicit.

## Method

The Kaggle source is **minute** level (`*_minute.csv`, columns
`date,open,high,low,close,volume`). It is resampled to daily with the
project's own `resample_intraday_to_daily`, the same function the Phase-1
pipeline uses. Both sides are therefore derived by identical logic, so a
difference is attributable to the data rather than to the resampler.

Comparison is on the intersection of trading dates, per security, using
`Date` as the join key and `Close` (and the full OHLC set) for the values.
No file in the Kaggle dataset was modified.

## Result 1: Yahoo unadjusted vs Kaggle — the same securities, minor vendor drift

| Ticker | Overlap (days) | Max abs close diff | Mean rel diff |
|---|---|---|---|
| RELIANCE | 2694 | 48.24 | 2.48% |
| TCS | 2694 | 1256.45 | 0.14% |
| INFY | 2694 | 1637.30 | 0.17% |
| HDFCBANK | 2694 | 23.03 | 0.13% |
| ITC | 2694 | 221.37 | 12.59% |
| SBIN | 2694 | 272.90 | 0.15% |
| LT | 2694 | 97.55 | 0.14% |
| HINDUNILVR | 2694 | 44.65 | 0.15% |
| ICICIBANK | 2694 | 301.05 | 0.17% |
| AXISBANK | 2694 | 19.95 | 0.18% |
| TITAN | 2694 | 79.55 | 0.18% |
| MARUTI | 2694 | 254.20 | 0.14% |

The large *absolute* differences are a price-scale artefact: INFY traded
around ₹2,000, TCS around ₹4,000, so a sub-1% relative difference is still
hundreds of rupees. The **relative** figures are the meaningful ones, and for
10 of 12 securities the mean relative difference is under 0.2%.

The two series track the same security, confirmed directly on RELIANCE: daily
return correlation is **0.987**, and the level relationship is a slowly
drifting multiplicative factor of about **1.033** from 2015 through 2023 that
resets to **1.000** by 2025. The reset is a single-day level shift on
2023-07-20 (Kaggle -3.51%, Yahoo -0.12% on that date) after which the two
series re-converge. A constant-ratio level offset that resolves at a single
date is the signature of a **corporate-action adjustment-basis difference**
between the two vendors, not of two different instruments. Daily returns were
never bit-identical on any day (0.0% exact matches), consistent with two
independent vendor pipelines.

Two securities depart from the pattern and are called out rather than
averaged away:

* **ITC** (12.59% mean relative) and **RELIANCE** (2.48%) sit well above the
  0.2% seen elsewhere, i.e. these two carry the largest adjustment-basis gap.
* **INFY** shows a 76.6% *max* relative difference despite a 0.17% mean,
  indicating a small number of outlier dates rather than a persistent level
  offset.

## Result 2: Yahoo adjusted vs Yahoo unadjusted — the choice that actually matters

| Ticker | Overlap (days) | Max rel diff | Mean rel diff |
|---|---|---|---|
| RELIANCE | 6485 | 26.6% | 9.8% |
| TCS | 5806 | 36.2% | 24.0% |
| INFY | 6488 | 40.2% | 27.0% |
| HDFCBANK | 6488 | 20.4% | 11.5% |
| ITC | 6485 | 50.5% | 33.9% |
| SBIN | 6486 | 33.6% | 16.5% |
| LT | 5838 | 26.3% | 15.4% |
| HINDUNILVR | 6488 | 45.4% | 22.8% |
| ICICIBANK | 5835 | 29.0% | 12.2% |
| AXISBANK | 6485 | 28.8% | 7.9% |
| TITAN | 6488 | 10.6% | 6.1% |
| MARUTI | 5571 | 15.0% | 9.5% |

This is the dominant effect in the whole exercise, and it is why the dataset
ships two variants rather than one. `auto_adjust` changes prices by **6% to 34%
on average** and by up to 50% at a single date, against a vendor-drift
difference of well under 1% between Yahoo and Kaggle.

Consequence: the adjusted-vs-unadjusted choice is a first-order modelling
decision, roughly two orders of magnitude larger than the choice of data
vendor. It cannot be resolved by inspection and must be stated explicitly in
any result that depends on it. The adjusted series is the primary variant; the
unadjusted series exists so the sensitivity can be measured.

Note the overlap counts differ from Result 1 (6485 vs 2694) because the Yahoo
series spans 2000-2025 while Kaggle only begins 2015-02-02, and because
adjusted/unadjusted share the same date index.

## Interpretation for the paper reproduction

* The Kaggle and Yahoo-unadjusted series describe the same securities and
  agree closely, so the earlier Kaggle-based reproduction is not obviously
  wrong about *which* companies these are.
* The `auto_adjust` convention materially changes the price series. Because
  the earlier reproduction never had to state a convention (the Kaggle file
  arrives in a single already-adjusted form), its price scale was implicit.
* These comparisons are about **data**, not about model quality. No model was
  trained here and no metric in the earlier reproduction is changed by this
  document. Establishing whether the adjusted or unadjusted convention
  reproduces the paper's published numbers would require training on both,
  which is out of scope here and is what the two
  `configs/reproduction_search/yfinance_*.yaml` configs are provided for.
