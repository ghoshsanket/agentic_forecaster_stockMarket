# The USER-SUPPLIED LEGACY NIFTY-50 universe

`universe_id: USER_SUPPLIED_LEGACY_NIFTY50_UNIVERSE`

This document explains the historical universe used for the legacy Yahoo Finance
reconstruction, why its exact 50 labels are preserved verbatim, and — most
importantly — the **POND'S date anomaly**, which qualifies the whole list.

## 1. What this universe is

The user recovered, from memory, the list of securities that the **lost
original implementation** was intended to use. That list is authoritative for
reproduction and is stored, unmodified, in
`configs/nifty50_legacy_user_supplied.yaml` under `tickers`.

It is a genuinely different universe from the project's modern
`configs/nifty50.yaml`. The two are compared in
`metadata/modern_vs_legacy_universe.csv`:

| Relationship | Count |
|---|---|
| exact label overlap | 13 |
| renamed-equivalent (same listed issuer, different label) | 5 |
| legacy only | 32 |
| modern only | 32 |

The five renamed-equivalent pairs are:

| Legacy label | Modern label | Same listed issuer |
|---|---|---|
| `HINDLEVER` | `HINDUNILVR` | Hindustan Unilever Ltd |
| `INFOSYSTCH` | `INFY` | Infosys Ltd |
| `L&T` | `LT` | Larsen & Toubro Ltd |
| `SBI` | `SBIN` | State Bank of India |
| `TISCO` | `TATASTEEL` | Tata Steel Ltd |

So the two universes share **18 of 50 names** and differ in **32**. Neither
universe overwrites the other, and neither is silently substituted for the
other.

## 2. The POND'S date anomaly

### What the user recalled

The user recalls this list as the NIFTY-50 constituent snapshot as at
**2000-01-03**, which is also the first trading day in the requested download
window.

### What the corporate record shows

**Pond's (India) Limited was amalgamated into Hindustan Lever Limited with
effect from 15-Oct-1998**, retrospective to an appointed date of 1-Jan-1998:

* Boards approved the scheme on 16-Mar-1998, swap ratio **3 HLL : 4 PIL**
* Sanctioned by the Madras High Court on 7-Aug-1998 and the Bombay High Court
  on 10-Sep-1998
* Effective **15-Oct-1998**; record date 15-Feb-1999
* The POND'S scrip was separately listed on NSE and BSE and was **extinguished**
  on 15-Oct-1998

Source: Hindustan Unilever's own *Manual for Shareholders* (merger/demerger
table) and *Chronology of Key Events*; HUL shareholder approval at the 1998 AGM.

### The contradiction

```
15-Oct-1998   POND'S (India) Ltd ceases to exist as a listed security
                    |
                    |   14 months later
                    v
2000-01-03   the user's recalled "snapshot date"
```

**A security that ceased to exist in 1998 cannot be a constituent of an index
snapshot taken in 2000.** The recalled date and the membership are internally
inconsistent.

### Conclusion

The supplied list is therefore **not** claimed to be the literal 2000-01-03
constituent snapshot. It is one of three things:

1. an older NIFTY-50 constituent set than 2000-01-03;
2. a legacy data source whose labels were never refreshed after the 1998
   amalgamation — the most likely explanation, since a stale label is exactly
   the kind of defect that survives in an old data file; or
3. the exact list the lost original implementation used, even though its own
   historical-date label was imperfect.

For reproduction purposes the **user-supplied list is authoritative**. It is
preserved exactly, all 50 labels, in the supplied order, with the original
spellings including `POND'S`, `L&T`, `M&M`, `P&G`, `BRITISH OXYGEN (BOC)` and
`RHONE-POUL`.

### What was NOT done

* POND'S was **not deleted** from the universe.
* POND'S was **not** rewritten to `HINDUNILVR`.
* The date label was **not** silently changed to a "more accurate" year.
* `HINDUNILVR.NS` was **not** downloaded as a stand-in for POND'S.

POND'S is classified `merger_into_other_company`, its lineage policy is
`do_not_download`, its canonical sheet is intentionally empty, and
`metadata/legacy_coverage.csv` marks it `DATE_ANOMALY_TRUE` with
`ceased_to_exist_date = 1998-10-15`.

The same treatment is applied consistently to the other 8 securities that had
already been absorbed by the snapshot date or later: BURROUGHS, COCHINREFN,
IBP, ICICI, HDFC, RANBAXY, RHONE-POUL and SATYAMCOMP.

`date_confidence` in the config is recorded as
**`needs_independent_verification`**. Establishing the true constituent date
would require the NIFTY-50 index constituent file for 2000-01-03, which is
outside the scope of this data task and has **not** been fabricated.

## 3. The 50 labels, preserved

```
ACC, BAJAJ-AUTO, BPCL, BRITANNIA, BRITISH OXYGEN (BOC), BSES, BURROUGHS,
CIPLA, COCHINREFN, COLPAL, DRREDDY, EPL, GAIL, GE SHIPPING, GLAXO, GRASIM,
GUTRLCEMENT, HDFC, HDFCBANK, HINDALCO, HINDLEVER, HPCL, IBP, ICICI, IDBI,
IFCI, INDIACEM, INFOSYSTCH, IOC, ITC, KNOLLPHARM, L&T, M&M, MADRASCEM, MTNL,
NESTLEIND, NIIT, NOCIL, ONGC, P&G, POND'S, RANBAXY, RELIANCE, RHONE-POUL,
SATYAMCOMP, SBI, TATACHEM, TELCO, TISCO, VSTILL
```

Several of these are **not tickers** but human-readable labels
(`GE SHIPPING`, `BRITISH OXYGEN (BOC)`, `P&G`, `VSTILL`), several are
**superseded tickers** (`INFOSYSTCH`, `TELCO`, `TISCO`, `HINDLEVER`,
`KNOLLPHARM`), and several are **ticker-format variants** of the current
symbol (`HPCL` → `HINDPETRO`, `L&T` → `LT`, `SBI` → `SBIN`). Resolving them
requires the corporate-identity registry, not a string transform.

## 4. What happened to each security

See `docs/LEGACY_SECURITY_LINEAGE.md` for the full table and
`configs/legacy_security_lineage.yaml` for the machine-readable registry with
evidence for every entry. Summary:

| Outcome | Count |
|---|---|
| same surviving listed security (unchanged or same-entity rename) | 35 |
| demerger (listed shell continued, business perimeter changed) | 3 |
| statutory continuation | 3 |
| **merged into another issuer — series ends, successor NOT appended** | 9 |

Of the 41 securities with retrievable history, 23 cover the full window from
**2000-01-03**; the rest begin later because Yahoo's NSE history for those
symbols does not reach 2000 (13 begin 2002-07-01). This is a vendor coverage
limit, reported in `metadata/coverage_summary.json`, and it is **not** filled
with synthetic data.

## 5. Files

| Purpose | Path |
|---|---|
| Universe config (authoritative 50 labels) | `configs/nifty50_legacy_user_supplied.yaml` |
| Corporate-identity registry with evidence | `configs/legacy_security_lineage.yaml` |
| Per-security evidence notes | `<legacy root>/corporate_history/<label>.md` |
| Modern-vs-legacy comparison | `<legacy root>/metadata/modern_vs_legacy_universe.csv` |
| Coverage report | `<legacy root>/metadata/legacy_coverage.csv` |
| POND'S anomaly in machine-readable form | `legacy_coverage.csv` → `date_anomaly` column |

`<legacy root>` is `$AGENTIC_DATA_ROOT/yfinance_legacy_nifty50_2000_2025`,
which is completely independent of the earlier modern-universe dataset at
`$AGENTIC_DATA_ROOT/yfinance_daily_2000_2025`. The modern dataset was not
read for writing and not modified.
