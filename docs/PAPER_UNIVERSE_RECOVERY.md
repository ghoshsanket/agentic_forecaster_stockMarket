# Recovering the paper's evaluation universe

Machine-readable evidence: `results/reproduction_recovery/paper_universe_evidence.json`.

## The headline finding

**The user-supplied legacy universe is NOT the paper's evaluation universe.**

The paper's representative predictions for **2023-07-05** explicitly list:

```
RELIANCE    TCS    INFY    HDFCBANK    ITC
```

The user-supplied legacy set (`configs/nifty50_legacy_user_supplied.yaml`)
contains **four** of those five and is **missing `TCS`** entirely. `INFY` is
present only under the superseded label `INFOSYSTCH`.

```
USER_SUPPLIED_LEGACY_NIFTY50_UNIVERSE  !=  PAPER_EVALUATION_UNIVERSE
```

This is not a small discrepancy. It means the fixed 50-label legacy list
cannot, by itself, be the set that produced the paper's 2022/2023 results.
Exactly one of the following must be true:

1. the original experiment used a **time-varying** NIFTY-50 constituent
   universe (membership applied per date), not a fixed list; or
2. the paper used a **different fixed 50-security list** that the legacy
   recollection does not match; or
3. the paper's example tickers were chosen for presentation and are not a
   complete description of the evaluated universe.

Nothing in the local artifacts distinguishes these. The contradiction is
recorded rather than resolved by assumption.

## The three universes

The project now names three distinct universes. They are **not** aliases of one
another, and none overwrites another.

| ID | Config | Nature | Contains TCS? | Status |
|---|---|---|---|---|
| **A. `MODERN_RECONSTRUCTED_UNIVERSE`** | `configs/nifty50.yaml` | today's NIFTY-50, fixed by the reconstruction team | yes | Used for the completed Kaggle reproduction |
| **B. `USER_SUPPLIED_LEGACY_UNIVERSE`** | `configs/nifty50_legacy_user_supplied.yaml` | the user's recollection of the original implementation | **no** | **LEGACY UNIVERSE SENSITIVITY DATASET** |
| **C. `PAPER_EVALUATION_UNIVERSE`** | — | the set that produced the paper's reported numbers | unknown | **UNKNOWN / TO BE RECOVERED** |

**B must not be aliased to C.** The legacy dataset remains valuable as a
sensitivity dataset — for historical-universe analysis, adjusted-vs-unadjusted
comparison, old-stock lineage research, and testing whether more training
history helps — but it is labelled *LEGACY UNIVERSE SENSITIVITY DATASET* until
evidence establishes it was the paper's universe. It has not been.

## The 2025-11-04 snapshot candidate, and how DUMMYTATAM is handled

`configs/nifty50_paper_snapshot_2025_11_04.yaml` adds a fourth, explicit
candidate: `PAPER_FIXED_NIFTY50_2025_11_04_CANDIDATE`, built from official NSE
Indices evidence and used for the new paper-snapshot dataset.

The 50 names come from NSE's own constituent file
(`nsearchives.nseindia.com/content/indices/ind_nifty50list.csv`), and the
2025-11-04 composition is established by **bracketing** official press
releases (2025-03-28, 2025-09-30, 2025-10-14, 2025-11-17) plus the 2026-03-30
review confirming no Nifty 50 change.

### Precise wording

The correct statement is:

> `PAPER_FIXED_NIFTY50_2025_11_04_CANDIDATE` represents the 50 ordinary
> listed/tradable security lines reconstructed for a fixed Yahoo workbook,
> **excluding the temporary DUMMYTATAM index-only demerger placeholder.**

It is **not** accurate to say "the Nifty-50 composition on 2025-11-04 equals
the current 50-row NSE file". On 2025-11-04 the index methodology *did*
temporarily include `DUMMYTATAM`, so the index carried 51 lines. The candidate
list is the 50 *tradable* lines with that synthetic entity removed.

### DUMMYTATAM in detail

Per NSE Indices press release `ind_prs07102025.pdf` (7 Oct 2025): following the
Tata Motors demerger, "demerged entity TML Commercial Vehicles Ltd. (with a
dummy Symbol 'DUMMYTATAM') shall be included at zero price without divisor
adjustment" in Nifty 50 effective 2025-10-14 (close of 2025-10-13), alongside
NSE circular NSE/CMTR/70614 of 2025-10-03.

* It was **never a tradable listed security** and has **no price history**.
* It is an **index-maintenance entity**, not a company.
* Therefore it **cannot be a Yahoo Finance sheet**, and **no price data has
  been fabricated** for it.
* The parent was **not** removed: Tata Motors Ltd was renamed Tata Motors
  Passenger Vehicles Ltd, kept its Nifty 50 slot and its **original ISIN
  INE155A01022**, moving symbol `TATAMOTORS` -> `TMPV` on 2025-10-24. The ISIN
  continuity proves the same listed entity under a new name, so **TMPV** is
  retained as the continuing original Tata Motors security line.
* `DUMMYTATAM` was cleared on 2025-11-17 when the CV entity failed the price
  band. The 2025-12-05 HUL ice-cream demerger produced the same kind of dummy
  (`DUMMYHDLVR`), which corroborates the mechanism.

**Naming trap:** the name "Tata Motors Limited" now belongs to a *different*,
newly incorporated company (`TMCV`, listed 2025-11-12) whose Yahoo history
begins only on that date. `TATAMOTORS.NS` returns nothing on Yahoo; `TMPV.NS`
carries the continuous history.

## The 2000-2015 observations do not enter the paper folds

The reconstructed Yahoo workbook now begins around **2000-01-03** for 37 of the
50 paper-snapshot securities (13 list later, e.g. JIOFIN 2023-08-21).

Equation 21's walk-forward training begins in **2016**. The paper's two folds
are:

* fold_0: train 2016-2020, validation 2021, test 2022
* fold_1: train 2016-2021, validation 2022, test 2023

**Therefore the extra 2000-2015 observations do not enter the paper walk-forward
folds at all.** They cannot improve Table-II walk-forward performance. They
remain useful for:

* reproducing the original workbook faithfully,
* historical sensitivity analysis,
* investigating the publication's separate **85/15 chronological split**
  description, which may cover a longer span and may correspond to the lost
  original implementation. That protocol is evaluated separately under
  `forensic_diagnostics/` and is labelled **FORENSIC / ALTERNATE PAPER
  PROTOCOL**. It must not be mixed into the official walk-forward result, and
  it must not be chosen merely because its test metric is closer to 0.815.

## Local search: what was actually found

A read-only sweep of user-owned artifacts under
`/home/iemiedc2026/Documents/Sanket/Research` and `/home/iemiedc2026/Documents/Sanket`,
plus every commit in the repository's git history, found **four** distinct
50-item lists. No other user's files were read.

| List | Source | TCS? | Verdict |
|---|---|---|---|
| `0d0c3b9` | `git 0d0c3b9:configs/nifty50.yaml` (current) | yes | Reconstruction. Its own header says membership is time-varying and the publication gives no as-of date, so the team fixed the list. Contains post-2022 additions (`SHRIRAMFIN`, `TRENT`, `INDIGO`, `ZOMATO`), inconsistent with a 2023-07-05 prediction date. |
| `a43b5a9` | `git a43b5a9:configs/nifty50.yaml` | yes | Intermediate edit. Its own diff comment records fabrication: *"INFRATEL → not present in dataset; replaced by TATACOMM's parent (already listed)"*. |
| `17ea2f1` | `git 17ea2f1:configs/nifty50.yaml` ("First implementation") | yes | Earliest and closest in vintage, **but** its header calls it a *"NIFTY 100 constituent subset"* and it is written in Kaggle vendor spellings. It lists `INFRATEL` and `TATAMOTORS`, neither of which exists in the Kaggle dataset — proving it was reverse-engineered from the downloaded data, not transcribed from the paper. |
| `legacy` | `configs/nifty50_legacy_user_supplied.yaml` | **no** | User recollection, recovered from memory. |

**Conclusion: no artifact on this machine transcribes the paper's 50-security
list.** Three lists contain TCS, but all three are reconstruction-team edits of
`configs/nifty50.yaml`, and none is dated or sourced to the publication.

### Supporting absences

* **The paper itself is not present locally.** All 184 PDFs under `Research` are
  generated prediction reports. `paper_reference.json` holds only four scalar
  metrics plus the DOI `10.1109/IEMENTECH202669403.2026.11434302`.
* **No index-constituent file exists locally.** Nothing matching
  `*constituent*` / `*nifty*` / `*nse*`.
* The closest local superset, `dataset/agentic-forecaster/metadata/discovered_tickers.txt`
  (531 symbols), is a **NIFTY-100-era vendor symbol listing** from the Kaggle
  download, not a dated NIFTY-50 constituent history. It contains `TCS`,
  `SHREECEM`, `HDFCLIFE`, `PFC` and lacks `INFRATEL` and `TATAMOTORS`. It
  **cannot** reconstruct a 2022-2023 constituent list.

## No paper universe has been selected

Work stopped before obtaining any external constituent list, and no further
50-stock universe has been downloaded. The four candidate policies below are
set out for an explicit decision.

### Candidate 1 — NIFTY-50 as at the experiment/download date

* **Advantages:** reproducible with a single public snapshot; closest to what a
  2026 re-implementer would obtain today; already approximated by universe A.
* **Disadvantages:** introduces **look-ahead bias** if used for 2022/2023
  evaluation, because it applies 2026 membership backwards; contains names
  (`ZOMATO`, `SHRIRAMFIN`, `TRENT`, `INDIGO`) that did not exist as
  constituents then, and omits names that did.
* **Constituent count:** 50.
* **Includes TCS/RELIANCE/INFY/HDFCBANK/ITC:** yes.
* **Survivorship:** severe — later entrants and delisted names are excluded, and
  future composition is applied to the past.

### Candidate 2 — NIFTY-50 as at 2023-07-05 (the prediction date)

* **Advantages:** matches the paper's own presentation date; a single dated
  snapshot; avoids importing post-2023 entrants.
* **Disadvantages:** a **single date still does not represent a 2022–2023
  training-plus-test period**; membership changed within that window, so
  training-period composition is still an approximation.
* **Constituent count:** 50.
* **Includes TCS/RELIANCE/INFY/HDFCBANK/ITC:** yes.
* **Survivorship:** moderate — biased toward names that survived to mid-2023.

### Candidate 3 — time-varying NIFTY-50 membership over 2022-2023

* **Advantages:** the **only** policy that avoids survivorship bias entirely;
  each date is evaluated against the membership actually in force; consistent
  with the hypothesis that the original experiment used an index-relative
  universe rather than a fixed file.
* **Disadvantages:** needs a genuine dated constituent history (NSE publishes
  change files, which are **not** available locally); produces a
  non-constant universe that complicates cross-sectional models and makes
  per-stock "one model per stock" accounting uneven; harder to reproduce.
* **Constituent count:** 50 per date, varying composition over the window.
* **Includes TCS/RELIANCE/INFY/HDFCBANK/ITC:** yes, for the relevant dates.
* **Survivorship:** none, by construction.

### Candidate 4 — the earliest committed modern list (`17ea2f1`)

* **Advantages:** the only local 50-list that contains TCS **and** predates the
  post-2022 additions, so it is the closest in vintage to a 2023-07-05
  experiment; requires no external data.
* **Disadvantages:** **still a reconstruction**, reverse-engineered from the
  Kaggle column names rather than from the paper; contains `INFRATEL` and
  `TATAMOTORS`, absent from the Kaggle dataset, so it was never a working list
  as written; its header mislabels the universe as a NIFTY-100 subset.
* **Constituent count:** 50.
* **Includes TCS/RELIANCE/INFY/HDFCBANK/ITC:** yes.
* **Survivorship:** unknown — cannot be assessed for a fabricated list.

## What is needed to close this

1. The publication's own universe statement, if it names one. The paper is not
   present locally; DOI `10.1109/IEMENTECH202669403.2026.11434302`.
2. Alternatively, an as-of-dated NIFTY-50 constituent file for 2022-2023
   (NSE publishes these; they are not in the workspace).
3. Failing both, an explicit decision to adopt one of the four policies above,
   with the resulting survivorship caveat stated in the write-up.

Until then, `PAPER_EVALUATION_UNIVERSE` stays **UNKNOWN**, and the completed
Kaggle reproduction in `results/paper_reproduction/` should continue to be
described as a *reconstruction*, not as a reproduction of a proven universe.

## Related documents

* `docs/LEGACY_UNIVERSE_HISTORY.md` — the legacy list, and the POND'S 1998
  date anomaly that independently limits it.
* `docs/YFINANCE_LEGACY_DATASET.md` — the legacy dataset itself, now labelled a
  sensitivity dataset.
* `docs/PAPER_TRACEABILITY.md` — how paper-defined values map to code. Its
  "50 NIFTY-50 stocks → `configs/nifty50.yaml`" row is a *count* claim resting
  on a reconstructed list, and should be read with this document.
