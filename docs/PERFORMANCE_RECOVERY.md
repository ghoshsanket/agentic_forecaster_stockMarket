# Performance recovery

Goal: recover the publication's predictive performance **scientifically**,
without repeatedly tuning against the final 2022/2023 test years.

## The one rule

**The 2022 and 2023 test years must remain unseen until exactly one
configuration has been frozen.**

Everything else in this document is machinery in service of that rule. A
configuration chosen by minimising distance to the published numbers tells us
nothing about whether the reconstruction is correct; it only tells us that we
fitted the answer.

## The test-set firewall

`agentic_forecaster.recovery.firewall` enforces the rule in three independent
places, so a single oversight cannot leak:

| Mechanism | Catches |
|---|---|
| `assert_pre_test_dates()` | any attempt to score a date >= 2022-01-01 |
| `firewall_guard()` | a whole search stage / run |
| `assert_config_windows_safe()` | a config whose val/test window reaches 2022 |

`assert_labels_not_firewalled()` additionally **refuses** label arrays that
arrive without dates, because without dates it is impossible to prove the
labels are pre-2022 — and a silent pass would defeat the firewall.

The firewall does **not** block verifying that test files and test dates exist.
It only blocks reading test labels or computing test metrics.

## Search folds (all strictly pre-2022)

| Fold | Train | Validation |
|---|---|---|
| `SEARCH_FOLD_A` | 2016-01-01 → 2018-12-31 | 2019 |
| `SEARCH_FOLD_B` | 2016-01-01 → 2019-12-31 | 2020 |
| `SEARCH_FOLD_C` | 2016-01-01 → 2020-12-31 | 2021 |

`SEARCH_FOLD_C` is the closest pre-test analogue of paper fold 0's split.
`fold_config_for()` applies a fold and **removes any inherited `test_*`
window**, so a copy-pasted paper config cannot silently evaluate 2022.

The paper folds (`fold_0`, `fold_1`) remain defined in the same module but are
reachable only through the final, frozen, explicitly authorised run.

## Dataset variants

* `paper_snapshot_2025_unadjusted_perf` — **PRIMARY original-recollection
  candidate** (the author recalls the original as probably unadjusted).
* `paper_snapshot_2025_adjusted_perf` — **sensitivity variant**.

Do not assume adjusted data performs better. On the modern universe the two
variants differed by 6–34% in mean price, so the choice is first-order and must
be measured.

## Staged search (not a Cartesian product)

| Stage | Question | Axes |
|---|---|---|
| **0** | can the model learn at all? | tiny overfit; shuffled-label control |
| **A** | training length × feature set | T3/T30/T50/T100 × F1/F2, 8–10 stocks, folds A/B/C |
| **B** | RSI / scaler / lookback | R0/R1/R2 × standard/minmax/robust × 10/20/30/40/60 |
| **C** | architecture / dropout / L2 / class weight | A0–A3 × 0.0/0.2/0.5 × 0/1e-6…1e-3 × none/pos_weight |
| **D** | calibration / seed stability | none/temperature/platt/isotonic × 5 seeds |
| **E** | confirm on all 50 | best 2–3, pre-2022 validation only |

**STAGE 0 is a hard gate — but only its first two checks are.** The gate exists
to protect the *interpretation* of later numbers, so it is scoped to questions
that can be answered on a tiny subset with no risk of fooling us:

| Stage-0 check | Kind | Criterion |
|---|---|---|
| tiny overfit | **HARD GATE** | the model can drive TRAIN loss toward zero |
| shuffled-label control | **HARD GATE** | shuffled TRAIN labels must not beat chance on validation |
| real-label reference | **NOT A GATE** | reported for context only; never blocks the search |

If either hard gate fails, the search stops and the pipeline is diagnosed.
Numbers from a broken pipeline are uninterpretable.

The real-label reference is deliberately **not** a gate. One ticker over one
pre-2022 validation window is not a test of predictive signal — it has neither
the breadth nor the statistical power to establish it, so a low value is weak
evidence, not disproof. Treating it as a gate would let a single noisy window
discard an otherwise sound pipeline, and it would smuggle a signal requirement
into a stage that exists to test *mechanics*. Whether signal exists at all is
answered across stocks and windows by the **Stage-A pilot** (§ below), which is
the first stage with the breadth to support the question.

### Training length is AUTHOR-CONFIRMED at 10 epochs

| Setting | Value | Evidence class |
|---|---|---|
| Adam learning rate | 1e-3 | **PAPER-DEFINED** |
| Adam beta1 / beta2 | 0.9 / 0.999 | **PAPER-DEFINED** |
| Batch size | 64 | **PAPER-DEFINED** |
| Early-stopping patience | 10 | **PAPER-DEFINED** |
| **Maximum training epochs** | **10** | **AUTHOR-CONFIRMED** |

The original author confirmed on 2026-09-29 that the original implementation
trained for **a maximum of 10 epochs**. This supersedes the earlier
reconstruction assumption that no epoch cap was specified.

Because patience is also 10, the epoch cap binds at or before early stopping can
fire, so **10 epochs is the effective schedule** — the two settings do not
compound into a longer budget.

Two epoch values that appeared earlier are explicitly *not* the original:

| Value | What it actually was |
|---|---|
| `T3` (3 epochs) | an early **reconstruction shortcut**, never the author's choice |
| `T100` (100 epochs) | a **recovery diagnostic**, never a reproduction candidate |

Consequently the epoch budget is **no longer a search axis**. Stage A compares
the compact (F1) and expanded (F2) feature sets at the confirmed schedule, and
T3/T100 are retained only to quantify what the confirmed budget costs. If a
diagnostic configuration beats T10, that is a finding about the diagnostic — not
grounds to select it. Faithful reconstruction of the original implementation is
the objective, so **T10 remains the primary candidate unless new author evidence
says otherwise**.

## Feature families

| Family | Contents |
|---|---|
| **F0** | OHLCV only |
| **F1** | Phase-1 compact set (reference) |
| **F2** | F1 + SMA5, SMA20, SMA5−SMA20, Bollinger upper/lower/%B, OBV — the features named in the publication's example explanations |
| **F3** | F2 + already-existing causal short-return/momentum features |

F2 exists because the paper's example explanations mention 5-day MA, 20-day MA,
Bollinger %B and OBV, which the compact F1 set does not contain. That is
evidence the lost implementation may have had them.

All features remain **causal** (trailing windows only, no future information),
and a test asserts every F1/F2/F3 indicator actually builds.

## Other recovered dimensions

* **RSI**: R0 rolling arithmetic (matches the displayed equations), R1
  Wilder/RMA (reference), R2 EMA. The selected variant *replaces* the default
  so a run never carries two RSI columns.
* **Lookback**: 30 is AUTHOR-CONFIRMED; 10/20/40/60 are validation-only
  sensitivity, explored only after training length and features.
* **Scalers**: Standard / MinMax / Robust, fit on **TRAIN only**; plus a
  controlled raw-volume vs `log1p(volume)` preprocessing comparison.
* **L2**: the paper has an L2 term but never states lambda; search
  0 / 1e-6 / 1e-5 / 1e-4 / 1e-3.
* **Calibration**: none / temperature / Platt / isotonic, selected on pre-2022
  validation Brier/ECE, fitted on validation only and firewall-checked.
* **Class weighting**: ordinary BCE vs `BCEWithLogitsLoss(pos_weight=…)`, with
  `pos_weight` derived from **TRAIN labels only**.
* **Seeds**: 11/23/42/73/101, reported as mean/std on validation.

## Brier definition (pinned)

```
Brier = mean((p_up - y)^2),  y in {0,1}
```

`p_up` is the probability of UP. It is **not** a directional confidence after
flipping DOWN forecasts. A regression test asserts both the identity and that
the flipped variant gives a different (wrong) number.

## Experiment ledger

`results/reproduction_recovery/experiment_ledger.csv` — append-only, 31 columns.
Poor experiments are **never** deleted. Every search row must carry
`test_evaluated = false`; the ledger refuses to append a row claiming
otherwise while the firewall is active, because such a row would mean the
firewall leaked and the ledger is untrustworthy.

## Selection policy

Rank on pre-2022 validation only, using accuracy, F1, Brier and ECE (plus
cross-sectional P@3 where enough stocks/dates exist). Prefer configurations
that beat majority/random consistently, perform across **all three** search
windows, behave stably across stocks, have lower Brier, and do not depend on one
lucky seed. Do not cherry-pick a single stock, and do not select by distance to
the published numbers.

## Freeze

`scripts/freeze_recovered_config.py` records the chosen config's SHA-256, git
commit, dataset variant and manifest hash, universe id, validation metrics and
supporting experiment ids into
`results/reproduction_recovery/frozen_config_manifest.json`.

**After that file exists, no parameter may change.** Any edit changes the hash
and the final run refuses to start.

## Final test protection

`scripts/run_recovered_paper.py` refuses unless **all** of:

1. the frozen manifest exists,
2. the config SHA-256 matches it,
3. `FINAL_TEST=1` is set explicitly.

Only then does it run the two paper folds (2022 and 2023).

## The 2000–2015 data does not help Table II

The workbook begins ~2000 for 37 of 50 securities, but Equation 21's training
starts in **2016**, so the extra 2000–2015 rows **do not enter the paper
folds**. They are useful for reproducing the workbook, historical sensitivity,
and the separate 85/15 split investigation (see
`docs/PAPER_UNIVERSE_RECOVERY.md`).

## The 85/15 protocol is forensic only

The publication also describes an earlier 85/15 chronological split. It is
evaluated separately under `forensic_diagnostics/`, labelled **FORENSIC /
ALTERNATE PAPER PROTOCOL**, and never mixed into the official walk-forward
result. It must not be selected merely because its test metric is nearer
0.815 — that would be tuning on the answer, which is the thing this whole
framework exists to prevent.

## Commands

```bash
# audit everything (read-only, safe at any time)
uv run python scripts/recovery_audit.py --print-plan

# STAGE 0
uv run python scripts/run_reproduction_search.py \
    --config configs/reproduction_search/paper_snapshot_2025_unadjusted_perf.yaml \
    --stage 0

# dry-run one search point (firewall active, no training)
uv run python scripts/run_reproduction_search.py \
    --config configs/reproduction_search/paper_snapshot_2025_unadjusted_perf.yaml \
    --search-fold SEARCH_FOLD_C --training-length T30 --feature-family F2 --dry-run

# freeze the chosen configuration
uv run python scripts/freeze_recovered_config.py \
    --config configs/recovered_paper.yaml \
    --dataset-variant unadjusted \
    --universe-id PAPER_FIXED_NIFTY50_2025_11_04_CANDIDATE

# FINAL test - the only thing that may score 2022/2023
FINAL_TEST=1 uv run python scripts/run_recovered_paper.py \
    --config configs/recovered_paper.yaml --device auto --run-id recovered_paper_final
```

## The Stage-A pilot (what actually runs first)

Because the epoch budget is author-confirmed, Stage A is no longer an epoch
search. It is entered through a controlled pilot, run in two generations. Both
generations share the same 8 stocks and the same 3 folds, so every epoch budget
is measured on identical data.

**Generation 1 — epoch diagnostics (retained, relabelled).** These predate the
author confirmation and are kept because they quantify what the confirmed budget
costs. They are diagnostics, **not** candidates:

| Config | Training length | Features | Status |
|---|---|---|---|
| **P0** | `T3_RECONSTRUCTION_SHORTCUT` (3 epochs) | F1 | reconstruction shortcut; a control |
| **P1** | `T100_DIAGNOSTIC` (100 epochs) | F1 | recovery diagnostic only |
| **P2** | `T100_DIAGNOSTIC` (100 epochs) | F2 | recovery diagnostic only |

**Generation 2 — the author-confirmed pilot (primary).** Epoch budget fixed at
the confirmed 10; the open question is the *feature set*, which is still not
completely established because the original feature list was lost:

| Config | Training length | Features | Question it answers |
|---|---|---|---|
| **P10-F1** | `T10_AUTHOR_CONFIRMED` (10 epochs) | F1 | does the compact methodology reconstruction discriminate? |
| **P10-F2** | `T10_AUTHOR_CONFIRMED` (10 epochs) | F2 | do the paper's named features (SMA5/SMA20/SMA5−SMA20, Bollinger bands and %B, OBV) help? |

This is **2 configurations × 3 folds × 8 stocks = 48 fits**
(`scripts/run_stage_a_pilot.sh --generation author-confirmed`).

The epoch budget is **held fixed** across F1 and F2 on purpose. Because F1 and
F2 are compared at exactly 10 epochs, the F1→F2 difference is attributable to
the features alone and cannot be confounded by a training-length difference.

### A diagnostic win is not a selection criterion

T100 and T3 are reported alongside T10 for completeness, and the comparison is
reported honestly. But if T100 outperforms T10, the correct response is **not**
to select T100. The objective is faithful reconstruction of the original
implementation, and the author has confirmed 10 epochs. A longer budget
performing better would be evidence that the *reconstruction* is not yet
faithful in some other respect — a forensic finding to investigate, not a
licence to substitute a different schedule. **T10 remains the primary candidate
unless new author evidence says otherwise.**

Everything else is held fixed: lookback 30, standard scaler, raw volume, A0
(2×64), dropout 0.2, weight decay 1e-4, no class weighting, **calibration none**,
seed 42. The stocks are RELIANCE, TCS, INFY, HDFCBANK, ITC, LT, SUNPHARMA and
TATASTEEL.

Calibration is `none` for all of Stages A–C so every validation metric is
computed from the **raw** `p(up)`. Comparing calibration methods requires a
calibrated score, which would confound the comparison with everything else, so
it is deferred to Stage D (§ below).

`scripts/summarize_stage_a_pilot.py` reduces the pilot to per-fold and
cross-fold accuracy/F1/Brier/ECE, the majority baseline, cross-sectional P@3,
the T100 best-epoch distribution (what fraction exceed 3 and 10 epochs), a
P0→P1 and P1→P2 diagnostic, and a signal verdict. It reads **only** pilot
artifacts; it never imports `PAPER_REFERENCE` and never reads 2022/2023, so the
pilot cannot become a tuning target for the published numbers.

### Stage-D calibration is nested, not reported on the same data

Stage D fits a calibrator on validation scores and then reports how well that
calibrator works. Doing both on the same validation set would report the
calibrator's *training* error and make every method look perfect. So the
validation window is split chronologically into an earlier **fit 60%** and a
later **score 40%** (`temporal_calibration_split`), the calibrator is fit on the
former and evaluated on the latter, and the split is enforced disjoint by test.
The official 2022 validation and 2023 test windows stay untouched for the final
calibration refit.

## Late-listed constituents

The paper-snapshot universe is the official NIFTY 50 as of 2025-11-04, and
several of its members listed far later than the 2016 training start. A ticker
with no price history before its listing date has no features to build at that
date, and the usual fix — padding with synthetic or forward-filled history —
would fabricate returns and leak future information backwards in time.

The rule is therefore: **each fold uses only the tickers that actually have
sufficient history for that fold's training window**, and the excluded tickers
are reported explicitly rather than silently dropped. The minimum-history
requirement is the fold's `lookback` plus enough sessions to form the feature
window, so a ticker that only lists in 2024 contributes nothing to a fold whose
training window ends in 2018 but may legitimately appear in later folds. The
`ticker_subset` column in the experiment ledger records exactly which tickers
each run actually scored, so per-ticker coverage is always auditable.
