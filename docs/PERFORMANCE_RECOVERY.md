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

**STAGE 0 is a hard gate.** If the model cannot overfit a tiny TRAIN subset, or
if shuffled TRAIN labels do not fall to chance, the search stops and the
pipeline is diagnosed. Numbers from a broken pipeline are uninterpretable.

### Training-length recovery is the first substantive experiment

The publication specifies Adam lr 1e-3, beta1 0.9, beta2 0.999, batch 64 and
early-stopping **patience 10**, but states **no epoch cap**. The reproduction's
`max_epochs: 3` was a budget choice, not a property of the paper. If best
epochs cluster in the 12–25 range, that immediately demonstrates the
three-epoch reconstruction was undertrained.

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
