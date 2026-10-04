# MODEL V2 — scientific status, design and boundaries

> **MODEL V2 IS NOT THE ORIGINAL PAPER MODEL.**
>
> **Classification: `NEW_EXPERIMENTAL_ARCHITECTURE`.**
>
> This classification is recorded machine-readably in every V2 config, manifest,
> checkpoint, ledger row, summary and report (`v2/__init__.py:MODEL_V2_CLASSIFICATION`,
> `is_original_paper_model: false`). It is not a stylistic note: V2 must never be
> presented as a reconstruction of the published model, and no V2 number is
> compared against a published number anywhere in the code
> (`tests/unit/test_v2_system.py::test_v2_never_imports_the_paper_reference`).

---

## 1. Why V2 exists at all

The faithful paper reproduction is complete and its verdict is negative. Its
per-security Attention-LSTM, trained at the author-confirmed schedule, stayed
near **52–53 % mean pre-2022 accuracy** across the three recovery folds
(`results/reproduction_recovery/stage_a_pilot_summary.json`, e.g. T10-F1
cross-fold mean accuracy 0.5168 against a mean majority baseline of 0.5282), and
legitimate structural diagnostics did not recover the originally reported
performance. That work is frozen and is **not** revisited here.

Three facts therefore motivate a different system rather than more search:

1. The faithful per-stock Attention-LSTM pre-2022 validation remained near
   approximately 52–53 %.
2. Legitimate structural diagnostics (data adjustment, pooled versus per-stock,
   split protocol, target alignment, aggregation, confidence filtering,
   deliberately invalid leakage probes) did **not** recover the originally
   reported performance.
3. The user has prior dataset-specific evidence that **LSTM outperforms TCN** on
   this dataset.

Consequently V2 keeps the LSTM temporal inductive bias and changes the
representation and the architecture around it: one **shared** model across many
securities, **stationary** features instead of raw price levels, **context**
derived from the cross-section itself, **Transformer** attention over the same
window, **multi-task** supervision, and two optional adaptation stages.

### 65 % is an aspiration, not a selection criterion

65 % is **not** a tuning objective, not a gate, and not a requirement for success.
Architecture selection uses mean macro-ticker directional accuracy across the two
development folds, with ROC-AUC, Brier, F1 and fold consistency as secondary
criteria (`scripts/summarize_v2_dev.py:select_winner`). The only place 65 % appears
as a *reported* quantity is the explicitly labelled `SELECTIVE_65_COVERAGE`
coverage measurement, which requires at least 10 % coverage and 200 observations
before it may be quoted at all.

---

## 2. What is preserved, and what is separate

**Immutable and untouched by V2**

| Artefact | Status |
|---|---|
| `configs/paper.yaml` | unchanged |
| `src/agentic_forecaster/models/attention_lstm.py` | unchanged |
| `agents/model_agent.py`, `data/agent.py` (paper behaviour) | unchanged |
| `features/engineer.py` (paper feature engineer) | unchanged |
| `training/trainer.py` (paper binary trainer) | unchanged |
| `results/reproduction_recovery/**` | unchanged |
| existing datasets | read-only |
| existing final-test firewall (`recovery/firewall.py`) | unchanged and still active |

**Separate V2 namespace**

```
src/agentic_forecaster/v2/     features  context  store  dataset  model  losses
                               trainer   metrics  checkpoint  meta   experiment
                               firewall  sectors  ledger  sanity
configs/v2/                    base + one config per ablation step + sector_map
scripts/                       build_v2_context  run_v2_experiment  run_v2_sanity
                               summarize_v2_dev  run_v2_lockbox  run_v2_dev_pilot.sh
results/v2/                    ledger, dev summary, dev report, selection, sanity
$AGENTIC_OUTPUT_ROOT/v2/       runtime experiment directories + checkpoints
$AGENTIC_PROCESSED_DATA_ROOT/v2/   sector map provenance + context store
```

No V2 module imports `paper_reference`; no V2 artifact is written into
`results/reproduction_recovery/`.

### One forensic fix, no forensic re-run

`recovery/forensics.py:build_target_day_leak` verified kept samples with
`targets[j]` where `j` indexes the *kept* list and `i` the *original* list. Once
any earlier sample was dropped the two diverge, so every later kept sample was
audited against the wrong target date. The verification loop now uses
`targets[i]`, consistent with the returned `target_dates`. The regression test
`tests/unit/test_forensics_target_day_leak.py` pins the behaviour with an
intentionally missing target-date vector in the **middle** of the sample list, so
`j != i` for the samples that follow it. **No forensic experiment was re-run and
no forensic result was modified.**

---

## 3. Data, universe and prediction time

| Item | V2 choice |
|---|---|
| Source | already-downloaded `yfinance_paper_snapshot_2025_11_04` (no new download) |
| Primary variant | **adjusted** daily bars (corporate-action adjusted) |
| Mixing | adjusted and unadjusted bars are never mixed in one experiment |
| Universe | `configs/nifty50_paper_snapshot_2025_11_04.yaml` (fixed 50-name snapshot) |
| Store cap | **2021-12-31**, hard |
| Supervised development securities | RELIANCE, TCS, INFY, HDFCBANK, ITC, LT, SUNPHARMA, TATASTEEL |
| Context population | every available security of the fixed universe |
| Sequence length | 60 (the paper's 30-day window is untouched) |

**Adjusted is primary because V2 is a new model, not a reconstruction**, and
stationary return-derived features benefit from corporate-action-adjusted prices.
The unadjusted variant stays available as a later sensitivity check.

### Survivorship bias — stated, not hidden

This remains a **fixed-universe reconstruction**. The 50 names are the composition
around the publication's data-collection date, so projecting them backwards
through 2005–2021 carries survivorship bias: names that failed, were merged away
or were not index members earlier are absent. V2 does **not** claim to be a
historical point-in-time constituent universe, and no V2 result should be read as
a backtest of a survivorship-bias-free index. The same caveat applies to the
static sector map, which is today's industry classification applied backwards.

### Prediction time

**After the close on day `t`, predict the next actual trading day `t+1`.** Every
input at `t` may use date-`t` closes, volumes, breadth and cross-sectional ranks.
No input may ever contain a `t+1` value. `target_date` is retained explicitly on
every sample, and a sample belongs to a split only if **both** `origin_date` and
`target_date` fall inside it — a 60-day window may legitimately begin in December
to predict a January observation, while scaler fitting, label fitting and loss
computation use TRAIN samples only.

### No new information sources in V2.0

No news, sentiment, options, FII/DII, VIX, macro or alternative data is used. The
first V2 experiment is therefore attributable to **architecture + representation**,
not to new data.

---

## 4. Stationary features instead of price levels

The paper model consumes raw OHLCV plus indicators, which is legitimate for one
model per stock and **not** legitimate for one shared model across securities: a
level of 1500 and a level of 30 are not comparable quantities, and a shared model
would learn security identity from the price axis. V2 therefore has its own
feature engineer (`v2/features.py`, 27 features) built only from
return/range/normalised-indicator quantities, every one causal and bounded.

Indicator warm-up produces NaN and **is never forward-filled**; a sample whose
required history does not exist is dropped, never padded.

## 5. Context from the cross-section, with leave-one-out discipline

A stock's own return is inside every market aggregate, so a market feature fed
back to that stock partly restates its own input. Every market and sector
statistic supplied to stock *i* therefore **excludes *i***:

* same-date statistics use the `sum/count` and `sum/sumsq` identities
  (`(S − r_i)/(C − 1)`, and the matching leave-one-out standard deviation);
* rolling statistics first build each stock's **own** leave-one-out proxy series,
  then roll that proxy causally over time;
* when the remaining group is too small the value is **unavailable (NaN)**, never
  invented.

`market_available_count` is recorded for every date. Missing history is never
fabricated: a security with no bar on date *t* simply does not contribute.

> Note on one schema choice: a leave-one-out sector **dispersion** needs at least
> three available sector members (0 degrees of freedom otherwise). With the seven
> broad sectors used here, one sector has three members, so the value is available
> only part of the time and the affected samples are dropped by the usual rule.
> This is stated rather than papered over with an imputation.

## 6. Three targets

| Target | Definition | Weight |
|---|---|---|
| `y_direction` | `int(close[t+1] > close[t])` — the headline classification | 1.00 (`BCEWithLogits`) |
| `y_return` | `clip(log(close[t+1]/close[t]) / realized_vol_20[t], −5, +5)` | 0.50 (`SmoothL1`) |
| `y_rank` | percentile rank of the next-day return among the securities available on the **target** date | 0.25 (`SmoothL1`) |

`y_rank` is a **label**: it legitimately uses `t+1` returns and can never become an
input. The clipping bound was fixed before any experiment.

## 7. Architecture and the ablation ladder

```
stock sequence   -> unidirectional LSTM (2 x 96, dropout 0.20)      H_lstm [B,T,96]
context sequence -> Linear(F,32) + GELU + LayerNorm                  H_ctx  [B,T,32]
[H_lstm, H_ctx]  -> Linear(128,64) + LayerNorm + GELU
+ sinusoidal positional encoding (>= 128 positions)
-> CAUSAL TransformerEncoder (2 layers, d_model 64, 4 heads, FF 128, GELU, norm_first)
-> learned temporal attention pooling                                 pooled [B,64]
+ Linear(96,64) applied to the LAST LSTM hidden state, then LayerNorm -> temporal [B,64]
+ ticker embedding 16 | sector embedding 8 | regime embedding 16
-> fusion MLP -> z [B,64]
-> FiLM (optional) -> residual adapter (optional) -> task heads
```

* The **full** LSTM state sequence is returned, never only the final state.
* Attention is masked **causally** even though the whole window is available at the
  origin: it keeps the temporal representation auditable
  (`tests/unit/test_v2_model.py::test_transformer_is_actually_causal`).
* The **LSTM residual is explicit**, so the architecture can fall back towards the
  LSTM representation if the Transformer contributes little.
* V2 is **one shared model**. Security identity enters only through a 16-d
  ticker embedding (plus sector and regime embeddings), never a price level and
  never a separate parameter set.

| Variant | Adds | Purpose |
|---|---|---|
| **V2-A** | shared LSTM + ticker embedding + direction head | shared-LSTM baseline |
| **V2-B** | causal Transformer + attention pooling + LSTM residual | Transformer's contribution |
| **V2-C** | market/sector/relative/rank context + sector + regime encoder | **information/context** contribution |
| **V2-D** | normalised-return and cross-sectional-rank heads | richer supervision |
| **V2-E** | FiLM ticker/sector/regime conditioning | conditional adaptation |
| **V2-F** | residual adapter + Reptile-style head/adapter meta-learning | fast adaptation |

Nothing else was added: no TCN, no hyperparameter grid, no extra variant.

## 8. FiLM is not meta-learning

**V2-E (FiLM)** is *conditional adaptation*: gamma and beta are a deterministic,
bounded function of the identity/regime embeddings (`gamma = 1 + 0.1·tanh(·)`,
`beta = 0.1·tanh(·)`), learned jointly with everything else.

**V2-F (Reptile)** is *meta-learning*: a task-specific parameter set is obtained by
inner gradient steps on that task's support set, and the shared meta-initialisation
is then moved towards those adapted parameters (Reptile). Its honest label is
`REPTILE_STYLE_HEAD_ADAPTER`. It is **not** MAML and **not** full-network
meta-learning: only the residual adapter and the three heads are adapted; the LSTM,
the Transformer, the context encoder and the embeddings stay frozen. No
third-party meta-learning dependency is used — PyTorch only.

A meta task is **not** one security. It is *security + chronological historical
episode*: 120 support samples, then the immediately following 20 query samples,
stride 20, built from TRAIN data only. At validation time each security is adapted
from the last 120 labelled TRAIN samples preceding the validation window, using 3
support updates and **no validation label**.

## 9. Two firewalls

| Firewall | Rule | Enforced in |
|---|---|---|
| Hard paper-test | any target date ≥ **2022-01-01** raises immediately | `v2/firewall.py`, plus the store is *capped* at 2021-12-31 so such a label cannot even exist |
| V2 lockbox | 2021 may not be scored during development; only `run_v2_lockbox.py` with `V2_LOCKBOX=1` may, and never 2022+ | `v2/firewall.py`, `load_run_config`, the dev script |

The 2022/2023 paper test years are untouched by this programme. No V2 script sets
`FINAL_TEST`, none invokes `run_recovered_paper.py`, and every ledger row records
`test_2022_2023_evaluated = false` (the ledger refuses to append a row claiming
otherwise).

## 10. Reporting discipline

* Direction: micro accuracy, macro-ticker accuracy, per-ticker accuracy, F1,
  balanced accuracy, ROC-AUC, Brier, ECE, train-majority baseline.
* Coverage: accuracy and F1 at 100/75/50/30/20/10 %, ranked by
  `|sigmoid(logit) − 0.5|`, each point carrying its own `n` and coverage.
* Selective accuracy is **never** called overall accuracy. A coverage figure is
  only reported with ≥ 10 % coverage **and** ≥ 200 observations.
* P@3 up / P@3 down, computed per date.
* Return: MAE, RMSE, Pearson, Spearman (normalised and de-normalised).
* Rank: MAE, daily Spearman rank IC, mean rank IC — a **negative IC is reported**,
  not hidden.
* Multi-head agreement (V2-D onward) is **analysis-only** and never selects a model.
* Complexity per variant: parameter counts, peak GPU memory, training time, best
  epoch — so that tiny accuracy gains can be judged against their cost.

## 11. Interpretation labels

`NO_SIGNAL` · `WEAK` · `PROMISING` · `STRONG` (≥ 57 % all-sample accuracy on the
2021 lockbox with broad ticker stability) · `EXCEPTIONAL` (≥ 60 % on the same
basis), plus `SELECTIVE_65_COVERAGE` when a 65 % selective accuracy is available at
a meaningful coverage. **65 % is not required for success.**

## 12. Interpretation boundaries

* One dataset, one universe, one fixed-horizon task, one shared model.
* Eight supervised securities per run (the full-50 run is future work and will need
  the ticker-balanced sampler that is already implemented and enabled).
* No transaction costs, no position sizing, no portfolio construction: this is a
  forecasting study, not a profitability claim.
* The context gate is a **diagnostic** about whether same-data context carries
  signal, not evidence of economic value.
---

## 13. The PRE-COVID EXPERIMENTAL REGIME (separate track)

`PRE_COVID_EXPERIMENTAL_REGIME` — **not** a claim about causation.  This track
establishes one thing only: whether a model trained and evaluated entirely
before 2020 behaves differently.  It does **not** claim that COVID caused any
earlier failure; that would be a causal claim this design cannot support.

Everything about the 2020+ period lives in a **separate future task**.  Nothing
from 2020 onward is consumed here: not for features, market/sector context,
cross-sectional ranks, scaler fitting, training, validation, early stopping,
architecture selection, meta-learning episodes, confidence selection, metrics or
model selection.  The PRE-COVID processed store is physically capped at
`2019-12-31`, and `PostCovidDataAccessError` fires at the point of access.

| | ordinary V2 | PRE-COVID |
|---|---|---|
| horizon | … 2021-12-31 | … **2019-12-31** |
| store | `…/v2/context_store` | `…/v2/pre_covid/context_store` |
| configs | `configs/v2/*.yaml` | `configs/v2/pre_covid/*.yaml` |
| ledger | `results/v2/experiment_ledger.csv` | `results/v2/pre_covid/experiment_ledger.csv` |
| report | `results/v2/V2_DEV_REPORT.md` | `results/v2/pre_covid/PRE_COVID_V2_REPORT.md` |
| selection folds | 2019, 2020 | **2017, 2018** |
| lockbox | 2021 (`V2_LOCKBOX=1`) | **2019** (`PRECOVID_LOCKBOX=1`) |

A `2019-12-31` origin paired with a `2020-01-01` target **cannot exist**: the
target frame itself stops at 2019-12-31, and a post-boundary row is rejected
rather than filtered.

### Survivorship bias label

Every PRE-COVID number carries
`SURVIVORSHIP_BIASED_FIXED_UNIVERSE_RESEARCH_TRACK`.  The universe is a fixed,
later-reconstructed constituent list projected backwards; reports say
"available securities from the reconstructed fixed universe", never "the NIFTY-50
constituents on that historical date".  This limitation does not invalidate the
architecture experiment, but it travels with every result.

### Two populations, kept separate

* **SUPERVISED** — only securities passing every eligibility rule (≥ 1000 usable
  TRAIN samples through 2016, ≥ 180 usable samples in each of 2017, 2018 and
  2019, causal features throughout).  The count is derived and frozen in
  `configs/v2/pre_covid/supervised_universe.yaml` **before** any training.
* **CONTEXT** — every security of the reconstructed fixed universe with a valid
  bar on date `t`, so a later listing contributes from the day it exists and is
  never backfilled.

---

## 14. Roadmap: the future post-2019 EVENT / REGIME / SENTIMENT MODEL

**DOCUMENT ONLY.  Nothing in this section is implemented, downloaded or trained
by the PRE-COVID programme.**  The 2020+ period is a separate task with separate
data provenance.

Motivation: the PRE-COVID regime behaves differently from the later regime.  The
2020+ model is a *different* problem — event-driven and regime-aware — not a
continuation of the pre-2020 forecasting problem, and it must be evaluated as
such.

Potential **point-in-time** inputs, each of which must be timestamp-clean
(an observation is usable only from the moment it became public, never
back-dated):

* news sentiment (timestamped article flow, not article date)
* India VIX (and comparable implied-volatility series)
* global index stress (S&P/Nasdaq/other EM drawdowns at the close of `t`)
* realized market volatility and breadth at higher frequency
* macro-event indicators (policy decisions, CPI, GDP prints, with release times)
* pandemic / event indicators (lockdowns, restrictions, wave dates)
* sector-specific news and event signals

Engineering requirements for that future track:

1. an event store with an explicit `available_at` timestamp per observation, and
   a unit test that no observation can be consumed before its `available_at`;
2. a regime-aware evaluation protocol (the 2020+ window must not be used to tune
   the pre-2020 architecture, and vice versa);
3. its own firewall and ledger, exactly as the PRE-COVID track has its own;
4. an honest comparison against the PRE-COVID model on the SAME dates, reported
   as two regimes rather than as one time series.

---

## 15. The MULTI-HORIZON track: does a longer horizon help?

**Separate track.** `configs/v2/multi_horizon/`, `results/v2/multi_horizon/`,
`$AGENTIC_OUTPUT_ROOT/v2/multi_horizon/`,
`$AGENTIC_PROCESSED_DATA_ROOT/v2/multi_horizon/`. The ordinary V2 store, the
PRE-COVID store, both their ledgers, the paper reproduction and the forensic
results are never written by this track.

The PRE-COVID programme found only a weak one-day edge, so this track changes
**only the forecast horizon** and asks:

> "Does the existing PRE-COVID price-derived dataset contain more predictable
> directional information at 3-, 5- or 10-trading-day horizons than at the
> one-day horizon?"

| objective | horizon | label |
|---|---|---|
| `ABS_DIR_1D_CONTROL` | 1 | `y = 1 if log(Close[t+1]/Close[t]) > 0` (**CONTROL**) |
| `ABS_DIR_3D` | 3 | `y = 1 if log(Close[t+3]/Close[t]) > 0` |
| `ABS_DIR_5D` | 5 | `y = 1 if log(Close[t+5]/Close[t]) > 0` |
| `ABS_DIR_10D` | 10 | `y = 1 if log(Close[t+10]/Close[t]) > 0` |

`H` counts **future trading observations** of that security, taken from its own
ordered trading rows, so weekends and exchange holidays are skipped by
construction; `date + timedelta(days=H)` is never used. `ABS_DIR_1D_CONTROL` is
asserted in code to reproduce the existing `y_direction` **exactly** (label, target
date and return) on all 198,746 shared keys. A 10-day label therefore describes
whether `Close[t+10]` is above or below `Close[t]`, and is never reported as
"next-day accuracy".

Five chronological development folds (2014-2018) with a sealed 2019 lockbox; the
absolute final allowed target date is `2019-12-31` and a December-2019 origin whose
multi-day target would land in 2020 is dropped. Stage 1 screens fixed Logistic and
HistGradientBoosting models; stage 2 (shared LSTM, LSTM+Transformer) runs **only**
for a horizon that screen-passes, so no compute is spent on a horizon the evidence
has already rejected.

**Outcome: no 3D/5D/10D horizon screen-passed.** Mean ROC-AUC stayed between 0.505
and 0.519 and mean balanced accuracy at ~0.50 for every horizon and both models;
accuracy above the train-majority baseline at 5D/10D is class imbalance, not
directional skill. Recommended next action: `ADD_EXOGENOUS_INFORMATION`. See
`results/v2/multi_horizon/MULTI_HORIZON_REPORT.md`.
