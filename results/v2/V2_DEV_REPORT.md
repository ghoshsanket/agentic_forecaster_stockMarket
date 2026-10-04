# MODEL V2 DEVELOPMENT REPORT

**MODEL V2 IS NOT THE ORIGINAL PAPER MODEL.** Classification: `NEW_EXPERIMENTAL_ARCHITECTURE`. See `docs/MODEL_V2.md`.

- generated: `2026-10-04T04:54:11.659498+00:00`
- development folds: V2_DEV_FOLD_A, V2_DEV_FOLD_B
- seed: 42
- 2022/2023 labels accessed: **NO**
- paper reference table used for comparison: **NO**

## 0. Stage outcome

- CONTEXT_SIGNAL_GATE: **FAILED** -> the staged programme STOPPED here.
- V2-D, V2-E and V2-F were **deliberately not run**: same-data context did not produce enough signal to justify multi-task or meta complexity.
- Consequence: **no winner was frozen, no seed-stability run was made, and the 2021 lockbox was NOT opened.** 2021 remains sealed.
- 'MATERIAL' / 'SMALL' in the delta tables below are the pre-registered decision rules (>= 0.5 pp mean improvement with no severe per-fold drop), not claims about statistical significance.

## 1. Side-by-side development metrics

| variant | macro acc (mean A+B) | micro acc | F1 | AUC | Brier | ECE | baseline | delta vs baseline | P@3 up | P@3 down |
|---|---|---|---|---|---|---|---|---|---|---|
| V2-A | 0.5056 | 0.5056 | 0.6302 | 0.4883 | 0.2502 | 0.0130 | 0.5091 | -0.0034 | 0.5130 | 0.4936 |
| V2-B | 0.5205 | 0.5205 | 0.6214 | 0.5035 | 0.2498 | 0.0086 | 0.5091 | 0.0115 | 0.5168 | 0.5004 |
| V2-C | 0.5206 | 0.5206 | 0.5511 | 0.5242 | 0.2495 | 0.0188 | 0.5104 | 0.0103 | 0.5161 | 0.4964 |

### Component deltas (mean macro accuracy across A+B)

| step | delta | note |
|---|---|---|
| V2-B vs V2-A | 0.0149 | MATERIAL |
| V2-C vs V2-B | 0.0001 | SMALL |
| V2-D vs V2-C | n/a | V2-D or V2-C not available |
| V2-E vs V2-D | n/a | V2-E or V2-D not available |
| V2-F vs V2-E | n/a | V2-F or V2-E not available |

## 2. Per-fold results

| variant | fold | macro | micro | F1 | AUC | Brier | baseline | delta | best epoch | params | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V2-A | V2_DEV_FOLD_A | 0.5093 | 0.5093 | 0.6233 | 0.5216 | 0.2499 | 0.5095 | -0.0002 | 4.0000 | 156,370 | 33.0219 |
| V2-A | V2_DEV_FOLD_B | 0.5020 | 0.5020 | 0.6370 | 0.4550 | 0.2506 | 0.5086 | -0.0066 | 1.0000 | 156,370 | 25.2618 |
| V2-B | V2_DEV_FOLD_A | 0.5181 | 0.5181 | 0.5559 | 0.5186 | 0.2500 | 0.5095 | 0.0086 | 5.0000 | 223,314 | 44.2007 |
| V2-B | V2_DEV_FOLD_B | 0.5230 | 0.5230 | 0.6868 | 0.4885 | 0.2496 | 0.5086 | 0.0144 | 1.0000 | 223,314 | 27.5774 |
| V2-C | V2_DEV_FOLD_A | 0.5248 | 0.5248 | 0.5287 | 0.5200 | 0.2499 | 0.5109 | 0.0139 | 3.0000 | 230,746 | 45.0402 |
| V2-C | V2_DEV_FOLD_B | 0.5165 | 0.5165 | 0.5734 | 0.5283 | 0.2492 | 0.5098 | 0.0067 | 8.0000 | 230,746 | 79.0929 |

## 3. Context signal gate

- evaluated: True  **passed: False**
- criterion_a: `V2-C mean macro >= 0.55 AND beats baseline in BOTH dev folds` -> passed=False
- criterion_b: `V2-C improves mean macro over V2-B by >= 0.015 AND baseline delta positive in BOTH folds AND AUC >= 0.54` -> passed=False
- meaning: same-data context has NOT produced enough signal to justify multi-task / meta complexity; the programme stops here

## 4. Adaptation and meta-learning

- meta learning justified: **False**

## 5. Per-ticker validation accuracy

| variant | fold | HDFCBANK | INFY | ITC | LT | RELIANCE | SUNPHARMA | TATASTEEL | TCS |
|---|---|---|---|---|---|---|---|---|---|
| V2-A | V2_DEV_FOLD_A | 0.5165 | 0.5331 | 0.5000 | 0.4876 | 0.4793 | 0.5124 | 0.4917 | 0.5537 |
| V2-A | V2_DEV_FOLD_B | 0.4960 | 0.5400 | 0.4680 | 0.4800 | 0.4880 | 0.5000 | 0.5040 | 0.5400 |
| V2-B | V2_DEV_FOLD_A | 0.5083 | 0.5124 | 0.5083 | 0.5579 | 0.4876 | 0.5041 | 0.5124 | 0.5537 |
| V2-B | V2_DEV_FOLD_B | 0.5320 | 0.5400 | 0.4680 | 0.5160 | 0.5160 | 0.5200 | 0.5480 | 0.5440 |
| V2-C | V2_DEV_FOLD_A | 0.5413 | 0.5372 | 0.5702 | 0.5331 | 0.4917 | 0.5124 | 0.5041 | 0.5083 |
| V2-C | V2_DEV_FOLD_B | 0.5280 | 0.5480 | 0.4960 | 0.4640 | 0.5040 | 0.5480 | 0.5280 | 0.5160 |

## 6. Complexity per variant

| variant | parameters | peak GPU memory (MB) | training seconds | best epoch |
|---|---|---|---|---|
| V2-A | 156,370 | 297 | 33 | 4 |
| V2-B | 223,314 | 298 | 44 | 5 |
| V2-C | 230,746 | 301 | 45 | 3 |

## 7. Accuracy versus coverage

### V2-A / V2_DEV_FOLD_A

| coverage retained | n | accuracy | F1 |
|---|---|---|---|
| 1.0000 | 1,936 | 0.5093 | 0.6233 |
| 0.7500 | 1,452 | 0.5158 | 0.6511 |
| 0.5000 | 968 | 0.5103 | 0.6624 |
| 0.3001 | 581 | 0.5267 | 0.6806 |
| 0.1999 | 387 | 0.5297 | 0.6894 |
| 0.1002 | 194 | 0.5619 | 0.7195 |

### V2-A / V2_DEV_FOLD_B

| coverage retained | n | accuracy | F1 |
|---|---|---|---|
| 1.0000 | 2,000 | 0.5020 | 0.6370 |
| 0.7500 | 1,500 | 0.4960 | 0.6480 |
| 0.5000 | 1,000 | 0.4840 | 0.6471 |
| 0.3000 | 600 | 0.4783 | 0.6471 |
| 0.2000 | 400 | 0.4525 | 0.6231 |
| 0.1000 | 200 | 0.4200 | 0.5915 |

### V2-B / V2_DEV_FOLD_A

| coverage retained | n | accuracy | F1 |
|---|---|---|---|
| 1.0000 | 1,936 | 0.5181 | 0.5559 |
| 0.7500 | 1,452 | 0.5186 | 0.5568 |
| 0.5000 | 968 | 0.5217 | 0.5299 |
| 0.3001 | 581 | 0.5026 | 0.4578 |
| 0.1999 | 387 | 0.5090 | 0.3581 |
| 0.1002 | 194 | 0.4691 | 0.0000 |

### V2-B / V2_DEV_FOLD_B

| coverage retained | n | accuracy | F1 |
|---|---|---|---|
| 1.0000 | 2,000 | 0.5230 | 0.6868 |
| 0.7500 | 1,500 | 0.5187 | 0.6831 |
| 0.5000 | 1,000 | 0.5170 | 0.6816 |
| 0.3000 | 600 | 0.5100 | 0.6755 |
| 0.2000 | 400 | 0.4925 | 0.6600 |
| 0.1000 | 200 | 0.4800 | 0.6486 |

### V2-C / V2_DEV_FOLD_A

| coverage retained | n | accuracy | F1 |
|---|---|---|---|
| 1.0000 | 1,936 | 0.5248 | 0.5287 |
| 0.7500 | 1,452 | 0.5207 | 0.5233 |
| 0.5000 | 968 | 0.5238 | 0.5262 |
| 0.3001 | 581 | 0.5060 | 0.5160 |
| 0.1999 | 387 | 0.5168 | 0.5428 |
| 0.1002 | 194 | 0.5619 | 0.5972 |

### V2-C / V2_DEV_FOLD_B

| coverage retained | n | accuracy | F1 |
|---|---|---|---|
| 1.0000 | 2,000 | 0.5165 | 0.5734 |
| 0.7500 | 1,500 | 0.5200 | 0.5904 |
| 0.5000 | 1,000 | 0.5390 | 0.6168 |
| 0.3000 | 600 | 0.5600 | 0.6480 |
| 0.2000 | 400 | 0.5950 | 0.6798 |
| 0.1000 | 200 | 0.6200 | 0.6984 |

## 8. Maximum meaningful coverage at 60 / 62 / 65 %

A coverage point is reported only with at least 10 % coverage AND at least 200 observations.

| variant | fold | >=60% | >=62% | >=65% |
|---|---|---|---|---|
| V2-A | V2_DEV_FOLD_A | n/a | n/a | n/a |
| V2-A | V2_DEV_FOLD_B | n/a | n/a | n/a |
| V2-B | V2_DEV_FOLD_A | n/a | n/a | n/a |
| V2-B | V2_DEV_FOLD_B | n/a | n/a | n/a |
| V2-C | V2_DEV_FOLD_A | n/a | n/a | n/a |
| V2-C | V2_DEV_FOLD_B | 0.1000 (acc 0.6200, n 200) | 0.1000 (acc 0.6200, n 200) | n/a |


## 10. Interpretation

**NO_SIGNAL (context) / WEAK (architecture programme)**

- best development variant: V2-C at 52.06% mean macro accuracy vs a 51.04% train-majority baseline (+1.03 pp)
- mean ROC-AUC 0.524, mean Brier 0.2495, mean ECE 0.0188
- CONTEXT_SIGNAL_GATE passed: False
- SELECTIVE_65_COVERAGE: not achieved at any meaningful coverage (>=10% coverage and >=200 observations)
- 2021 lockbox: NOT RUN (the context signal gate failed)

Reading: the shared-model programme reproduces the same weak-signal regime as the per-stock reconstruction. The Transformer step is the only measurable improvement, and cross-sectional context adds essentially nothing, which is why the staged programme stopped before multi-task and meta-learning. Nothing here supports a profitability claim, and nothing here approaches the 65% aspiration at any meaningful coverage.

## 11. Answers required by the V2 programme

- **A. Does a shared LSTM outperform the old independent-stock reconstruction?** No. V2-A (shared LSTM) reached 50.56% mean macro accuracy across DEV A+B against a train-majority baseline of 50.91%, and did not beat that baseline in both folds (False). The historical local per-stock reconstruction sits at roughly 52-53%.
- **B. Does Transformer attention improve the shared LSTM?** Yes, marginally: +1.49 pp mean macro accuracy (V2-B 52.05% vs V2-A 50.56%), and V2-B beats the majority baseline in BOTH folds (A=0.0086, B=0.0144). The gain is small and ROC-AUC stays near 0.50 (0.504), so ranking ability is weak.
- **C. Does market/sector/cross-sectional CONTEXT produce the largest gain?** No. The full same-data context changed mean macro accuracy by +0.01 pp (V2-C 52.06% vs V2-B 52.05%), with ROC-AUC moving from 0.504 to 0.524.
- **D. Does multi-task supervision improve DIRECTION accuracy?** NOT TESTED. The CONTEXT_SIGNAL_GATE failed, so V2-D was deliberately not run.
- **E. Does FiLM conditioning help?** NOT TESTED. The gate failed before V2-E.
- **F. Does actual Reptile-style adaptation help over FiLM?** NOT TESTED, and deliberately so: the gate failed and no adaptation signal was established, so the meta stage was not justified.
- **G. Which component contributes the largest directional-accuracy gain?** The Transformer (V2-B over V2-A: +1.49 pp) is the largest measured step; the context block (V2-C over V2-B) contributes +0.01 pp, i.e. essentially nothing.
- **H. Is improvement present in BOTH 2019 and 2020?** V2-B vs majority baseline: A=0.0086, B=0.0144; V2-C vs majority baseline: A=0.0139, B=0.0067. V2-B and V2-C beat the baseline in 2019 and in 2020; V2-A does not.
- **I. Does the winner beat the legitimate train-majority baseline consistently?** Yes: the highest-mean variant (V2-C, 52.06%) beats the train-majority baseline in both development folds by A=0.0139, B=0.0067 -- under 1.5 pp.
- **J. What is the 2021 lockbox result?** NOT RUN. The CONTEXT_SIGNAL_GATE failed, so the staged programme stopped before winner selection, seed stability and the lockbox. 2021 has NOT been read by any V2 script and remains a sealed architecture lockbox.
- **K. What overall accuracy is achieved?** Best development all-sample accuracy: 52.06% mean macro (V2-C) across DEV A+B; per fold 2019=52.48%, 2020=51.65%. No 2021 or 2022+ accuracy exists.
- **L. At what coverage does directional accuracy reach 60 / 62 / 65 %?** 60%: best single fold: B V2-C at 10% coverage (n=200, accuracy 62.00%) | 62%: best single fold: B V2-C at 10% coverage (n=200, accuracy 62.00%) | 65%: not reached: no coverage level with >=10% coverage and >=200 observations reached 65% accuracy on either development fold
- **M. Was ANY 2022/2023 target label accessed?** NO. The V2 feature store is hard-capped at 2021-12-31, the V2 test firewall raises on any target date >= 2022-01-01, and every ledger row records test_2022_2023_evaluated=false.
- **context signal gate** passed=False: same-data context has NOT produced enough signal to justify multi-task / meta complexity; the programme stops here

## 12. Recommended next action

**STOP the architecture search and treat the next step as a data / target question, not a bigger model**

V2-A/B/C all land within ~1.5 pp of a majority baseline with ROC-AUC near 0.50, and the same-data context contributes ~0 pp. Enlarging the model (V2-D multi-task, V2-E FiLM, V2-F meta) would add complexity without evidence that the inputs carry directional signal.

1. verify the signal ceiling first: a ridge / logistic baseline on the same V2 features, scored the same way, to establish whether ANY model can beat the majority baseline on these eight securities in 2019-2020
2. keep the 2021 lockbox sealed; re-open it only after a genuine improvement is demonstrated on DEV A+B
3. if the ceiling check is also flat, treat the task as a volatility / ranking problem rather than a direction problem, and re-target the evaluation metric (rank IC, selective coverage) before any new architecture
4. only after that: run the full 50-security universe, where the cross-sectional context has far more peers and the ticker-balanced sampler already implemented matters
5. 2022 and 2023 remain untouched until an architecture has earned its place

