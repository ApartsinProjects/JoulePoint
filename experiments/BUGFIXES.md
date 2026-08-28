# Defect register

Every bug and design error found during the audit, its root cause, the fix, and how the
result changed. All fixes are applied to the **source** files, so re-running any experiment
reproduces the corrected numbers.

## Code defects

| ID | File | Defect | Root cause | Effect of fix |
|---|---|---|---|---|
| **B1** | `mlperf_experiment.py` | Machine cold start returned `NaN` for both metrics | `rank_metrics` compares cells **within a row**; holding out a whole column leaves one test cell per row, so the `len(js)<2` guard skipped every row | Added `coldstart_metrics`, which ranks the held-out column against the observed columns of the same row. **Result inverted**: hybrid goes from "no measurable benefit" to **78.1% pairwise / 1.04x regret**, against 52.9% / 3.81x with no column information |
| **B2** | `mlperf_experiment.py` | Hybrid lost to a column-mean baseline on load cold start | `fit_hybrid` overwrote **well-estimated measured** column effects with a noisy spec-based regression, even for columns with plenty of data | Specs are now used only where `seen[j]` is false. Task A hybrid MAE 0.161 → **0.130**, pairwise 78.9 → **84.4**, now beating additive |
| **B3** | `mlperf_experiment.py` | A fully held-out row kept `r[i]=0` | `fit_additive` only updates rows with observations | Row effect imputed from observed rows. No effect on within-row ranking (a row constant cancels), corrects absolute error |
| **B4** | `e4_e5_models.py` | DCN-v2 cross scored 82.1%, reported as underperformance | `alpha=200` was arbitrary and over-regularised | Replaced with `RidgeCV`. Score **84.2%** (manual sweep peaks at 87.5% at alpha=0.1) |
| **B5** | `e4_e5_models.py` | MLP scored 78.8%, reported as "worse than a fixed ranking" | A single untuned configuration | Retuned to (128,64), alpha=1.0. Score **86.7%**, which **beats** the fixed ranking. **My original claim was wrong** |
| **B6** | `e6_contention.py` | `p_random` was not reproducible across runs | Seeded from `hash()`, which is salted per interpreter process | Replaced with `zlib.crc32` of the key |
| **B7** | `e6_contention.py` | Dead import and unused variables | Left from an earlier draft | Removed |

## Design errors

| ID | Experiment | Error | Effect of fix |
|---|---|---|---|
| **D1** | N1 consolidation | Powered time charged from `t=0` to `last_busy + threshold`, so **mid-horizon sleep gaps were never credited** | Exact per-slot interval accounting. Saving corrected from **0.2% to 13.2%** at a 300 s threshold. Consolidation is worth *more* than placement (7–10%) |
| **D2** | N6 procurement | No service constraint, so the search trivially bought the slowest efficient device | Added a 60 s mean-delay SLA **and** raised arrival rate to where queueing binds. Optimum changed from homogeneous `{L4:10}` (delay 2142 s, infeasible) to **mixed `{L4:4, L40S:6}`** at 13.4 s delay |
| **D3** | E5 / pilot grid | Regret is 1.0000 for every method | Not a bug: the L4 is argmin in 24/24 rows, closest margin 1.04x. The **grid** is degenerate, not the metric. Requires loads where the L4's memory binds (unrun: N4) |
| **D4** | E4 | Additive + spec prior saturates at k≥1 | Not a bug: additive ranking is row-invariant, so pairwise accuracy is a step function of column order. Also proves additive's 81.7% is a hard ceiling |

## Known limitations, not yet fixed

- **MLPerf Task C regret stays at 4.24x.** Root cause is data coverage, not the model: in 2 of 9 rows the true energy-optimal accelerator is a sparsely-observed edge device (Qualcomm Cloud AI 100 DM.2, 3 observations). Related design error: mixing Jetson and M.2 edge parts with H100s in one matrix is wrong for a datacenter placement study. The matrix should be restricted to datacenter-class accelerators.
- **E8 (allocation amount) is blocked.** The N3 probe confirmed power capping, MIG and clock control are all denied on Modal serverless (`Insufficient Permissions` on both L4 and A100). Needs bare metal.
- **Ranking-loss training not implemented.** Models are fitted with squared error and evaluated on ranking, which optimises the wrong objective.
- **Replicate noise measured only on the pilot grid**, not on BUTTER-E or MLPerf.
