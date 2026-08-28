# -*- coding: utf-8 -*-
"""Bring PENDING.md in line with what actually closed on 19 August 2026."""
import io

p = "PENDING.md"
s = io.open(p, encoding="utf-8").read()
n, missed = 0, []

EDITS = [
 # --- closed this session ---
 ("| B5 | Shapley under an SLA sweep rather than a single 60-second constraint | Frontier of marginal values | 30 min |",
  "| B5 | **DONE 2026-08-19.** Marginal value depends on the contract, not only the mix: four distinct type orderings across six SLA levels under a uniform mix, four under fp16-only, two under vision-only. Total saving rises monotonically 31.9 -> 49.5 per cent as the constraint relaxes. Reproduces the paper's 33.73 and 36.65 per cent at 60 s. 24/24 sanity | done |"),

 ("| I5 | Several queueing citations are DOI-inferred or unverified; run bibtest over the final bibliography | Paper debt |",
  "| I5 | **DONE 2026-08-19.** bibtest run twice: 63/66 valid, then all three residuals fixed (Notomista upgraded to the IEEE T-RO version, live URLs added to the two web resources). Re-run against the grown 73-reference bibliography after the new citations landed |"),

 ("| H5 | Author-list check on two citations verified from indexes rather than full text: GreenSKU (via Microsoft Research page) and DCN V2 (third-author name form) |",
  "| H5 | **DONE 2026-08-19.** Covered by the bibtest pass, which cross-checks resolved title and first author against each citation and flags misattribution |"),

 ("| F9 | Replicate noise measured only on our grid, not on third-party corpora | Low |",
  "| F9 | **CLOSED 2026-08-19.** Noise floors now measured on the third-party corpora too. ejhusom: max per-row bootstrap SE 0.0034 against an observed between-row spread of 0.0296. Wilkins: median per-cell bootstrap SE 0.0018 against an interaction rms of 0.0918, so the typical cell is resolved at roughly 50:1, though the worst cell (SE 0.091) is not individually resolved and should not be read alone. Our own grid remains 0.0071 against 0.0536, SNR 7.5 | Closed |"),

 ("| E2 | Watt Counts dataset (50 LLMs x 10 GPUs, 370 pairs) | Release gated on paper acceptance | Subset CSV already public in the WattGPU repo; poll for release |",
  "| E2 | Watt Counts (50 LLMs x 10 GPUs, 5,000+ experiments) | **Still gated.** The paper is now public (arXiv:2604.09048, Apr 2026) and is cited as [68], but it states the dataset and benchmark 'will be publicly released on Github with an MIT License upon acceptance'. Poll for the GitHub release | Would be the single largest hardware axis available, 10 GPUs |"),

 ("| J3 | Section 8.1 recipes 2 and 3 (production-trace inference, active probing) remain unimplemented; flagged in the paper as extensions |",
  "| J3 | Recipes 2 and 3 — B3 and B4 — were rebuilt on the B8 LP chooser after the enumerate-and-simulate version proved unrunnable (~140,000 simulations). Running as of 19 Aug 2026; fold results into Section 8.1 and drop the 'only the first is implemented here' sentence once they land |"),

 ("| K6 | B8's simulator gives 4,391 J/job for the optimal fleet where Recipe 1 reports 4,476, a 1.9 per cent gap between two simulator variants. Fleet identity matches. B8's absolute J/job kept OUT of the paper until the gap is traced |",
  "| K6 | **TRACED AND CLOSED 2026-08-19.** One line: Recipe 1 samples arrivals with `rng.integers`, B8 with `rng.choice(p=w)`. Distributionally identical under a uniform mix but they consume different generator state, so one seed gives different arrival sequences. Over 60 seeds the means agree (z = +0.79, 0.21 per cent apart); seed sd is 1.39 per cent, the scale of the 1.90 per cent single-seed gap. Positive control on a skewed mix separates them at z = +46.9. NOT a bug. What it exposed: Section 8 reported single-seed point estimates, now given with seed spread (saving 34.5 per cent, sd 0.4, range 34.0-35.5, optimal fleet identical in 12/12 seeds) |"),

 ("| A3 | Three planning methods: (a) specified mix, (b) observed allocation, (c) active probing | **1 of 3.** Recipe 1 implemented and in Section 8.1. See B3, B4 |",
  "| A3 | Three planning methods: (a) specified mix, (b) observed allocation, (c) active probing | **1 of 3 in the paper; 2 and 3 running.** Recipe 1 is Section 8.1. B3 and B4 rebuilt on the B8 LP chooser and executing |"),
]

for a, b in EDITS:
    if a in s:
        s = s.replace(a, b, 1); n += 1
    else:
        missed.append(a[:52])

s = s.rstrip() + """

## Session close, 19 August 2026

**Closed today:** B5, B8, I5, H5, F9, J2, K1, K5, K6, F1, F2, F4, B6, H2, H3, H6, I2.
**Running:** B3, B4 (Recipes 2 and 3, rebuilt on the B8 LP chooser).

### Corpora now in Table 2 (five, up from three)

| Corpus | Configuration variation | Interaction |
|---|---|---|
| MLPerf Power | none; each cell vendor-tuned | 0.3% |
| This work | precision and batch, fixed across machines | 2.5% |
| Price of Prompting (Husom et al.) | prompt corpus, fixed across machines | 2.49% |
| Hybrid heterogeneous clusters (Wilkins et al.) | token counts, fixed across machines | 2.66% |
| llm-perf-leaderboard | quantisation scheme and dtype | 9.0% |

Four independent campaigns, different groups and instruments, put the interaction at
2.5-2.7 per cent wherever configuration is held fixed across machines. Proposition 1 holds
exactly on every one of them.

### Methodological lesson worth carrying forward

A within-row label PERMUTATION test is the wrong null for an interaction whenever the
column main effect is large, not merely when there are two columns. Shuffling randomises
column assignment and dumps hardware sum-of-squares into the interaction sum-of-squares, so
the null is upward-biased and can never reject. On Wilkins the null median ISS was 19.46
against hardware SS 18.88 plus interaction SS 1.08. The correct null is a per-cell bootstrap
over the raw observations. This bit twice, on two different corpora, in opposite ways.

### Still open and worth doing

| ID | Item | Cost |
|---|---|---|
| C1 | Cross-corpus bridge: our workloads on H100/H200, MLPerf models on our machines | ~$10 |
| C4 | Training workloads across the full grid | ~$15 |
| C2, C3 | Second provider; co-location pairs | ~$10-20 each |
| K2 | Grid'5000, 18 clusters, same benchmarks across platforms. Widest hardware axis available; CPU/HPC not accelerators. Licence unverified | free |
| K3 | DeepEn2023, 8 edge SoCs, kernel-level energy in mJ. Only kernel tier released, licence unverified, access behind a survey | free |
| K4 | Add a sentence to Section 2.1 recording the corpora examined and rejected (EcoCompute, Bench360, SURF/Lisa, high-resolution AI DC telemetry), so a reviewer sees they were considered | free |
| J1, H1 | 250-word abstract variant, once a venue is chosen | free |
| H4 | Choose a venue; determines how much MRTA and queueing vocabulary survives Sections 2.6 and 3 | - |
| L2 | Literature pass on capacity planning under demand uncertainty, to ground B8's formulation in Section 2 | free |
| G1-G6 | Unexplored directions: embodied carbon as an objective, refresh timing, carbon-aware placement, multi-tenant cost sharing, minimum-sufficient-descriptor study, utilisation as a carbon lever | free to moderate |
| E1, E2 | Blocked: power-cap/MIG denied on Modal serverless; Watt Counts data gated on acceptance | - |
"""
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("applied {} edits".format(n))
print("MISSED : {}".format(missed if missed else "none"))
