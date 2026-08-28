# Dataset verification, 19 August 2026

Every candidate checked against one requirement: a workload-by-hardware matrix of MEASURED
energy with enough **headroom** — (energy of the best single fixed target / oracle energy) − 1 —
that a learned placement method has something to win.

| Dataset | Shape | Measured | AI | Counterfactual matrix | **HEADROOM** | Verdict |
|---|---|---|---|---|---|---|
| **Watt Counts / WattGPU subset** | 49 models x **8 real GPUs**, 4,798 runs | Yes, NVML integrated | Yes | Yes | **0.11-2.35%** | **Resolved: NO** |
| Grid'5000 (the "18-cluster" corpus) | **5** x 18 after gating | Yes, wattmeters, joules direct | No | Yes | **8.33%** | Already used; row-starved |
| SWEAT | 35 configs x 5 Intel CPU hosts, 30 reps | Yes, J per 1 s interval | No | Yes, complete | not computed | Robustness only |
| MIT Supercloud | huge | telemetry | partly | **No** | n/a | **Reject** |
| GPU autotuning corpora | config space x ~5 | Yes | kernels | Yes | not computed | Wrong granularity |
| PowerBench / TokenPowerBench | tool, not corpus; 1x/4x/8x H100 | Yes in paper | Yes | one GPU model | n/a | **Reject** |

## 1. Watt Counts / WattGPU subset — resolved, and it is a NO

The gated Watt Counts data is published as a subset in the WattGPU repository
(`https://github.com/maufadel/wattgpu`, **Apache-2.0**, `data/watt_counts_subset.csv`, 22 MB).
Downloaded and analysed: 4,798 runs, 49 models, **8 genuinely different datacenter GPUs**
(A100-SXM4-40GB, A30, H100 NVL, H200, L4, L40S, T4, V100-SXM2-32GB), with an arrival-rate axis.

Headroom on the dense 16 x 8 sub-matrix at each arrival rate:

| arrival rate | headroom | interaction | distinct winners | best GPU's win share |
|---|---|---|---|---|
| batch | **0.63%** | 7.82% | 3 | 81% |
| 0.017 qps | **0.11%** | 0.20% | 2 | 88% |
| 0.33 qps | **2.35%** | 1.07% | 2 | 75% |

The reported evidence for headroom was "H100 best on 45 of 50 models in batch, H200 on 2, L4 on
3". That is a 90 per cent best-machine share, and it converts to 0.1-2.4 per cent of energy. The
qualitative statement that the optimum varies is true; the quantity available to exploit is not.

This is the corpus anyone would reach for, on the widest real accelerator axis available, and it
lands in the same band as everything else.

## 2. Grid'5000 — the "9 x 18" figure is optimistic

18 platforms is right; the usable workload count is **5**, not 9, after a measurement-adequacy
gate that drops three benchmarks whose wattmeter window over-covers a short run by 3-4x. Ungated,
one such benchmark alone held 43 per cent of the interaction sum of squares — window padding
masquerading as architecture. Headroom 8.33 per cent, the highest we hold, but five rows cannot
support a workload-side model: the loading regression fits five coefficients from four points and
the ridge penalty saturates in three of five folds.

## 3. SWEAT — real, complete, and worth one check

Zenodo record 20181490, **CC-BY-4.0**, published 14 May 2026, `SWEAT.zip` 328.8 MB (2.6 GB
extracted). 35 benchmark configurations (20 CPU, 5 memory, 10 disk I/O) executed **30 times on
each of 5 Intel hosts**, from a 2-core i3-6100 to a 64-core Xeon Platinum 8358. Energy directly
measured and reported in joules per second interval. The same 35 configurations run on every host,
so the matrix is complete.

Limits: five targets, all Intel CPUs, no accelerators, no AI workloads, and the Zenodo creator
field reads "Anonymous", which lowers provenance confidence.

Why it is still worth one pass: at 35 rows it is the only complete corpus with enough workloads to
test the row-scarcity finding directly. Two independent agents concluded the descriptor-to-loading
regression fails below roughly 18-20 rows; SWEAT is above that threshold with a complete matrix
and 30 repetitions per cell, so it can separate row scarcity from everything else.

## 4. MIT Supercloud — reject, and the reason generalises

The objection is decisive and applies to every production accounting archive: a job runs on the
machine that received it, so the counterfactual E(i, j) for the machines it did NOT run on is
never observed. Hundreds of thousands of jobs give one column, not a matrix. This is the same
failure as PM100 and the SURF/Lisa traces.

## 5. GPU autotuning corpora — wrong granularity

Kernel-level measurement was already assessed and rejected via DeepEn2023: a kernel is an operator
whose hardware assignment is not a scheduling decision, because the model owning it is already
resident. No autotuning corpus carrying energy rather than latency was located.

## 6. PowerBench — a tool, not a corpus

Resolves to TokenPowerBench (arXiv:2512.03024), which is released as an open-source *benchmark
harness* rather than a measurement release. Its hardware topology axis is 1x, 4x and 8x H100:
one GPU model at different device counts, which is the same shape that disqualified ML.ENERGY v3,
where a monotone device-count axis inflates the column count without adding decisions.

## Conclusion

Seven corpora now have measured headroom: our grid 0.00, Watt Counts 0.11-2.35, extended 1.97,
training 2.00, llm-perf 1.81, Grid'5000 8.33, ML.ENERGY 1.57. Spanning laptop to datacenter, CPU
to accelerator, 2 to 18 targets, and 5 to 13,121 workloads, the energy available to ANY placement
method is between zero and eight per cent and usually near two.

That is not a gap in our search. It is a property of the problem, and it is the paper's own
"variance share is not decision relevance" argument carried one step further: the interaction is
real, it is decision-relevant, and it is worth very little.
