# Data provenance — AI workload power shaping

Every dataset used by this project, where it came from, its licence, and what it is
suitable for. `power_shaping/data/` is gitignored (large / re-downloadable); this file plus
`fetch_public_data.py` are the committed record. "Elasticity" below means a **power-cap or
DVFS sweep** (multiple power/frequency settings per workload → a power-vs-throughput curve),
which is what the shaping problem needs; "operating point" means a single power setting per
config (useful for baseline power distributions, not for intervention response).

## Datasets in use

| File in `data/raw/` | Source | DOI / URL | Licence | Kind | Used by |
|---|---|---|---|---|---|
| `dvfs_sweep.csv`, `SRP_total_power.csv`, `SRP_exp_performance.csv`, `CAISOexp_totalpower_cleaned.csv` | Emerald / Phoenix field demo (Nature Energy 2025), 256-GPU cluster | github.com/ai-emerald/emerald-ai-demo-may-2025 · paper 10.1038/s41560-025-01927-1 | Apache-2.0 (repo) | **Elasticity** (8 LLM workloads × 6 GPU power caps) + real grid-experiment power traces | `poca_killtest.py`, `pocb_sim.py`, `validate_real.py` |
| `zeus_summary_power_v100.csv`, `..._a40.csv` | Zeus (USENIX NSDI'23), You et al. | github.com/ml-energy/zeus `.../zeus_nsdi23/trace/` | Apache-2.0 | **Elasticity** (vision / recommendation / speech / NLP × 100–250 W power limits) | `zeus_killtest.py` |
| `grid5000_dvfs.csv` | Georges Da Costa, "Power, performance and system measures of HPC benchmarks on multiple hardware" | Zenodo record **14914799** (10.5281/zenodo.14914799); concept 10.5281/zenodo.10982238; paper 10.1016/j.suscom.2025.101106 | **CC-BY-4.0** (redistributable w/ attribution) | **Elasticity** (CPU DVFS: NAS Parallel Benchmarks × frequencies 0.8–1.2 GHz, wattmeter energy in joules) | *(available; not yet wired in)* |
| `eehpc/data_sources.zip` | Energy-efficient-HPC DVFS study (Slurm + likwid/RAPL) | reused from sibling energey project; **original citation to be confirmed** | unconfirmed (internal reuse) | **Elasticity** (CPU DVFS 1.0–3.7 GHz over **STREAM = memory-bound** vs **gromacs = compute-bound** + ciao) | *(available; not yet wired in)* |
| `azure_llm_conv.csv`, `azure_window.parquet` | Azure LLM Inference Trace 2024 (DynamoLLM, HPCA'25) | github.com/Azure/AzurePublicDataset release `dataset-llm-2024` | CC-BY-4.0 | Real inference **arrivals** (27.3M requests; no power, no priority) | `prep_azure.py`, `pocb_sim.py` |

## Reusable from the sibling energy project (`/e/Projects/Grants/energey/data/`)

These were measured/collected for the fleet-composition energy paper. They are **operating
points, not sweeps**, so they give baseline power/energy heterogeneity and features, but
**cannot alone provide intervention-response elasticity** (same limitation the spec notes for
Watt Counts). Kept here as a pointer, not copied in.

| Dataset | Source | Kind | Suitable for power-shaping? |
|---|---|---|---|
| `grid5000` | Da Costa / Zenodo 14914799 (CC-BY-4.0) | CPU DVFS sweep (NPB) | **Yes — elasticity** (copied in as `grid5000_dvfs.csv`) |
| `eehpc` | energy-efficient-HPC DVFS study (likwid/RAPL) | CPU DVFS sweep, STREAM vs gromacs | **Yes — elasticity, mem- vs compute-bound** (copied in as `eehpc/`) |
| `ejhusom` | HuggingFace LLM Inference Energy dataset (CC-BY-SA-4.0) | LLM energy, operating points across laptop/workstation/server | Baseline power heterogeneity only |
| `llm-perf` | HuggingFace llm-perf-leaderboard | Quantization × GPU operating points | Baseline / feature engineering |
| `wattgpu` | Watt Counts subset + GPU spec-feature table | >5k experiments (operating points) + static GPU specs | Baseline + hardware features |
| `mlperf-power` | MLCommons cm4mlperf-results | Single max-power operating points | Baseline, machine axis |
| `butter-e` | BUTTER-E, OpenEI 5991 (10.25984/2329316) | 21k workloads × 2 machine classes, operating points | Load axis, baseline |
| `mlenergy-v3` | ML.ENERGY leaderboard v3 | LLM/diffusion operating points (no power_limit) | Baseline only |
| `wilkins` | Wilkins LLM serving | token-limit sweep, not power | Not elasticity |
| `sweat` | SWEAT (workload classification, 5 infra × 35 configs) | fixed-frequency energy operating points | **Checked — not a power/DVFS sweep, not suitable for elasticity** |

## What is still missing (the one real gap) -- now partly filled by own measurement

No *public* dataset gives **GPU** power-cap sweeps across genuinely heterogeneous memory- vs
compute-bound workloads. We filled this ourselves: `data/raw/own_aws_sweep.csv` is a measured
**A10G power-cap elasticity sweep of 26 diverse workloads** (GEMM/conv/attention/FFT/Cholesky
through memory-bound gather/reduction/softmax/scatter; 6 caps 50-100% TDP; a 13x range in
sheddable power), collected with the self-cleaning `aws_sweep_expand.py` harness for ~$1.
`data/raw/aws_sweep_multi.csv` adds L4 + T4 for cross-hardware (50 workload x GPU curves).
This directly grew the within-construct workload count from ~7 to 26; the leakage-free Oracle
Capture learning curve (`aws_data_impact.py`) rises from failure at 6 workloads to ~48% at 26,
demonstrating that distinct-workload count -- not model capacity -- was the bottleneck.

## Provenance scripts

- `fetch_public_data.py` (this folder) — re-downloads emerald + Azure + Zeus + grid5000.
- Sibling project references: `/e/Projects/Grants/energey/experiments/fetch_datasets.py`
  (BUTTER-E), `.../k2_grid5000.py` (grid5000 Zenodo fetch + md5 verification + licence read),
  `.../fetch_mlperf_power.py`, `.../fetch_wilkins.py`.
