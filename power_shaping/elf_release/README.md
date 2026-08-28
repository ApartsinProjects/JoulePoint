# ELF: the Energy-Latency Frontier dataset for GPU inference

ELF is a dense power-cap response dataset for AI inference on data-center GPUs. For 20 inference models
across four NVIDIA GPUs, it sweeps the board power limit from the driver floor to the thermal design power
(TDP) and records, at each operating point, the board power actually drawn, the achieved throughput, and a
latency breakdown. It is the measured surface behind the paper *From GPUs to Joules: Scheduling AI Inference
in the Space of Energy and its Derivatives*, and it lets others compute power-response laws, compute-power
elasticities, energy-optimal operating points ("Joule points"), and energy-aware routing tables without any
new hardware runs.

## Why this dataset

Most GPU energy studies report a few operating points or a single knob. ELF is a *dense, multi-axis* sweep:
the power cap (driver floor to TDP, ~8 levels) crossed with batch size (a fixed batch and an auto-calibrated
saturating batch), plus a graphics-clock (DVFS) sweep that reaches *below* the driver's power-cap floor, for a
broad model set on four GPU classes. That density is what makes the response *surface*, and its derivatives,
measurable.

## Start here: the consolidated master table

**`data/elf_master.csv` is the headline artifact** and the single source of truth. It merges every sweep below
into one table: one row per measured operating point (card x workload x actuator setting), rep-averaged, across
**both actuators** (power cap and clock/DVFS) and **both load regimes** (saturated and non-saturated). Each row
carries its provenance, the operating point, saturation flags, canonical-subset flags, the measurements, the
card's datasheet specs, the workload's static features, and the fitted response-law parameters, so most analyses
need only this one file and a column filter. Every column is documented in **`data/elf_master_dictionary.md`**
(also copied to the release root). The individual per-sweep CSVs below are the raw inputs it is built from.

Key columns for selecting the right subset: `control_primary` (the law / curve / Table-2 / frontier regime,
3-rep batch 32), `loaded_primary` (the Joule-point regime: the single max-draw mode per card+workload),
`actuator` (`power_cap` or `clock`), `is_saturated` / `saturation_frac`, and `cap_below_floor` (the requested
cap is below the driver floor, so the card drew its floor instead).

## Contents

| file | rows | what |
|---|---|---|
| **`data/elf_master.csv`** | **2636** | **consolidated master table: every operating point, both actuators, both regimes, fully labelled (start here)** |
| `data/elf_master_dictionary.md` | | data dictionary for every master-table column |
| `data/aws_sweep_spectrum.csv` | 1400 | T4 / L4 / A10G, 20 models, cap sweep at batch 32, 2 reps, with a 4-phase latency split |
| `data/aws_sweep_a100.csv` | 480 | A100, same schema as above |
| `data/aws_sweep_batch.csv` | 2736 | T4 / L4 / A10G, control (batch 32) + saturate (auto-calibrated batch), cap sweep, `util`; all three cards at 3 reps |
| `data/aws_sweep_a100_batch.csv` | 912 | A100, control + saturate, 3 repetitions |
| `data/aws_sweep_clock.csv` | 640 | T4 / L4, graphics-clock (DVFS) sweep at the loaded batch, reaching below the power-cap floor, 2 reps |
| `data/aws_sweep_clock_a10g.csv` | 320 | A10G, focused low-range clock (DVFS) sweep, 2 reps |
| `data/reaction_time.csv` | 3322 | A10G, power and latency at ~50 Hz while the cap is stepped (actuation dynamics) |
| `code/` | | the exact measurement scripts, plus the consolidation + verification code (see below) |

Total: about 6500 measured rows across the sweep collections (each operating point measured two to three times) plus one high-rate actuation trace, consolidated into the 2636-row master table.

### Column schema

Sweep files (`aws_sweep_*`):

- `gpu` — device name as reported by the driver (e.g. `NVIDIA A100-SXM4-40GB`).
- `workload` — model identifier (see model list).
- `mode` — `control` (batch 32) or `saturate` (auto-calibrated near-TDP batch). *Batch files only.*
- `batch` — batch size at this operating point. *Batch files only.*
- `cap_frac`, `cap_w` — the requested power cap, as a fraction of TDP and in watts.
- `rep` — repetition index. *2 reps in the spectrum and A100 bs32 files; 3 reps for every card in the g-card and A100 batch files.*
- `power_w` — measured board power draw in **watts** (mean over a ~2 s sampling window at 20 Hz).
- `util` — GPU utilization per cent (NVML). *Batch files only.*
- `t_load_ms`, `t_h2d_ms`, `t_compute_ms`, `t_d2h_ms` — latency split (model load / host-to-device /
  compute / device-to-host), milliseconds. *Spectrum / A100 files only.*
- `throughput` — inferences (forward passes of the batch) per second.

Reaction file (`reaction_time.csv`): `t_ms` (time), `cap_set_w` (the cap in force), `power_w`, `util`,
`iter_ms` (recent per-iteration latency), sampled at ~50 Hz while the cap is stepped.

### GPUs

| GPU | TDP (W) | driver power floor (W) |
|---|---|---|
| Tesla T4 | 70 | ~60 |
| L4 | 72 | ~40 |
| A10G | 300 | ~100 |
| A100 (SXM4 40GB) | 400 | ~100 |

### Models (20)

`llm_decode` (autoregressive LLM decode step with a pre-filled KV cache, Llama-shaped), `sd_unet`
(Stable-Diffusion UNet forward), `vit_b_16`, `vit_b_32`, `swin_t`, `convnext_tiny`, `convnext_small`,
`resnet50`, `resnet152`, `wide_resnet50_2`, `resnext50_32x4d`, `densenet121`, `vgg16`, `efficientnet_b0`,
`efficientnet_v2_s`, `regnet_y_1_6gf`, `regnet_y_8gf`, `mnasnet1_0`, `mobilenet_v3_large`,
`shufflenet_v2_x1_0`. All at FP16.

## Methodology

Each instance was a single AWS EC2 GPU node (g4dn/T4, g6/L4, g5/A10G, p4d 8xA100 sharded). Power caps were
set with `nvidia-smi -pl`. Power was sampled from NVML at 20 Hz over a ~2 s steady window per operating
point; throughput is iterations completed over the same window. In the batch files, `saturate` mode
auto-calibrates the batch to the one whose uncapped draw is highest (OOM-safe), then sweeps caps at that
batch; `control` mode fixes batch 32. Control and saturate are swept back to back in one thermal session.
The clock/DVFS sweep (`aws_sweep_clock*.csv`) instead locks the graphics clock with `nvidia-smi -lgc` at the
loaded batch, which reaches board powers *below* the driver's power-cap floor and so exposes the energy minimum
that the power cap cannot reach on the small cards. The reaction-time trace steps the cap on a live workload and
logs at ~50 Hz to measure settling time.

## Known limitations (please read before using)

1. **Repetitions.** `aws_sweep_batch.csv` (all three g-cards) and `aws_sweep_a100_batch.csv` are three reps
   per operating point; treat single-point differences below a few per cent as noise. The spectrum / A100
   (bs32) files are two reps.
2. **A100 full-power regression.** `efficientnet_v2_s` and `mnasnet1_0` reach higher throughput at an
   intermediate cap than uncapped, reproducibly across all three A100 reps (a clock/boost-management effect),
   so the uncapped point is not their maximum. An earlier single-rep artifact on `swin_t` did not reproduce
   across three reps and is not a regression.
3. **Driver floor clamp.** At the lowest requested cap on each card (A10G ~100 W, A100 ~100 W, L4 ~40 W),
   the driver enforces its minimum limit, so `power_w` can exceed the requested `cap_w`; the master table flags
   these rows as `cap_below_floor`. Analyses that use measured `power_w` are unaffected; treat the requested
   `cap_w` at the deepest level as a lower bound. To reach below the floor, use the clock/DVFS sweeps
   (`aws_sweep_clock*.csv`, `actuator == "clock"` in the master table).
4. **A few L4 mid-range points** exceed their cap by up to ~20% (also flagged `cap_below_floor`); treat with caution.
5. **Thermal drift.** On the A10G, board draw at fixed cap and work drifts by a median ~14% between a cold
   and warm pass; a binding cap removes most of this.

## Reproducing / extending

The `code/` directory contains the exact AWS measurement harnesses. Each launches a throwaway VPC + GPU
instance, runs the sweep, exfiltrates results via an S3 gateway endpoint, and self-cleans. See
`code/README_code.md`. Running them incurs AWS charges and needs an account with GPU quota.

## Citation

If you use ELF, please cite:

> A. Apartsin, Y. Aperstein. *From GPUs to Joules: Scheduling AI Inference in the Space of Energy and its
> Derivatives.* 2026.

## License

Data: **CC-BY-4.0**. Code: **MIT** (see `code/LICENSE`). See `LICENSE` in this directory.
