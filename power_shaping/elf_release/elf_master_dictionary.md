# ELF master dataset - data dictionary

`data/processed/elf_master.csv`: one row per measured operating point (card x workload x actuator setting), rep-averaged. 2636 rows, 51 columns. Merges every most-recent released sweep across both actuators (power cap and clock) and both load regimes (saturated and non-saturated).

| column | description |
| --- | --- |
| `provenance` | source sweep that produced the row (cap_batch_3rep, cap_a100_batch_3rep, cap_spectrum_finegrid_2rep, cap_a100_finegrid_2rep, clock_dvfs) |
| `actuator` | how the operating point was set: 'power_cap' (nvidia-smi -pl) or 'clock' (nvidia-smi -lgc, DVFS) |
| `card` | GPU model (T4, L4, A10G, A100) |
| `workload` | inference job type (20 models: CNN/ViT/Swin/LLM-decode/SD-UNet) |
| `mode` | batch regime: control = batch 32; saturate = auto-calibrated max-draw batch; bs32 = fine-grid batch-32 sweep; clock = DVFS sweep at the loaded batch |
| `batch` | inference batch size |
| `cap_w` | power cap set, watts (power_cap rows; NaN for clock) |
| `cap_frac` | cap_w / TDP as measured by the sweep tool (power_cap rows) |
| `clock_mhz` | graphics clock locked, MHz (clock rows; NaN for power_cap) |
| `n_reps` | number of measured repetitions averaged into this row (2 or 3) |
| `power_w` | mean board power draw, watts |
| `power_w_sd` | standard deviation of board power across reps |
| `throughput` | mean throughput, inference steps per second (one step = one forward pass of the whole batch) |
| `throughput_sd` | standard deviation of throughput across reps |
| `util_pct` | mean GPU utilization percent (only sweeps that recorded it: batch + clock; NaN for fine-grid) |
| `t_compute_ms` | mean compute time per step, milliseconds |
| `energy_J_per_step` | energy per inference step = power_w / throughput, joules |
| `latency_ms` | latency per step = 1000 / throughput, milliseconds |
| `draw_frac_of_cap` | power_w / cap_w (power_cap rows); >1 means the cap is below the driver floor |
| `cap_pct_tdp` | cap_w as a percentage of the card's TDP |
| `cap_below_floor` | True when the requested cap is below the driver's enforceable floor (draw exceeds cap); the card draws its floor instead |
| `conf_uncapped_draw` | the config's board power at its least-throttled point (highest cap or clock), watts |
| `saturation_frac` | this config's uncapped draw / the peak uncapped draw for that card+workload, in (0,1] |
| `is_saturated` | True when this config's uncapped draw is >=97% of the peak uncapped draw for that card+workload (the GPU is loaded) |
| `uncapped_draw_pct_tdp` | the config's uncapped (least-throttled) board draw as a percentage of the card's TDP; how close the workload comes to full card power without a cap |
| `card_saturating` | True when uncapped_draw_pct_tdp >= 90, i.e. the workload draws near full card power uncapped (a card-usage saturation flag, distinct from is_saturated) |
| `control_primary` | True for the canonical law/curve subset: 3-rep batch sweeps, mode=control (batch 32). Filter for Fig 1, section 5, Table 2, Fig 7. |
| `loaded_primary` | True for the canonical loaded subset (saturating_set): 3-rep batch, the single max-draw mode per card+workload. Filter for the Joule point, FIX, Fig 4, Fig 5. |
| `tdp_w` | card thermal design power, watts (datasheet) |
| `sm` | streaming-multiprocessor count (datasheet) |
| `mem_bw_gbs` | memory bandwidth, GB/s (datasheet) |
| `fp16_tflops` | FP16 tensor throughput, TFLOPS dense (datasheet) |
| `fp32_tflops` | FP32 throughput, TFLOPS (datasheet) |
| `mem_gb` | device memory, GB (datasheet) |
| `tensor_cores` | tensor-core count (datasheet) |
| `boost_mhz` | rated boost clock, MHz (datasheet) |
| `l2_mb` | L2 cache, MB (datasheet) |
| `bus_bit` | memory bus width, bits (datasheet) |
| `ridge_point` | roofline ridge point = fp16_tflops*1000 / mem_bw_gbs (FLOP/byte) |
| `params_M` | model parameter count, millions (job feature) |
| `gflops` | model GFLOPs per inference (job feature) |
| `depth` | model depth, number of layers (job feature) |
| `conv_frac` | fraction of FLOPs in convolutions (job feature) |
| `op_type` | dominant op type from job_features (conv/attention/...) |
| `arith_intensity` | approximate arithmetic intensity, FLOP/byte (job label) |
| `op_class` | coarse op class (conv/attention) |
| `fit_P0_w` | fitted response-law power floor P0, watts, for this card x workload x mode (P(R)=P0+aR^beta) |
| `fit_beta` | fitted response-law exponent beta (superlinearity) |
| `fit_r2` | R^2 of the nonlinear response-law fit |
| `fit_Rstar_frac` | closed-form energy-optimal rate as a fraction of Rmax |
| `fit_Pmin_w` | fitted minimum power, watts |
