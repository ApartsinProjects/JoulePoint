# -*- coding: utf-8 -*-
"""Consolidate every most-recent ELF measurement into ONE master operating-point table.

Merges all released sweeps regardless of actuator or load: the 3-rep power-cap batch sweeps (control = batch 32
and saturate = auto-calibrated max-draw batch), the fine-grid batch-32 cap sweeps (spectrum + a100), and the
clock/DVFS sweeps. Every row is one operating point (a card x workload x actuator-setting), rep-averaged, with:
  - provenance (which sweep produced it) and actuator (power_cap or clock);
  - labels: card, workload/job type, batch, mode;
  - the operating point: cap_w / cap_frac (power cap) or clock_mhz (DVFS);
  - saturation: is_saturated flag + saturation_frac (this batch's uncapped draw / the peak uncapped draw for
    that card+workload) + measured GPU utilization;
  - measurements (mean over reps, with sd and n_reps): power_w, throughput, latency, energy per inference,
    draw as a fraction of the cap;
  - card parameters (datasheet specs) and job parameters (params, FLOPs, depth, op class, arithmetic intensity);
  - the fitted response-law parameters for that card x workload x mode (P0, beta, Rmax proxy, fit R2).

Outputs data/processed/elf_master.csv (+ a coverage summary and a data dictionary). Excludes the reaction-time
time series (a different structure) and the superseded .bak / shard / smoke / L40S / own_aws intermediates.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "data", "raw")
OUT = os.path.join(HERE, "data", "processed")
os.makedirs(OUT, exist_ok=True)

# Datasheet card specs (NVIDIA). fp16 = FP16 tensor TFLOPS (dense), mem_bw GB/s, boost MHz, L2 MB, bus bits.
CARD_SPEC = {
    "T4":   dict(tdp_w=70,  sm=40,  mem_bw_gbs=320,  fp16_tflops=65,  fp32_tflops=8.1,  mem_gb=16, tensor_cores=320, boost_mhz=1590, l2_mb=4,  bus_bit=256),
    "L4":   dict(tdp_w=72,  sm=58,  mem_bw_gbs=300,  fp16_tflops=121, fp32_tflops=30.3, mem_gb=24, tensor_cores=232, boost_mhz=2040, l2_mb=48, bus_bit=192),
    "A10G": dict(tdp_w=300, sm=80,  mem_bw_gbs=600,  fp16_tflops=70,  fp32_tflops=31.2, mem_gb=24, tensor_cores=320, boost_mhz=1710, l2_mb=6,  bus_bit=384),
    "A100": dict(tdp_w=400, sm=108, mem_bw_gbs=1555, fp16_tflops=312, fp32_tflops=19.5, mem_gb=40, tensor_cores=432, boost_mhz=1410, l2_mb=40, bus_bit=5120),
}
# Arithmetic intensity (FLOP/byte, approx roofline) and coarse op class per workload.
KAI = {"resnet50": 120, "resnet152": 120, "resnext50_32x4d": 110, "wide_resnet50_2": 130, "vgg16": 200,
       "densenet121": 90, "efficientnet_b0": 40, "efficientnet_v2_s": 55, "convnext_tiny": 60, "convnext_small": 65,
       "mobilenet_v3_large": 30, "mnasnet1_0": 35, "regnet_y_1_6gf": 70, "regnet_y_8gf": 90, "shufflenet_v2_x1_0": 25,
       "vit_b_16": 80, "vit_b_32": 80, "swin_t": 90, "llm_decode": 8, "sd_unet": 100}
KOP = {**{k: "conv" for k in ["resnet50", "resnet152", "resnext50_32x4d", "wide_resnet50_2", "vgg16", "densenet121",
       "efficientnet_b0", "efficientnet_v2_s", "convnext_tiny", "convnext_small", "mobilenet_v3_large",
       "mnasnet1_0", "regnet_y_1_6gf", "regnet_y_8gf", "shufflenet_v2_x1_0", "sd_unet"]},
       **{k: "attention" for k in ["vit_b_16", "vit_b_32", "swin_t", "llm_decode"]}}


def _card(s):
    s = s.str.replace("NVIDIA ", "", regex=False).str.replace("Tesla ", "", regex=False)
    return s.str.replace("A100-SXM4-40GB", "A100", regex=False).str.strip()


def _read(fname, prov, actuator, default_mode=None, default_batch=None):
    p = os.path.join(RAW, fname)
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d["card"] = _card(d.gpu)
    d["provenance"] = prov
    d["actuator"] = actuator
    if "mode" not in d.columns:
        d["mode"] = default_mode
    if "batch" not in d.columns:
        d["batch"] = default_batch
    for c in ["cap_w", "cap_frac", "clock_mhz", "util", "t_compute_ms", "power_w", "throughput", "rep"]:
        if c not in d.columns:
            d[c] = np.nan
    return d[["provenance", "actuator", "card", "workload", "mode", "batch", "cap_w", "cap_frac",
              "clock_mhz", "rep", "power_w", "util", "t_compute_ms", "throughput"]]


def main():
    sources = [
        # power-cap sweeps, 3-rep, control (batch 32) + saturate (auto-calibrated max-draw batch)
        _read("aws_sweep_batch.csv", "cap_batch_3rep", "power_cap"),
        _read("aws_sweep_a100_batch.csv", "cap_a100_batch_3rep", "power_cap"),
        # fine-grid power-cap sweeps, batch 32, 2-rep
        _read("aws_sweep_spectrum.csv", "cap_spectrum_finegrid_2rep", "power_cap", default_mode="bs32", default_batch=32),
        _read("aws_sweep_a100.csv", "cap_a100_finegrid_2rep", "power_cap", default_mode="bs32", default_batch=32),
        # clock / DVFS sweeps (loaded batch), reaches below the driver power-cap floor
        _read("aws_sweep_clock.csv", "clock_dvfs", "clock", default_mode="clock"),
        _read("aws_sweep_clock_a10g.csv", "clock_dvfs", "clock", default_mode="clock"),
    ]
    df = pd.concat([s for s in sources if s is not None], ignore_index=True)
    df = df[df.power_w > 0].copy()

    # rep-aggregate: one row per operating point (card, workload, provenance, actuator, mode, batch, cap_w, clock_mhz)
    keys = ["provenance", "actuator", "card", "workload", "mode", "batch", "cap_w", "cap_frac", "clock_mhz"]
    g = df.groupby(keys, dropna=False, as_index=False).agg(
        n_reps=("power_w", "size"),
        power_w=("power_w", "mean"), power_w_sd=("power_w", "std"),
        throughput=("throughput", "mean"), throughput_sd=("throughput", "std"),
        util_pct=("util", "mean"), t_compute_ms=("t_compute_ms", "mean"))

    # derived measurements
    g["energy_J_per_step"] = g.power_w / g.throughput          # J per inference step (whole batch forward pass)
    g["latency_ms"] = 1000.0 / g.throughput                    # ms per inference step
    g["draw_frac_of_cap"] = np.where(g.cap_w > 0, g.power_w / g.cap_w, np.nan)
    g["cap_pct_tdp"] = np.where(g.cap_w > 0, 100.0 * g.cap_w / g.card.map(lambda c: CARD_SPEC[c]["tdp_w"]), np.nan)
    # cap_below_floor: the requested cap is below the driver's enforceable power floor, so the card ignores it
    # and draws its floor instead (measured draw exceeds the set cap). These are the rows the clock actuator is
    # needed to reach, and they should not be read as a job drawing more than its cap.
    g["cap_below_floor"] = (g.actuator == "power_cap") & (g.draw_frac_of_cap > 1.02)

    # saturation: per (card, workload) find the PEAK uncapped draw across every measured batch/mode; each
    # config's saturation_frac = its own uncapped (max-cap / max-clock) draw / that peak. is_saturated >= 0.97.
    # A config's uncapped draw is its power at the least-throttled point (highest cap, or highest clock).
    conf_keys = ["provenance", "actuator", "card", "workload", "mode", "batch"]
    g["_throttle"] = np.where(g.actuator == "clock", g.clock_mhz, g.cap_w)
    _idx = g.groupby(conf_keys, dropna=False)["_throttle"].idxmax()
    unc = g.loc[_idx, conf_keys + ["power_w"]].rename(columns={"power_w": "conf_uncapped_draw"})
    g = g.merge(unc, on=conf_keys, how="left").drop(columns=["_throttle"])
    peak = g.groupby(["card", "workload"])["conf_uncapped_draw"].transform("max")
    g["saturation_frac"] = g.conf_uncapped_draw / peak
    g["is_saturated"] = g.saturation_frac >= 0.97
    # card-usage saturation: how close the uncapped (least-throttled) draw of this config's batch is to the
    # card's TDP, and a flag for configs that draw near full card power uncapped. (Distinct from is_saturated,
    # which is relative to the workload's own peak; this is relative to the card's rated power.)
    g["uncapped_draw_pct_tdp"] = 100.0 * g.conf_uncapped_draw / g.card.map(lambda c: CARD_SPEC[c]["tdp_w"])
    g["card_saturating"] = g.uncapped_draw_pct_tdp >= 90.0

    # canonical-subset flags: the exact subsets the paper's figures/tables use, so "correct subset" is one
    # column filter on this single table. The 3-rep batch sweeps are primary; the fine-grid 2-rep sweeps are
    # supplementary. control_primary = law/curve/Table2/Fig1/Fig7 regime (batch 32). loaded_primary =
    # saturating_set (one max-draw mode per card+workload) = Joule point / FIX / Fig4 / Fig5 regime.
    PRIM = ["cap_batch_3rep", "cap_a100_batch_3rep"]
    g["control_primary"] = g.provenance.isin(PRIM) & (g["mode"] == "control")
    _pm = g[g.provenance.isin(PRIM)].drop_duplicates(["card", "workload", "mode"])[["card", "workload", "mode", "conf_uncapped_draw"]]
    _best = _pm.sort_values("conf_uncapped_draw").groupby(["card", "workload"], as_index=False).tail(1)
    _keep = set(zip(_best.card, _best.workload, _best["mode"]))
    g["loaded_primary"] = [prov in PRIM and (c, w, m) in _keep
                           for prov, c, w, m in zip(g.provenance, g.card, g.workload, g["mode"])]

    # card parameters
    spec = pd.DataFrame(CARD_SPEC).T.reset_index().rename(columns={"index": "card"})
    spec["ridge_point"] = spec.fp16_tflops * 1000.0 / spec.mem_bw_gbs
    g = g.merge(spec, on="card", how="left")

    # job parameters
    jf = os.path.join(RAW, "job_features.csv")
    if os.path.exists(jf):
        j = pd.read_csv(jf).rename(columns={"model": "workload"})
        g = g.merge(j, on="workload", how="left")
    g["arith_intensity"] = g.workload.map(KAI)
    g["op_class"] = g.workload.map(KOP)

    # fitted response-law parameters (per card x workload x mode)
    emf = os.path.join(RAW, "energy_model_fit.csv")
    if os.path.exists(emf):
        f = pd.read_csv(emf)[["card", "workload", "mode", "P0", "beta", "r2_nl", "Rstar_frac", "Pmin"]]
        f = f.rename(columns={"P0": "fit_P0_w", "beta": "fit_beta", "r2_nl": "fit_r2", "Rstar_frac": "fit_Rstar_frac", "Pmin": "fit_Pmin_w"})
        g = g.merge(f, on=["card", "workload", "mode"], how="left")

    g = g.sort_values(["card", "workload", "actuator", "mode", "batch", "cap_w", "clock_mhz"]).reset_index(drop=True)
    out_csv = os.path.join(OUT, "elf_master.csv")
    g.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}: {len(g)} operating-point rows, {len(g.columns)} columns")

    # data dictionary
    DICT = {
        "provenance": "source sweep that produced the row (cap_batch_3rep, cap_a100_batch_3rep, cap_spectrum_finegrid_2rep, cap_a100_finegrid_2rep, clock_dvfs)",
        "actuator": "how the operating point was set: 'power_cap' (nvidia-smi -pl) or 'clock' (nvidia-smi -lgc, DVFS)",
        "card": "GPU model (T4, L4, A10G, A100)",
        "workload": "inference job type (20 models: CNN/ViT/Swin/LLM-decode/SD-UNet)",
        "mode": "batch regime: control = batch 32; saturate = auto-calibrated max-draw batch; bs32 = fine-grid batch-32 sweep; clock = DVFS sweep at the loaded batch",
        "batch": "inference batch size",
        "cap_w": "power cap set, watts (power_cap rows; NaN for clock)",
        "cap_frac": "cap_w / TDP as measured by the sweep tool (power_cap rows)",
        "clock_mhz": "graphics clock locked, MHz (clock rows; NaN for power_cap)",
        "cap_pct_tdp": "cap_w as a percentage of the card's TDP",
        "n_reps": "number of measured repetitions averaged into this row (2 or 3)",
        "power_w": "mean board power draw, watts",
        "power_w_sd": "standard deviation of board power across reps",
        "throughput": "mean throughput, inference steps per second (one step = one forward pass of the whole batch)",
        "throughput_sd": "standard deviation of throughput across reps",
        "util_pct": "mean GPU utilization percent (only sweeps that recorded it: batch + clock; NaN for fine-grid)",
        "t_compute_ms": "mean compute time per step, milliseconds",
        "energy_J_per_step": "energy per inference step = power_w / throughput, joules",
        "latency_ms": "latency per step = 1000 / throughput, milliseconds",
        "draw_frac_of_cap": "power_w / cap_w (power_cap rows); >1 means the cap is below the driver floor",
        "cap_below_floor": "True when the requested cap is below the driver's enforceable floor (draw exceeds cap); the card draws its floor instead",
        "is_saturated": "True when this config's uncapped draw is >=97% of the peak uncapped draw for that card+workload (the GPU is loaded)",
        "saturation_frac": "this config's uncapped draw / the peak uncapped draw for that card+workload, in (0,1]",
        "uncapped_draw_pct_tdp": "the config's uncapped (least-throttled) board draw as a percentage of the card's TDP; how close the workload comes to full card power without a cap",
        "card_saturating": "True when uncapped_draw_pct_tdp >= 90, i.e. the workload draws near full card power uncapped (a card-usage saturation flag, distinct from is_saturated)",
        "control_primary": "True for the canonical law/curve subset: 3-rep batch sweeps, mode=control (batch 32). Filter for Fig 1, section 5, Table 2, Fig 7.",
        "loaded_primary": "True for the canonical loaded subset (saturating_set): 3-rep batch, the single max-draw mode per card+workload. Filter for the Joule point, FIX, Fig 4, Fig 5.",
        "conf_uncapped_draw": "the config's board power at its least-throttled point (highest cap or clock), watts",
        "tdp_w": "card thermal design power, watts (datasheet)",
        "sm": "streaming-multiprocessor count (datasheet)",
        "mem_bw_gbs": "memory bandwidth, GB/s (datasheet)",
        "fp16_tflops": "FP16 tensor throughput, TFLOPS dense (datasheet)",
        "fp32_tflops": "FP32 throughput, TFLOPS (datasheet)",
        "mem_gb": "device memory, GB (datasheet)",
        "tensor_cores": "tensor-core count (datasheet)",
        "boost_mhz": "rated boost clock, MHz (datasheet)",
        "l2_mb": "L2 cache, MB (datasheet)",
        "bus_bit": "memory bus width, bits (datasheet)",
        "ridge_point": "roofline ridge point = fp16_tflops*1000 / mem_bw_gbs (FLOP/byte)",
        "params_M": "model parameter count, millions (job feature)",
        "gflops": "model GFLOPs per inference (job feature)",
        "depth": "model depth, number of layers (job feature)",
        "conv_frac": "fraction of FLOPs in convolutions (job feature)",
        "op_type": "dominant op type from job_features (conv/attention/...)",
        "op_class": "coarse op class (conv/attention)",
        "arith_intensity": "approximate arithmetic intensity, FLOP/byte (job label)",
        "fit_P0_w": "fitted response-law power floor P0, watts, for this card x workload x mode (P(R)=P0+aR^beta)",
        "fit_beta": "fitted response-law exponent beta (superlinearity)",
        "fit_r2": "R^2 of the nonlinear response-law fit",
        "fit_Rstar_frac": "closed-form energy-optimal rate as a fraction of Rmax",
        "fit_Pmin_w": "fitted minimum power, watts",
    }
    md = ["# ELF master dataset - data dictionary", "",
          f"`data/processed/elf_master.csv`: one row per measured operating point (card x workload x actuator setting), "
          f"rep-averaged. {len(g)} rows, {len(g.columns)} columns. Merges every most-recent released sweep across both "
          "actuators (power cap and clock) and both load regimes (saturated and non-saturated).", "",
          "| column | description |", "| --- | --- |"]
    for c in g.columns:
        md.append(f"| `{c}` | {DICT.get(c, '(undocumented)')} |")
    missing = [c for c in g.columns if c not in DICT]
    if missing:
        print("WARN: undocumented columns:", missing)
    open(os.path.join(OUT, "elf_master_dictionary.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(f"wrote {os.path.join(OUT, 'elf_master_dictionary.md')} ({len(g.columns)} columns documented)")

    # coverage summary
    print("\n== coverage: rows by card x actuator x saturation ==")
    cov = g.groupby(["card", "actuator", "is_saturated"]).size().rename("rows").reset_index()
    print(cov.to_string(index=False))
    print("\n== provenance ==")
    print(g.groupby("provenance").agg(rows=("power_w", "size"), cards=("card", "nunique"),
          workloads=("workload", "nunique"), reps=("n_reps", "median")).to_string())
    return g


if __name__ == "__main__":
    main()
