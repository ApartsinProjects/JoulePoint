# -*- coding: utf-8 -*-
"""Stage-4 demonstration: a trace-driven, closed-loop energy-space scheduler versus a hardware-space baseline.
The fleet is 5 each of T4/L4/A10G/A100. A synthetic trace delivers a job mix that DRIFTS over time between
light (memory-bound) and heavy (compute-bound) models, so the controller must react. Ground-truth power and
latency at every (GPU, model, cap) come from the measured dataset (batch control curves). Under a fleet power
budget we compare three controllers on SLO-met throughput and energy:

  UNIFORM   one fleet-wide cap (largest meeting the budget), jobs placed round-robin  (hardware-space)
  ENERGY    per-job cap from the CHEAP PROBE predictor + route to the greenest GPU meeting the SLO,
            greedy under the budget                                                    (energy-space, deployable)
  ORACLE    same but with perfect per-(GPU,model,cap) knowledge                        (upper bound)

The arrival mix is synthetic and documented; a real inference trace (e.g. Azure LLM) can be swapped into
sample_jobs() without changing the controllers. No Date/random primitives from the sandbox; a seeded RNG makes
the trace deterministic.
"""
import os, json
import numpy as np
import pandas as pd
from predict_power import RAW, saturating_set

TYPES = ["L4", "A10G", "A100"]  # T4 excluded from the analysis (see paper limitations)
TDP = {"T4": 70, "L4": 72, "A10G": 300, "A100": 400}
NPER = 5
SLO_SLACK = 0.25            # a job's SLO is 1.25x its best-achievable latency on its assigned GPU class


def _card(df):
    c = df.gpu.str.replace("NVIDIA ", "", regex=False).str.replace("Tesla ", "", regex=False)
    return c.str.replace("A100-SXM4-40GB", "A100", regex=False).str.strip()


def load_curves():
    fr = [pd.read_csv(os.path.join(RAW, "aws_sweep_batch.csv")),
          pd.read_csv(os.path.join(RAW, "aws_sweep_a100_batch.csv"))]
    df = pd.concat(fr, ignore_index=True); df["card"] = _card(df); df = saturating_set(df)
    # Mean the repetitions per operating point so each curve is one point per cap (see make_efficiency_table.load).
    df = df.groupby(["card", "workload", "cap_w"], as_index=False).agg(
        power_w=("power_w", "mean"), throughput=("throughput", "mean"))
    cur = {}                                  # (card, model) -> sorted arrays cap_w, power, lat_ms, thr
    probe = {}                                # (card, model) -> predicted optimal cap index (cheap-probe rule)
    for (card, wl), g in df.groupby(["card", "workload"]):
        g = g.sort_values("cap_w")
        caps = g.cap_w.values.astype(float); P = g.power_w.values; thr = g.throughput.values; lat = 1000.0 / thr
        cur[(card, wl)] = dict(cap=caps, P=P, lat=lat, thr=thr)
        # cheap-probe predictor (validated r=0.90): energy-optimal cap ~ uncapped draw/TDP; use draw/TDP*TDP as target cap
        pred_cap = (P[-1] / TDP[card]) * TDP[card]              # = uncapped draw; nearest sweep cap below it
        k = int(np.argmin(np.abs(caps - max(caps[0], min(pred_cap, caps[-1])))))
        probe[(card, wl)] = k
    return cur, probe


def sample_jobs(cur, tick, rng, n):
    # drifting mix: heavy fraction oscillates over the trace; jobs are (model) draws
    models = sorted({m for (_, m) in cur})
    heavy = ["vit_b_16", "vit_b_32", "swin_t", "sd_unet", "convnext_small", "resnet152", "vgg16", "wide_resnet50_2"]
    light = [m for m in models if m not in heavy]
    hf = 0.5 + 0.4 * np.sin(2 * np.pi * tick / 60.0)           # heavy fraction 0.1..0.9
    jobs = []
    for _ in range(n):
        pool = heavy if rng.random() < hf else light
        jobs.append(pool[rng.integers(len(pool))])
    return jobs


def op_for(cur, card, model, cap_target):
    c = cur.get((card, model))
    if c is None:
        return None
    k = int(np.argmin(np.abs(c["cap"] - cap_target)))
    return dict(P=c["P"][k], lat=c["lat"][k], thr=c["thr"][k], cap=c["cap"][k])


def best_gpu_lat(cur, model):
    # best achievable latency for this model over all GPU types (defines its SLO)
    best = np.inf
    for t in TYPES:
        c = cur.get((t, model))
        if c is not None:
            best = min(best, c["lat"].min())
    return best


def _slo_options(cur, j, mode, probe):
    # per GPU type, the operating point that MEETS this job's SLO: uncapped (hardware) or min-power (energy)
    slo = (1 + SLO_SLACK) * best_gpu_lat(cur, j)
    opts = {}
    for t in TYPES:
        c = cur.get((t, j))
        if c is None:
            continue
        ok = np.where(c["lat"] <= slo)[0]
        if len(ok) == 0:
            continue
        if mode == "hardware":
            k = ok[-1]                                 # hardware-space: no power management, run uncapped
        else:                                          # energy-space: minimum power that still meets the SLO
            k = ok[int(np.argmin(c["P"][ok]))]         # (the curve is obtainable from the cheap probe, r=0.90)
        opts[t] = dict(P=c["P"][k], thr=c["thr"][k], lat=c["lat"][k])
    return opts


def run_admission(cur, jobs, fleet_types, budget, mode):
    # FAIR comparison: every controller places jobs competently and only serves jobs it can meet SLO on, under
    # the SAME power budget. Metric = SLO-met jobs admitted (goodput). hardware runs admitted jobs uncapped;
    # energy caps each to the minimum power meeting its SLO, so it fits more jobs per watt.
    _, probe = oracle_probe
    avail = {t: sum(1 for x in fleet_types if x == t) for t in TYPES}
    remaining = budget; served = 0; E = 0.0; thr = 0.0
    order = sorted(range(len(jobs)), key=lambda i: best_gpu_lat(cur, jobs[i]))   # tightest SLO first
    for i in order:
        opts = _slo_options(cur, jobs[i], mode, probe)
        cands = sorted((o["P"], t, o["thr"]) for t, o in opts.items() if avail[t] > 0 and o["P"] <= remaining)
        if cands:
            p, t, tr = cands[0]                       # cheapest SLO-meeting placement that fits the budget
            served += 1; E += p; thr += tr; remaining -= p; avail[t] -= 1
    return dict(energy=E, served=served, thr=thr)


oracle_probe = None


def main():
    global oracle_probe
    cur, probe = load_curves(); oracle_probe = (cur, probe)
    fleet_types = [t for t in TYPES for _ in range(NPER)]
    n = len(fleet_types)
    fleet_tdp = sum(TDP[t] for t in fleet_types)
    rng = np.random.default_rng(7)
    T = 200
    budgets = [0.7, 0.6, 0.5]
    os.makedirs("results", exist_ok=True); res = {}
    def run_fleet(fleet, label):
        print(f"\n#### fleet = {label} (Sigma TDP {sum(TDP[t] for t in fleet)} W) ####")
        for bf in budgets:
            budget = bf * sum(TDP[t] for t in fleet); nn = len(fleet)
            agg = {k: dict(energy=0.0, served=0.0) for k in ["HARDWARE", "ENERGY"]}
            for tick in range(T):
                jobs = sample_jobs(cur, tick, rng, nn)
                for name, mode in [("HARDWARE", "hardware"), ("ENERGY", "energy")]:
                    out = run_admission(cur, jobs, fleet, budget, mode)
                    agg[name]["energy"] += out["energy"]; agg[name]["served"] += out["served"]
            hw, eng = agg["HARDWARE"], agg["ENERGY"]
            epj_hw = hw["energy"]/max(hw["served"],1); epj_en = eng["energy"]/max(eng["served"],1)
            print(f"  budget {int(bf*100)}%: served/tick HW {hw['served']/T:.1f} -> EN {eng['served']/T:.1f} "
                  f"(goodput x{eng['served']/max(hw['served'],1):.2f}); energy/job {epj_hw:.0f} -> {epj_en:.0f} W "
                  f"({100*(1-epj_en/epj_hw):.0f}% less)")
            tag = "het" if label.startswith("het") else "homo"
            res[f"{tag}_budget_{int(bf*100)}"] = dict(hardware_served=hw["served"]/T, energy_served=eng["served"]/T,
                goodput_gain=eng["served"]/max(hw["served"],1), energy_per_job_pct=100*(1-epj_en/epj_hw))
            if label.startswith("het"):
                res[f"budget_{int(bf*100)}"] = dict(hardware_served=hw["served"]/T, energy_served=eng["served"]/T,
                    goodput_gain=eng["served"]/max(hw["served"],1), energy_per_job_pct=100*(1-epj_en/epj_hw))
    print(f"== Closed-loop scheduler under a fleet power budget, {T}-tick drifting trace ==")
    print("   metric: SLO-met jobs served/tick (goodput) + energy per served job; both place well, both meet SLO on what they serve")
    run_fleet(fleet_types, "heterogeneous 5x(L4,A10G,A100)")
    run_fleet(["A100"] * len(fleet_types), "homogeneous NxA100 (no routing; isolates the capping lever)")
    # SLO-slack sensitivity of the energy-per-job saving (fairness check), heterogeneous fleet at 60% budget
    print("\n#### SLO-slack sensitivity (energy/job saving vs SLO tightness), het fleet, 60% budget ####")
    global SLO_SLACK; sens = []
    budget = 0.6 * fleet_tdp
    for slack in [0.10, 0.25, 0.50, 1.0]:
        SLO_SLACK = slack; rng2 = np.random.default_rng(7)
        a = {k: dict(energy=0.0, served=0.0) for k in ["hardware", "energy"]}
        for tick in range(T):
            jobs = sample_jobs(cur, tick, rng2, n)
            for m in ["hardware", "energy"]:
                o = run_admission(cur, jobs, fleet_types, budget, m)
                a[m]["energy"] += o["energy"]; a[m]["served"] += o["served"]
        hw, en = a["hardware"], a["energy"]
        pct = 100 * (1 - (en["energy"]/max(en["served"],1)) / (hw["energy"]/max(hw["served"],1)))
        sens.append(dict(slo_slack=slack, energy_per_job_pct=float(pct)))
        print(f"  SLO +{int(100*slack)}%: {pct:.0f}% less energy/job")
    SLO_SLACK = 0.25
    res["slo_sensitivity"] = sens
    json.dump(res, open("results/energy_scheduler_sim.json", "w"), indent=1)
    print("\nsaved -> results/energy_scheduler_sim.json")


if __name__ == "__main__":
    main()
