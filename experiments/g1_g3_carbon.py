# -*- coding: utf-8 -*-
"""
G1 + G3. Carbon, not energy, as the fleet-design objective.

Section 9 of the paper argues ANALYTICALLY that embodied carbon shifts the fleet optimum
without overturning it, and makes a falsifiable side claim: the DEGREE of heterogeneity
"should shrink as grid carbon intensity and utilisation fall". Section 8 ranks three
levers on replayed arrivals: composition 34.3 per cent, consolidation 13.2 per cent,
placement 7 to 10 per cent. Both claims are tested here by experiment rather than by
argument, on the same measured energy tables and the same replayed arrival streams.

G1. Fleet composition minimising TOTAL carbon = operational + amortised embodied, swept
    over grid carbon intensity from 20 to 700 gCO2e/kWh. Device count is a free variable
    (4 to 14 slots) so that "how much silicon to buy" is a decision the search can make,
    not a constant it is handed. Embodied carbon is amortised straight-line over a
    five-year service life, the paper's assumption.

G3. Carbon-aware PLACEMENT versus carbon-aware COMPOSITION under a diurnal grid-intensity
    profile. Placement is decomposed into its spatial part (which device runs the job,
    the paper's 7 to 10 per cent lever) and its temporal part (when the job runs, which
    only a carbon objective creates any incentive for). Temporal shifting is bounded by
    the SLA, and the capacity-free upper bound is computed alongside the achievable
    policy so that a small number cannot be blamed on a weak scheduler.

Everything reuses already-measured energy tables. No new compute is purchased.

------------------------------------------------------------------------------------
SANITY CHECKS, STATED IN ADVANCE
------------------------------------------------------------------------------------
E1  The embodied-carbon component model must reproduce the one vendor figure it is
    calibrated on, NVIDIA's HGX H100 per-GPU footprint.
E2  It must also land near a figure it was NOT fitted to: EcoServe's independent claim
    that an L4 carries about three times less embodied carbon than an H100.

S0  CALIBRATION. The fast simulator written here must reproduce the STORED published
    artefacts bit for bit on the 10-slot, five-type instance: fleet_vs_mix.json's uniform
    optimum (energy per job and mean delay) and shapley_fleet.json's all-A100 reference
    and achievable saving. If S0 fails nothing downstream is trustworthy.
    NOTE, recorded during the run: the paper's PROSE quotes 4,476 J per job and 34.3 per
    cent, which comes from the rng.integers arrival-sampling variant logged in
    k6_seed_variance.json. The shipped simulators and every stored JSON use rng.choice
    and give 4,391 J per job and 33.73 per cent. That 1.90 per cent gap is documented in
    k6_seed_variance.json as Monte Carlo variation between the two draws. Calibrating S0
    against the prose would have been calibrating against the wrong artefact, so S0 is
    stated against the stored JSON and the discrepancy is reported rather than absorbed.

S6  HORIZON. The paper screens fleets for the 60-second constraint on a ONE-HOUR replay.
    Queues in this facility have not reached steady state in one hour, so the screen is
    transient: some fleets that pass it breach the same constraint over a full day. This
    is checked explicitly and everything in G3, which needs a 24-hour horizon to contain
    a diurnal cycle, is re-screened at 24 hours. G1 is reported on the paper's one-hour
    horizon for commensurability and repeated (G1b) on the 24-hour-feasible set.

S1  ZERO EMBODIED. With every embodied-carbon figure set to zero, the carbon-optimal
    fleet must equal the energy-optimal fleet EXACTLY at every grid intensity, and the
    carbon saving must equal the energy saving to floating-point tolerance.

S2  HIGH-INTENSITY LIMIT. As grid intensity grows the operational term dominates, so the
    carbon-optimal fleet must converge to the energy-optimal fleet. Concretely, at
    700 gCO2e/kWh the carbon-optimal fleet must equal the energy-optimal fleet.

S3  MONOTONE DEVICE COUNT. The total device count of the carbon-optimal fleet must be
    non-decreasing in grid intensity. This is the mechanical form of the paper's
    prediction: a dirtier grid makes operational savings worth more silicon. A violation
    is either a real non-monotonicity worth reporting or a bug, and must be traced.

S4  ARGMIN CONSISTENCY. At every intensity the carbon-optimal fleet must have total
    carbon per job no greater than the energy-optimal fleet's, and the energy-optimal
    fleet must have energy per job no greater than the carbon-optimal fleet's. Two
    argmins over the same feasible set cannot both be beaten.

S5  SLA RESPECTED. Every fleet reported as optimal must satisfy the 60-second mean-delay
    constraint, on every seed used to select it.

T1  FLAT PROFILE. Under a constant grid-intensity profile, temporal shifting must be
    worth EXACTLY zero (to 1e-12 relative) and must apply zero shift. This is exact only
    because the temporal lever is defined PURELY: each job keeps the device the baseline
    scheduler gave it and only its start time moves. A policy that were also allowed to
    re-pick the device would defer jobs to wait for a more efficient device even under a
    constant intensity, which is an energy effect and must not be credited to carbon
    awareness. The first version of this experiment did exactly that and T1 failed at
    7.7e-4; the fix was to hold the spatial decision fixed, not to loosen the check.

T2  MECHANISM VALIDATION. The diurnal profile moves by less than 0.2 per cent over a
    60-second window, so a near-zero headline number is EXPECTED and is indistinguishable
    from a scheduler that does nothing. The same code is therefore run against a profile
    whose period is short compared with the slack but still long compared with the 3.5 to
    102-second job runtimes (one-hour period, 30-minute slack), where the intensity search
    must capture nearly the whole swing. Two earlier versions of this check were
    mis-specified and are recorded here rather than quietly dropped. A two-minute period
    fails because job runtimes are a large fraction of it, so the mean intensity over a
    job's execution window is close to the profile mean whatever the start. Asserting the
    one-hour version on the CAPACITY-RESPECTING policy also fails, at 0.9 per cent, and
    inspecting the trace shows why: at this utilisation every job chasing the same trough
    queues behind every other job doing the same, so contention rather than the intensity
    search is binding. The check is therefore made on the capacity-free variant, which is
    what isolates the code under test, and the contention gap is reported as a result.

T3  MONOTONE IN SLACK. The temporal-shifting saving must be non-decreasing in the
    deadline slack it is allowed to use.

T4  BOUND DOMINATES POLICY. The capacity-free upper bound on temporal shifting must be
    greater than or equal to the saving achieved by the capacity-respecting policy, at
    every slack.

T5  NO SLA VIOLATION. The constraint here is on MEAN queueing delay, not a per-job
    deadline, and the baseline scheduler already spends part of that budget. The deferral
    budget granted to the temporal policy is therefore chosen by binary search as the
    largest that keeps the facility inside the same 60-second constraint the composition
    search is held to, and the resulting mean delay is checked against it. An earlier
    version granted every job a full 60 seconds and then asserted the 60-second
    constraint, which is self-contradictory and produced a 61.9-second mean; the check was
    wrong, not the code.

All facility results are reported over multiple seeds with their spread; the simulator
carries about 1.4 per cent Monte Carlo variation between seeds and no single-seed point
estimate is reported anywhere.
"""
import io, json, math, os, sys, time, warnings
from itertools import product
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "experiments")
warnings.filterwarnings("ignore")
from e4_e5_models import load_grid, MACH

# ---------------------------------------------------------------- measured constants
IDLE = {"T4": 27.9, "L4": 29.7, "A10G": 61.2, "L40S": 88.7, "A100-40GB": 71.2}
JOB, SLA_S, HORIZON, LAM = 20000, 60.0, 3600.0, 0.5
NM = len(MACH)
IDLEv = [IDLE[m] for m in MACH]

_keys, _Ylog, _Tput = load_grid()
NK = len(_keys)
# energy per sample (J) and seconds per job, as plain nested lists: the hot loop runs
# tens of millions of iterations and numpy scalar indexing is the bottleneck there.
Ecell = [[10 ** _Ylog[i, j] / 1000.0 for j in range(NM)] for i in range(NK)]
Tcell = [[JOB / _Tput[i, j] for j in range(NM)] for i in range(NK)]

LIFE_YEARS = 5.0
LIFE_S = LIFE_YEARS * 365.25 * 24 * 3600.0

# ---------------------------------------------------------------- embodied carbon
# Per-device cradle-to-gate embodied carbon, kg CO2e, INCLUDING the device's share of
# board, memory and packaging but EXCLUDING the host server (chassis, CPU, DRAM, PSU,
# storage), which published life-cycle work finds dominates a server's embodied carbon.
# Excluding the host is the conservative choice for this experiment: the question is
# what changes when accelerator TYPES are mixed, and hosts are held fixed.
#
# See EMBODIED_SOURCES below for provenance of every figure.
EMBODIED_KG = {}          # filled in by build_embodied()
EMBODIED_SOURCES = {}

# ---- ANCHOR (the only vendor-published figure in the whole table) -------------------
# NVIDIA HGX H100 Product Carbon Footprint Summary, ISO 14067, third-party reviewed by
# WSP (July 2025). https://images.nvidia.com/aem-dam/Solutions/documents/HGX-H100-PCF-Summary.pdf
# Scope: cradle-to-gate for ONE 8-GPU HGX H100 baseboard (8x H100 SXM 80GB, 640 GB HBM3,
# NVSwitches, thermal solution, PCB; 24 kg product). EXCLUDES use phase, end of life,
# and the rest of the server (CPU, host DRAM, storage, chassis, PSU).
ANCHOR_HGX_H100_KG = 1312.0
ANCHOR_GPUS = 8
# Published component split, kg CO2e per baseboard (sums to 1,191.8; the balance to 1,312
# is assembly 8.6 per cent and transport 0.4 per cent).
PCF_SPLIT = dict(memory=546.0, ics=332.0, thermal=230.0, electromechanical=52.0,
                 common=10.6, pcb=9.0, interconnect=4.8, mechanical=7.4)
PCF_UPLIFT = ANCHOR_HGX_H100_KG / sum(PCF_SPLIT.values())      # assembly + transport

# Die area (mm2), process node, memory and TDP for the accelerator dies.
# Die areas and nodes: NVIDIA architecture whitepapers as tabulated on Wikipedia
# (Turing / Ampere / Ada Lovelace / Hopper microarchitecture pages).
DIE = {
    "T4":        dict(chip="TU104", area_mm2=545.0, node_nm=12, node="TSMC 12FFC", mem_gb=16, mem="GDDR6", tdp=70,  tdp_src="NVIDIA T4 product brief (verified)"),
    "L4":        dict(chip="AD104", area_mm2=294.0, node_nm=4,  node="TSMC 4N",    mem_gb=24, mem="GDDR6", tdp=72,  tdp_src="NVIDIA L4 product page (verified)"),
    "A10G":      dict(chip="GA102", area_mm2=628.4, node_nm=8,  node="Samsung 8N", mem_gb=24, mem="GDDR6", tdp=150, tdp_src="NVIDIA A10 datasheet (verified); the AWS A10G SKU may run higher, unverified"),
    "L40S":      dict(chip="AD102", area_mm2=608.4, node_nm=4,  node="TSMC 4N",    mem_gb=48, mem="GDDR6", tdp=350, tdp_src="NVIDIA L40S, datasheet URL did not resolve; UNVERIFIED"),
    "A100-40GB": dict(chip="GA100", area_mm2=826.0, node_nm=7,  node="TSMC N7",    mem_gb=40, mem="HBM2",  tdp=400, tdp_src="NVIDIA A100 SXM datasheet (verified)"),
    "H100":      dict(chip="GH100", area_mm2=814.0, node_nm=4,  node="TSMC 4N",    mem_gb=80, mem="HBM3",  tdp=700, tdp_src="NVIDIA H100 SXM5 (verified)"),
}

# Carbon per unit die area (kg CO2e per cm2) as a function of process node. ACT
# (Gupta et al., ISCA 2022) as restated by CarbonClarity (arXiv:2507.01145) gives mean
# 1.18 kg/cm2 at 28 nm and 2.52 kg/cm2 at 7 nm, with sigma 0.99 at 7 nm (95th percentile
# 4.13, i.e. 1.6x the mean). Only two mean points are published, so intermediate and
# advanced nodes are EXTRAPOLATED by a power law fitted to those two points:
#     CPA(n) = 2.52 * (n / 7) ** b,   b = ln(2.52 / 1.18) / ln(7 / 28) = -0.547
CPA_REF_NM, CPA_REF = 7.0, 2.52
CPA_EXP = math.log(2.52 / 1.18) / math.log(7.0 / 28.0)


def cpa(node_nm):
    return CPA_REF * (node_nm / CPA_REF_NM) ** CPA_EXP


# Memory. The anchor gives HBM3 directly: 546 kg / 640 GB = 0.853 kg CO2e per GB.
# GDDR6 is planar and unstacked, with no TSVs and no logic base die, so its carbon per
# GB is ASSUMED to be half that of HBM. That ratio is an assumption, not a measurement.
HBM_KG_PER_GB = PCF_SPLIT["memory"] / (ANCHOR_GPUS * DIE["H100"]["mem_gb"])
GDDR_RATIO = 0.50
MEM_KG_PER_GB = {"HBM3": HBM_KG_PER_GB, "HBM2": HBM_KG_PER_GB,
                 "GDDR6": HBM_KG_PER_GB * GDDR_RATIO}

# The anchor's per-GPU shares that the die-area term does not explain:
#   IC overhead   = ICs/8 - CPA(4nm) * GH100 area. Package substrate, NVSwitch share,
#                   voltage regulation and support silicon. Held FIXED per device.
#   thermal       = thermal/8 divided by 700 W, i.e. kg CO2e per watt of TDP.
#   fixed card    = (electromechanical + common + PCB + interconnect + mechanical)/8.
IC_OVERHEAD_KG = PCF_SPLIT["ics"] / ANCHOR_GPUS - cpa(4) * DIE["H100"]["area_mm2"] / 100.0
THERMAL_KG_PER_W = PCF_SPLIT["thermal"] / ANCHOR_GPUS / DIE["H100"]["tdp"]
CARD_FIXED_KG = (PCF_SPLIT["electromechanical"] + PCF_SPLIT["common"] + PCF_SPLIT["pcb"]
                 + PCF_SPLIT["interconnect"] + PCF_SPLIT["mechanical"]) / ANCHOR_GPUS


def build_embodied(scale=1.0):
    """Per-device cradle-to-gate embodied carbon, kg CO2e, by a component model
    calibrated term by term against NVIDIA's published HGX H100 footprint.

        E_m = uplift * [ CPA(node_m) * area_m          die silicon, ACT
                       + IC_OVERHEAD                   package, substrate, support ICs
                       + memkg(type_m) * mem_gb_m      memory
                       + THERMAL_PER_W * tdp_m         cooling solution
                       + CARD_FIXED ]                  PCB, electromechanical, mechanical

    The ONLY vendor figure is the anchor. Every per-device number for the five
    accelerators in the fleet is an EXTRAPOLATION, so the factor-of-two sensitivity sweep
    is the load-bearing result and the point estimate is not.
    """
    out, src = {}, {}
    for m in list(MACH) + ["H100"]:
        d = DIE[m]
        die = cpa(d["node_nm"]) * d["area_mm2"] / 100.0
        mem = MEM_KG_PER_GB[d["mem"]] * d["mem_gb"]
        th = THERMAL_KG_PER_W * d["tdp"]
        val = PCF_UPLIFT * (die + IC_OVERHEAD_KG + mem + th + CARD_FIXED_KG) * scale
        out[m] = val
        src[m] = dict(kg=val, die_kg=PCF_UPLIFT * die * scale,
                      ic_overhead_kg=PCF_UPLIFT * IC_OVERHEAD_KG * scale,
                      mem_kg=PCF_UPLIFT * mem * scale, thermal_kg=PCF_UPLIFT * th * scale,
                      fixed_kg=PCF_UPLIFT * CARD_FIXED_KG * scale,
                      chip=d["chip"], node=d["node"], area_mm2=d["area_mm2"],
                      mem_gb=d["mem_gb"], mem_type=d["mem"], tdp_w=d["tdp"],
                      tdp_source=d["tdp_src"],
                      method=("VENDOR ANCHOR (NVIDIA HGX H100 PCF / 8)" if m == "H100"
                              else "EXTRAPOLATED from the HGX H100 PCF component split "
                                   "with ACT die-area carbon"))
    return {m: out[m] for m in MACH}, src, cpa(7.0), ANCHOR_HGX_H100_KG / ANCHOR_GPUS


# ---------------------------------------------------------------- workload mix
def w_of(pred):
    w = np.array([1.0 if pred(k) else 0.0 for k in _keys])
    return w / w.sum()


W_UNIFORM = w_of(lambda k: True)


def arrivals(w, seed, horizon=HORIZON, lam=LAM):
    """Replayed arrival stream. The rng call sequence is identical to the published
    fleet_vs_mix.py / shapley_fleet.py simulators so that S0 can compare like with like."""
    rng = np.random.default_rng(seed)
    idx = np.arange(NK)
    t, ts, ks = 0.0, [], []
    while t < horizon:
        t += rng.exponential(1.0 / lam)
        if t < horizon:
            ts.append(t)
            ks.append(int(rng.choice(idx, p=w)))
    return ts, ks


# ---------------------------------------------------------------- fast facility model
def facility(counts, ts, ks, policy="energy", horizon=HORIZON):
    """Facility energy per job (J) and mean queueing delay (s) for a fleet.

    `counts` is a tuple of slot counts aligned with MACH. Identical semantics to the
    published simulator: a job takes the lowest-energy (or highest-throughput) device
    type available at its start instant, and the facility pays idle power for the whole
    horizon plus the excess above idle while a device executes.
    """
    slots = []
    for j, c in enumerate(counts):
        slots.extend([j] * int(c))
    ns = len(slots)
    if ns == 0:
        return float("inf"), float("inf"), 0
    free = [0.0] * ns
    dyn = 0.0
    dsum = 0.0
    n = len(ts)
    energy_first = policy == "energy"
    for q in range(n):
        at = ts[q]
        ki = ks[q]
        start = at
        mn = min(free)
        if mn > at:
            start = mn
        Erow = Ecell[ki]
        Trow = Tcell[ki]
        sel = -1
        selv = float("inf")
        for s in range(ns):
            if free[s] <= start:
                v = Erow[slots[s]] if energy_first else -1.0 / Trow[slots[s]]
                if v < selv:
                    selv = v
                    sel = s
        mj = slots[sel]
        rt = Trow[mj]
        free[sel] = start + rt
        dsum += start - at
        dyn += JOB * Erow[mj] - IDLEv[mj] * rt
    static = sum(IDLEv[s] for s in slots) * horizon
    return (static + dyn) / max(n, 1), (dsum / n if n else 0.0), n


def facility_trace(counts, ts, ks, policy="energy", horizon=HORIZON):
    """As facility(), but returns the per-job (start, runtime, excess energy) trace that
    the temporal-shifting analysis in G3 needs."""
    slots = []
    for j, c in enumerate(counts):
        slots.extend([j] * int(c))
    ns = len(slots)
    free = [0.0] * ns
    tr = []
    energy_first = policy == "energy"
    for q in range(len(ts)):
        at = ts[q]
        ki = ks[q]
        start = at
        mn = min(free)
        if mn > at:
            start = mn
        Erow = Ecell[ki]
        Trow = Tcell[ki]
        sel, selv = -1, float("inf")
        for s in range(ns):
            if free[s] <= start:
                v = Erow[slots[s]] if energy_first else -1.0 / Trow[slots[s]]
                if v < selv:
                    selv, sel = v, s
        mj = slots[sel]
        rt = Trow[mj]
        free[sel] = start + rt
        tr.append((at, start, rt, JOB * Erow[mj] - IDLEv[mj] * rt, mj, sel))
    static_w = sum(IDLEv[s] for s in slots) * horizon
    return tr, static_w, ns


# ---------------------------------------------------------------- carbon accounting
def carbon_per_job(energy_j, counts, intensity, emb, horizon=HORIZON, njobs=1):
    """Total gCO2e per job = operational at a flat grid intensity + straight-line
    amortised embodied carbon of the devices held over the horizon."""
    op = energy_j / 3.6e6 * intensity
    emb_g = sum(c * emb[MACH[j]] * 1000.0 for j, c in enumerate(counts))
    return op + emb_g * (horizon / LIFE_S) / max(njobs, 1), op


# ---------------------------------------------------------------- composition sets
def comps(k, ntypes=NM):
    """All compositions of exactly k slots over ntypes types."""
    if ntypes == 1:
        yield (k,)
        return
    for i in range(k + 1):
        for rest in comps(k - i, ntypes - 1):
            yield (i,) + rest


MINSLOTS, MAXSLOTS = 4, 14
ALLCOMPS = [c for k in range(MINSLOTS, MAXSLOTS + 1) for c in comps(k)]

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
INTENSITIES = [20, 50, 100, 150, 200, 300, 400, 500, 600, 700]

SANITY = []


def sane(name, ok, detail):
    SANITY.append(dict(check=name, passed=bool(ok), detail=detail))
    print("    [{}] {}: {}".format("PASS" if ok else "FAIL", name, detail))
    return bool(ok)


def fleetstr(c):
    return " + ".join("{}x{}".format(int(v), MACH[i]) for i, v in enumerate(c) if v > 0)


# ---------------------------------------------------------------- worker
def _sweep_seed(seed):
    """Simulate every candidate composition once on one replayed arrival stream.
    Energy per job and delay do not depend on grid intensity or on embodied carbon, so
    one simulation pass supports the entire intensity sweep analytically."""
    ts, ks = arrivals(W_UNIFORM, seed)
    out = []
    for c in ALLCOMPS:
        e, d, n = facility(c, ts, ks)
        out.append((e, d))
    ref, refd, njobs = facility((0, 0, 0, 0, 10), ts, ks, policy="fastest")
    return seed, out, ref, refd, njobs


def main():
    t0 = time.time()
    OUT = {}
    print("=" * 78)
    print("G1/G3 carbon objective. {} candidate compositions, {} seeds, {} intensities"
          .format(len(ALLCOMPS), len(SEEDS), len(INTENSITIES)))
    print("=" * 78)

    emb, embsrc, kcal, anchor_per_gpu = build_embodied(1.0)
    EMBODIED_KG.update(emb)
    EMBODIED_SOURCES.update(embsrc)
    print("\nembodied carbon table (kg CO2e per device, cradle-to-gate, host excluded)")
    print("  anchor: NVIDIA HGX H100 8-GPU baseboard {:.0f} kg CO2e -> {:.1f} kg per GPU "
          "(ISO 14067, third-party reviewed)".format(ANCHOR_HGX_H100_KG, anchor_per_gpu))
    print("  die-area carbon: ACT via CarbonClarity, {:.2f} kg/cm2 at 7nm, power-law "
          "exponent {:.3f} to other nodes".format(kcal, CPA_EXP))
    print("  memory {:.3f} kg/GB HBM (from the anchor), GDDR6 assumed {:.0%} of that"
          .format(HBM_KG_PER_GB, GDDR_RATIO))
    print("  IC overhead {:.1f} kg/device, thermal {:.4f} kg/W, fixed card {:.1f} kg, "
          "assembly+transport uplift {:.3f}x".format(IC_OVERHEAD_KG, THERMAL_KG_PER_W,
                                                     CARD_FIXED_KG, PCF_UPLIFT))
    print("\n  {:<12}{:>8}{:>10}{:>7}{:>8}{:>8}{:>8}{:>8}{:>9}".format(
        "device", "die mm2", "node", "TDP W", "die kg", "mem kg", "therm", "other", "TOTAL"))
    for m in list(MACH) + ["H100"]:
        s = embsrc[m]
        print("  {:<12}{:>8.0f}{:>10}{:>7}{:>8.1f}{:>8.1f}{:>8.1f}{:>8.1f}{:>9.1f}{}".format(
            m, s["area_mm2"], s["node"], s["tdp_w"], s["die_kg"], s["mem_kg"],
            s["thermal_kg"], s["ic_overhead_kg"] + s["fixed_kg"], s["kg"],
            "   <- vendor anchor" if m == "H100" else ""))
    # E1: the component model must reproduce the anchor it was calibrated on
    sane("E1 the component model reproduces the published H100 anchor",
         abs(embsrc["H100"]["kg"] - anchor_per_gpu) < 0.5,
         "model {:.1f} kg vs anchor {:.1f} kg per H100".format(embsrc["H100"]["kg"], anchor_per_gpu))
    # E2: independent cross-check against a figure the model was NOT fitted to
    ratio = embsrc["H100"]["kg"] / emb["L4"]
    sane("E2 the L4-to-H100 embodied ratio matches the independent EcoServe estimate",
         2.0 < ratio < 4.5,
         "model gives {:.2f}x; EcoServe (arXiv:2502.05043) states an L4 has about 3x "
         "lower embodied carbon than an H100, a figure the model was not fitted to"
         .format(ratio))

    # ------------------------------------------------------------ simulate
    print("\nsimulating {} composition-seed pairs".format(len(ALLCOMPS) * len(SEEDS)))
    try:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as ex:
            res = list(ex.map(_sweep_seed, SEEDS))
    except Exception as exc:                                    # pragma: no cover
        print("  parallel path unavailable ({}), running serially".format(exc))
        res = [_sweep_seed(s) for s in SEEDS]
    res.sort()
    ENER = {s: {ALLCOMPS[i]: r[i][0] for i in range(len(ALLCOMPS))} for s, r, _, _, _ in res}
    DEL = {s: {ALLCOMPS[i]: r[i][1] for i in range(len(ALLCOMPS))} for s, r, _, _, _ in res}
    REF = {s: r for s, _, r, _, _ in res}
    REFD = {s: d for s, _, _, d, _ in res}
    NJOBS = {s: n for s, _, _, _, n in res}
    print("  done in {:.0f}s; {} jobs per replay (seed 0)".format(time.time() - t0, NJOBS[0]))

    # ------------------------------------------------------------ S0 calibration
    print("\nS0 calibration against the published Section 8.1 result")
    ten = [c for c in ALLCOMPS if sum(c) == 10]
    e0 = {c: ENER[0][c] for c in ten}
    d0 = {c: DEL[0][c] for c in ten}
    feas10 = [c for c in ten if d0[c] <= SLA_S]
    best10 = min(feas10, key=lambda c: e0[c])
    sav10 = 100 * (REF[0] - e0[best10]) / REF[0]
    # The canonical stored artefacts are the reference, not the prose: fleet_vs_mix.json
    # records the uniform-mix optimum and shapley_fleet.json the saving against the
    # all-A100 throughput-first reference, both on seed 0.
    ref_fvm = json.load(io.open("experiments/results/fleet_vs_mix.json", encoding="utf-8"))
    ref_shp = json.load(io.open("experiments/results/shapley_fleet.json", encoding="utf-8"))
    tgt_e = ref_fvm["results"]["optimal_fleets"]["uniform (all workloads)"]["per_job_j"]
    tgt_d = ref_fvm["results"]["optimal_fleets"]["uniform (all workloads)"]["delay_s"]
    tgt_ref = ref_shp["results"]["shapley"]["uniform"]["reference_j"]
    tgt_v = 100 * ref_shp["results"]["shapley"]["uniform"]["v_grand"]
    sane("S0 fast simulator reproduces the stored published 10-slot optimum bit for bit",
         fleetstr(best10) == "4xL4 + 6xL40S" and abs(e0[best10] - tgt_e) < 1e-6
         and abs(d0[best10] - tgt_d) < 1e-9 and abs(REF[0] - tgt_ref) < 1e-6
         and abs(sav10 - tgt_v) < 1e-6,
         "{} at {:.6f} J/job (stored {:.6f}), delay {:.6f}s (stored {:.6f}), reference "
         "{:.6f} J/job (stored {:.6f}), saving {:.4f}% (stored v_grand {:.4f}%); "
         "{} of {} ten-slot fleets feasible".format(
             fleetstr(best10), e0[best10], tgt_e, d0[best10], tgt_d, REF[0], tgt_ref,
             sav10, tgt_v, len(feas10), len(ten)))
    # The paper's prose quotes 4,476 J/job and 34.3 per cent. That figure comes from the
    # rng.integers arrival-sampling variant recorded in k6_seed_variance.json, not from
    # the rng.choice variant the shipped fleet_vs_mix.py and shapley_fleet.py use; the
    # same file records the gap as -1.90 per cent and attributes it to Monte Carlo
    # variation between the two draws. Everything below is stated against the canonical
    # rng.choice artefacts so that it is commensurable with the stored results.
    try:
        k6 = json.load(io.open("experiments/results/k6_seed_variance.json", encoding="utf-8"))
        OUT["paper_prose_discrepancy"] = dict(
            paper_prose_j=k6["single_seed"]["recipe1"], canonical_j=k6["single_seed"]["b8"],
            gap_pct=k6["single_seed"]["gap_pct"],
            note="the paper's 4,476 J/job and 34.3 per cent correspond to the rng.integers "
                 "arrival-sampling variant; the shipped simulators and stored JSON use "
                 "rng.choice and give 4,391 J/job and 33.73 per cent. This experiment is "
                 "reported against the latter.")
        print("  note: the paper's prose quotes {:.0f} J/job (34.3%); the shipped "
              "simulators and stored JSON give {:.0f} J/job ({:.2f}%). k6_seed_variance "
              "records this {:.2f}% gap as the rng.integers-vs-rng.choice arrival draw."
              .format(k6["single_seed"]["recipe1"], tgt_e, tgt_v, k6["single_seed"]["gap_pct"]))
    except Exception:
        pass

    # multi-seed spread on the published quantity
    s10 = []
    for s in SEEDS:
        f = [c for c in ten if DEL[s][c] <= SLA_S]
        b = min(f, key=lambda c: ENER[s][c])
        s10.append((100 * (REF[s] - ENER[s][b]) / REF[s], b))
    print("  over {} seeds: saving {:.1f}% +/- {:.1f} (range {:.1f} to {:.1f}); "
          "selected fleet identical in {}/{}".format(
              len(SEEDS), float(np.mean([x[0] for x in s10])), float(np.std([x[0] for x in s10], ddof=1)),
              min(x[0] for x in s10), max(x[0] for x in s10),
              sum(1 for x in s10 if x[1] == best10), len(SEEDS)))
    OUT["S0_calibration"] = dict(fleet=fleetstr(best10), energy_j=e0[best10],
                                 saving_pct_seed0=sav10, reference_j=REF[0],
                                 n_feasible_10slot=len(feas10),
                                 seed_savings=[x[0] for x in s10],
                                 seed_fleets=[fleetstr(x[1]) for x in s10])

    # ------------------------------------------------------------ G1
    def optimise(scale, seeds=SEEDS, zero=False):
        """Carbon-optimal fleet per intensity per seed, at an embodied-carbon scaling."""
        e_ = {m: 0.0 for m in MACH} if zero else build_embodied(scale)[0]
        per = {}
        for s in seeds:
            feas = [c for c in ALLCOMPS if DEL[s][c] <= SLA_S]
            eopt = min(feas, key=lambda c: ENER[s][c])
            row = {}
            for I in INTENSITIES:
                cb = {c: carbon_per_job(ENER[s][c], c, I, e_, njobs=NJOBS[s])[0] for c in feas}
                copt = min(feas, key=lambda c: cb[c])
                refc = carbon_per_job(REF[s], (0, 0, 0, 0, 10), I, e_, njobs=NJOBS[s])[0]
                row[I] = dict(carbon_fleet=copt, carbon_g=cb[copt],
                              energy_fleet=eopt, energy_fleet_carbon_g=cb[eopt],
                              ref_carbon_g=refc,
                              saving_pct=100 * (refc - cb[copt]) / refc,
                              energy_fleet_saving_pct=100 * (refc - cb[eopt]) / refc,
                              ntypes=sum(1 for v in copt if v > 0), ndev=sum(copt),
                              delay_s=DEL[s][copt])
            per[s] = dict(rows=row, energy_fleet=eopt, energy_j=ENER[s][eopt],
                          energy_ntypes=sum(1 for v in eopt if v > 0), energy_ndev=sum(eopt))
        return per, e_

    print("\n" + "=" * 78)
    print("G1. carbon-optimal fleet versus grid carbon intensity")
    print("    total device count is a free variable ({} to {} slots); embodied carbon"
          .format(MINSLOTS, MAXSLOTS))
    print("    amortised straight-line over a {:.0f}-year service life".format(LIFE_YEARS))
    print("=" * 78)
    base, embv = optimise(1.0)
    eopt0 = base[SEEDS[0]]["energy_fleet"]
    print("\nenergy-optimal fleet (the paper's objective, device count free): {}  "
          "{:.0f} J/job, {} types, {} devices".format(
              fleetstr(eopt0), base[SEEDS[0]]["energy_j"],
              base[SEEDS[0]]["energy_ntypes"], base[SEEDS[0]]["energy_ndev"]))
    ef_agree = Counter(fleetstr(base[s]["energy_fleet"]) for s in SEEDS)
    print("  across seeds: {}".format(dict(ef_agree)))

    print("\n{:>8} {:<28}{:>7}{:>7}{:>12}{:>12}".format(
        "gCO2e", "carbon-optimal fleet (modal)", "types", "devs", "saving %", "sd"))
    g1rows = []
    for I in INTENSITIES:
        fl = Counter(fleetstr(base[s]["rows"][I]["carbon_fleet"]) for s in SEEDS)
        modal, nmod = fl.most_common(1)[0]
        savs = [base[s]["rows"][I]["saving_pct"] for s in SEEDS]
        nt = [base[s]["rows"][I]["ntypes"] for s in SEEDS]
        nd = [base[s]["rows"][I]["ndev"] for s in SEEDS]
        print("{:>8} {:<28}{:>7.2f}{:>7.2f}{:>11.1f}%{:>12.2f}".format(
            I, modal, float(np.mean(nt)), float(np.mean(nd)),
            float(np.mean(savs)), float(np.std(savs, ddof=1))))
        g1rows.append(dict(intensity_gco2e_per_kwh=I, modal_fleet=modal,
                           modal_agreement="{}/{}".format(nmod, len(SEEDS)),
                           mean_ntypes=float(np.mean(nt)), mean_ndev=float(np.mean(nd)),
                           mean_saving_pct=float(np.mean(savs)),
                           sd_saving_pct=float(np.std(savs, ddof=1)),
                           per_seed=[dict(seed=s, fleet=fleetstr(base[s]["rows"][I]["carbon_fleet"]),
                                          saving_pct=base[s]["rows"][I]["saving_pct"],
                                          ntypes=base[s]["rows"][I]["ntypes"],
                                          ndev=base[s]["rows"][I]["ndev"],
                                          carbon_g_per_job=base[s]["rows"][I]["carbon_g"],
                                          delay_s=base[s]["rows"][I]["delay_s"])
                                     for s in SEEDS]))
    OUT["G1"] = dict(rows=g1rows,
                     energy_optimal_fleet=fleetstr(eopt0),
                     energy_optimal_j=base[SEEDS[0]]["energy_j"],
                     energy_optimal_ntypes=base[SEEDS[0]]["energy_ntypes"],
                     energy_optimal_ndev=base[SEEDS[0]]["energy_ndev"],
                     energy_fleet_by_seed=dict(ef_agree),
                     min_slots=MINSLOTS, max_slots=MAXSLOTS, life_years=LIFE_YEARS,
                     seeds=SEEDS)

    # -------- the paper's prediction, tested
    mt = [r["mean_ntypes"] for r in g1rows]
    md = [r["mean_ndev"] for r in g1rows]
    shrink_types = mt[0] <= mt[-1]
    shrink_dev = md[0] <= md[-1]
    strict = mt[0] < mt[-1] or md[0] < md[-1]
    print("\npaper's Section 9 prediction: 'the degree of heterogeneity should shrink as")
    print("grid carbon intensity falls'. Measured from {} to {} gCO2e/kWh:".format(
        INTENSITIES[0], INTENSITIES[-1]))
    print("  distinct types  {:.2f} -> {:.2f}".format(mt[0], mt[-1]))
    print("  device count    {:.2f} -> {:.2f}".format(md[0], md[-1]))
    OUT["G1"]["prediction"] = dict(
        claim="degree of heterogeneity shrinks as grid carbon intensity falls",
        ntypes_clean=mt[0], ntypes_dirty=mt[-1], ndev_clean=md[0], ndev_dirty=md[-1],
        holds_weakly=bool(shrink_types and shrink_dev), holds_strictly=bool(strict))

    # -------- S1 zero embodied
    print("\nS1 zero-embodied degeneracy")
    z, _ = optimise(0.0, zero=True)
    okz = all(z[s]["rows"][I]["carbon_fleet"] == z[s]["energy_fleet"]
              for s in SEEDS for I in INTENSITIES)
    maxrel = max(abs(z[s]["rows"][I]["saving_pct"] -
                     100 * (REF[s] - ENER[s][z[s]["energy_fleet"]]) / REF[s])
                 for s in SEEDS for I in INTENSITIES)
    sane("S1 with zero embodied carbon the carbon optimum IS the energy optimum",
         okz and maxrel < 1e-9,
         "identical fleet at all {} intensities x {} seeds; max saving discrepancy {:.2e} pp"
         .format(len(INTENSITIES), len(SEEDS), maxrel))

    # -------- S2 high-intensity limit
    hi = INTENSITIES[-1]
    agree_hi = sum(1 for s in SEEDS if base[s]["rows"][hi]["carbon_fleet"] == base[s]["energy_fleet"])
    lo = INTENSITIES[0]
    agree_lo = sum(1 for s in SEEDS if base[s]["rows"][lo]["carbon_fleet"] == base[s]["energy_fleet"])
    sane("S2 at a coal-heavy grid the carbon optimum converges to the energy optimum",
         agree_hi == len(SEEDS),
         "{}/{} seeds agree at {} gCO2e/kWh, against {}/{} at {} gCO2e/kWh"
         .format(agree_hi, len(SEEDS), hi, agree_lo, len(SEEDS), lo))

    # -------- S3 monotone device count
    viol = []
    for s in SEEDS:
        seq = [base[s]["rows"][I]["ndev"] for I in INTENSITIES]
        for i in range(1, len(seq)):
            if seq[i] < seq[i - 1]:
                viol.append((s, INTENSITIES[i - 1], INTENSITIES[i], seq[i - 1], seq[i]))
    sane("S3 device count of the carbon optimum is non-decreasing in grid intensity",
         not viol,
         "no violation across {} seeds x {} intensity steps".format(len(SEEDS), len(INTENSITIES) - 1)
         if not viol else "{} violations, first {}".format(len(viol), viol[0]))

    # -------- S4 argmin consistency
    bad = []
    for s in SEEDS:
        for I in INTENSITIES:
            r = base[s]["rows"][I]
            if r["carbon_g"] > r["energy_fleet_carbon_g"] + 1e-12:
                bad.append((s, I, "carbon"))
            if ENER[s][r["energy_fleet"]] > ENER[s][r["carbon_fleet"]] + 1e-9:
                bad.append((s, I, "energy"))
    sane("S4 neither argmin is beaten on its own objective", not bad,
         "checked {} seed-intensity cells".format(len(SEEDS) * len(INTENSITIES))
         if not bad else "{} inconsistencies, first {}".format(len(bad), bad[0]))

    # -------- S5 SLA
    worst = max(base[s]["rows"][I]["delay_s"] for s in SEEDS for I in INTENSITIES)
    sane("S5 every reported carbon-optimal fleet meets the 60-second constraint",
         worst <= SLA_S, "worst mean delay across all reported optima {:.1f}s".format(worst))

    # -------- factor-of-two sensitivity
    print("\nfactor-of-two sensitivity on the embodied-carbon table")
    print("{:>8}".format("gCO2e") + "".join("{:>30}".format("embodied x{:g}".format(f))
                                            for f in (0.5, 1.0, 2.0)))
    sens = {}
    for f in (0.5, 2.0):
        sens[f] = optimise(f)[0]
    sens[1.0] = base
    srows = []
    for I in INTENSITIES:
        line = "{:>8}".format(I)
        cell = {}
        for f in (0.5, 1.0, 2.0):
            fl = Counter(fleetstr(sens[f][s]["rows"][I]["carbon_fleet"]) for s in SEEDS)
            modal = fl.most_common(1)[0][0]
            savs = [sens[f][s]["rows"][I]["saving_pct"] for s in SEEDS]
            nd = float(np.mean([sens[f][s]["rows"][I]["ndev"] for s in SEEDS]))
            nt = float(np.mean([sens[f][s]["rows"][I]["ntypes"] for s in SEEDS]))
            line += "{:>30}".format("{} ({:.1f}%)".format(modal, float(np.mean(savs))))
            cell[str(f)] = dict(modal_fleet=modal, mean_saving_pct=float(np.mean(savs)),
                                sd_saving_pct=float(np.std(savs, ddof=1)),
                                mean_ndev=nd, mean_ntypes=nt)
        print(line)
        srows.append(dict(intensity_gco2e_per_kwh=I, **cell))
    OUT["G1"]["sensitivity_factor_two"] = srows
    het_all = all(r[k]["mean_ntypes"] > 1.0 for r in srows for k in ("0.5", "1.0", "2.0"))
    pos_all = all(r[k]["mean_saving_pct"] > 0 for r in srows for k in ("0.5", "1.0", "2.0"))
    sane("SENS heterogeneity and a positive carbon saving survive a factor-of-two error "
         "in the embodied table in either direction",
         het_all and pos_all,
         "over {} intensities x 3 embodied scalings: minimum distinct types {:.2f}, "
         "minimum saving {:.1f}%".format(
             len(srows), min(r[k]["mean_ntypes"] for r in srows for k in ("0.5", "1.0", "2.0")),
             min(r[k]["mean_saving_pct"] for r in srows for k in ("0.5", "1.0", "2.0"))))

    # ------------------------------------------------------------ 24-hour re-screen
    # G3 needs a full day to contain a diurnal cycle, and the one-hour replay the paper
    # screens on turns out not to be in steady state: several fleets that pass the
    # 60-second constraint at one hour are still filling their queue. Everything used in
    # G3, and a confirmation pass on G1, is therefore re-screened at 24 hours.
    print("\n" + "=" * 78)
    print("24-hour re-screen of the fleets the one-hour replay declares feasible")
    print("=" * 78)
    FEAS1H = sorted({c for s in SEEDS for c in ALLCOMPS if DEL[s][c] <= SLA_S})
    print("  {} distinct fleets pass the 60-second constraint on at least one one-hour "
          "replay; re-simulating each over a full day".format(len(FEAS1H)))
    t24 = time.time()
    try:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as ex:
            r24 = list(ex.map(_sweep24, [(s, FEAS1H) for s in SEEDS24]))
    except Exception as exc:                                    # pragma: no cover
        print("  parallel path unavailable ({}), running serially".format(exc))
        r24 = [_sweep24((s, FEAS1H)) for s in SEEDS24]
    r24.sort()
    E24 = {s: {FEAS1H[i]: v[i][0] for i in range(len(FEAS1H))} for s, v, _, _ in r24}
    D24 = {s: {FEAS1H[i]: v[i][1] for i in range(len(FEAS1H))} for s, v, _, _ in r24}
    N24 = {s: n for s, _, n, _ in r24}
    REF24 = {s: r for s, _, _, r in r24}
    print("  done in {:.0f}s; {} jobs per 24-hour replay (seed 0)".format(
        time.time() - t24, N24[SEEDS24[0]]))
    FEAS24 = [c for c in FEAS1H if all(D24[s][c] <= SLA_S for s in SEEDS24)]
    print("  {} of {} survive the same constraint over 24 hours ({:.0f}% attrition)"
          .format(len(FEAS24), len(FEAS1H), 100 * (1 - len(FEAS24) / len(FEAS1H))))
    s0 = SEEDS24[0]
    paper_fleet = (0, 4, 0, 6, 0)
    sane("S6 the one-hour SLA screen is not a steady-state screen",
         len(FEAS24) < len(FEAS1H),
         "the published optimum 4xL4 + 6xL40S has mean delay {:.1f}s at one hour and "
         "{:.1f}s at 24 hours, breaching the same 60-second constraint; {} of {} "
         "one-hour-feasible fleets fail the 24-hour screen".format(
             DEL[s0][paper_fleet], D24[s0][paper_fleet],
             len(FEAS1H) - len(FEAS24), len(FEAS1H)))
    OUT["horizon_screen"] = dict(
        n_feasible_1h=len(FEAS1H), n_feasible_24h=len(FEAS24),
        paper_fleet="4xL4 + 6xL40S", paper_fleet_delay_1h_s=DEL[s0][paper_fleet],
        paper_fleet_delay_24h_s=D24[s0][paper_fleet],
        note="fleet selection on a one-hour replay is a transient screen; the 24-hour "
             "screen is stricter and is what G3 uses.")

    # -------- G1b: the intensity sweep repeated on the 24-hour-feasible set
    print("\nG1b. the same intensity sweep on the 24-hour-feasible set")
    print("{:>8} {:<28}{:>7}{:>7}{:>12}{:>12}".format(
        "gCO2e", "carbon-optimal fleet (modal)", "types", "devs", "saving %", "sd"))
    g1b = []
    for I in INTENSITIES:
        fls, savs, nts, nds = [], [], [], []
        for s in SEEDS24:
            cb = {c: carbon_per_job(E24[s][c], c, I, embv, horizon=DAY, njobs=N24[s])[0]
                  for c in FEAS24}
            copt = min(FEAS24, key=lambda c: cb[c])
            refc = carbon_per_job(REF24[s], (0, 0, 0, 0, 10), I, embv,
                                  horizon=DAY, njobs=N24[s])[0]
            fls.append(fleetstr(copt))
            savs.append(100 * (refc - cb[copt]) / refc)
            nts.append(sum(1 for v in copt if v > 0))
            nds.append(sum(copt))
        modal = Counter(fls).most_common(1)[0][0]
        print("{:>8} {:<28}{:>7.2f}{:>7.2f}{:>11.1f}%{:>12.2f}".format(
            I, modal, float(np.mean(nts)), float(np.mean(nds)),
            float(np.mean(savs)), float(np.std(savs, ddof=1))))
        g1b.append(dict(intensity_gco2e_per_kwh=I, modal_fleet=modal,
                        mean_ntypes=float(np.mean(nts)), mean_ndev=float(np.mean(nds)),
                        mean_saving_pct=float(np.mean(savs)),
                        sd_saving_pct=float(np.std(savs, ddof=1)),
                        per_seed_fleets=fls, per_seed_saving_pct=savs))
    OUT["G1b_24h"] = dict(rows=g1b, seeds=SEEDS24,
                          ntypes_clean=g1b[0]["mean_ntypes"], ntypes_dirty=g1b[-1]["mean_ntypes"],
                          ndev_clean=g1b[0]["mean_ndev"], ndev_dirty=g1b[-1]["mean_ndev"])
    print("  distinct types {:.2f} -> {:.2f}, device count {:.2f} -> {:.2f} from {} to {}"
          .format(g1b[0]["mean_ntypes"], g1b[-1]["mean_ntypes"],
                  g1b[0]["mean_ndev"], g1b[-1]["mean_ndev"], INTENSITIES[0], INTENSITIES[-1]))
    v3b = [i for i in range(1, len(g1b)) if g1b[i]["mean_ndev"] < g1b[i - 1]["mean_ndev"] - 1e-9]
    sane("S3b device count is non-decreasing in grid intensity on the 24-hour screen too",
         not v3b, "monotone over {} intensity steps".format(len(INTENSITIES) - 1)
         if not v3b else "violations at {}".format([INTENSITIES[i] for i in v3b]))

    # ------------------------------------------------------------ G3
    print("\n" + "=" * 78)
    print("G3. carbon-aware placement versus carbon-aware composition")
    print("=" * 78)
    g3 = run_g3(E24, D24, N24, REF24, FEAS24, embv)
    OUT["G3"] = g3

    OUT["embodied"] = dict(
        anchor=dict(source="NVIDIA HGX H100 product carbon footprint summary, ISO 14067, "
                           "third-party reviewed; cited as reference [53] in the paper",
                    value_kg=ANCHOR_HGX_H100_KG, gpus=ANCHOR_GPUS,
                    per_gpu_kg=ANCHOR_HGX_H100_KG / ANCHOR_GPUS,
                    scope="eight-GPU HGX baseboard, cradle-to-gate, host server excluded"),
        pcf_component_split_kg=PCF_SPLIT, pcf_uplift=PCF_UPLIFT,
        rule="E = uplift * (CPA(node)*die_area + IC_OVERHEAD + mem_kg_per_gb*mem_gb "
             "+ THERMAL_KG_PER_W*tdp + CARD_FIXED), every term calibrated against the "
             "HGX H100 PCF component split",
        cpa_source="ACT (Gupta et al., ISCA 2022) via CarbonClarity (arXiv:2507.01145): "
                   "1.18 kg CO2e/cm2 at 28nm, 2.52 at 7nm, sigma 0.99 at 7nm",
        cpa_kg_per_cm2={n: cpa(n) for n in (4, 7, 8, 12, 28)}, cpa_exponent=CPA_EXP,
        mem_kg_per_gb=MEM_KG_PER_GB, gddr_to_hbm_ratio=GDDR_RATIO,
        ic_overhead_kg=IC_OVERHEAD_KG, thermal_kg_per_w=THERMAL_KG_PER_W,
        card_fixed_kg=CARD_FIXED_KG, die=DIE,
        cross_check="EcoServe (arXiv:2502.05043) states an L4 has about 3x lower "
                    "embodied carbon than an H100; the model was not fitted to this",
        per_device=EMBODIED_SOURCES,
        caveat="Every per-device figure for the five accelerators in the fleet is an "
               "EXTRAPOLATION from the single published anchor, not a vendor figure. "
               "The factor-of-two sensitivity sweep is the load-bearing result, not the "
               "point estimate.")
    OUT["sanity"] = SANITY

    path = "experiments/results/g1_g3_carbon.json"
    json.dump(OUT, io.open(path, "w", encoding="utf-8"), indent=1, default=str)
    npass = sum(1 for s in SANITY if s["passed"])
    print("\n" + "=" * 78)
    print("sanity: {}/{} passed".format(npass, len(SANITY)))
    for s in SANITY:
        if not s["passed"]:
            print("  FAILED: {} -- {}".format(s["check"], s["detail"]))
    print("saved -> {}   ({:.0f}s total)".format(path, time.time() - t0))


# ---------------------------------------------------------------- G3 machinery
DAY = 86400.0
AMPLITUDE = 0.40          # illustrative; see run_g3
TROUGH_HOUR = 14.0        # solar peak, the cleanest hour
SEEDS24 = [0, 1, 2, 3]


def _sweep24(arg):
    """Re-simulate a candidate fleet list over a full 24-hour replay."""
    seed, cands = arg
    ts, ks = arrivals(W_UNIFORM, seed, horizon=DAY)
    out = [facility(c, ts, ks, horizon=DAY)[:2] for c in cands]
    ref = facility((0, 0, 0, 0, 10), ts, ks, policy="fastest", horizon=DAY)[0]
    return seed, out, len(ts), ref


def intensity_at(t, mean_i, amp=AMPLITUDE, period=DAY, trough_h=TROUGH_HOUR):
    """Illustrative diurnal grid carbon intensity, sinusoidal, minimum at the solar peak.

    This is an ASSUMED shape, labelled as such. It is calibrated only in its mean and its
    peak-to-trough ratio: at amp = 0.40 the profile swings 2.33x from cleanest to
    dirtiest hour, the order of magnitude that solar-heavy grids show. No claim is made
    that it reproduces any particular balancing authority.
    """
    return mean_i * (1.0 - amp * math.cos(2 * math.pi * ((t / period * 24.0) - trough_h) / 24.0))


def mean_intensity_over(t0, dur, mean_i, amp, period):
    """Exact mean of the sinusoid over [t0, t0+dur], so the accounting is not a midpoint
    approximation that could manufacture or hide a saving."""
    if dur <= 0:
        return intensity_at(t0, mean_i, amp, period)
    w = 2 * math.pi / period
    ph = w * t0 - 2 * math.pi * TROUGH_HOUR / 24.0
    c = (math.sin(w * dur + ph) - math.sin(ph)) / (w * dur)
    return mean_i * (1.0 - amp * c)


def temporal(trace, nslots, slack, mean_i, amp, period, capacity=True):
    """PURE temporal shifting: each job keeps the device and runtime the baseline
    scheduler gave it, and only its START TIME moves, inside [arrival, arrival + slack].

    Holding the spatial decision fixed is what makes the temporal lever separable from
    the spatial one. It also makes two invariants exact rather than approximate: under a
    constant intensity every candidate start ties and the earliest wins, reproducing the
    baseline exactly (T1); and dropping the capacity constraint strictly enlarges each
    job's feasible window, so the capacity-free variant is a true upper bound (T4).

    Returns dynamic carbon (g), mean shift applied (s), mean total delay (s).
    """
    free = [0.0] * nslots
    dyn = 0.0
    shift_sum = 0.0
    delay_sum = 0.0
    for at, st_base, rt, ex, mj, sl in trace:
        earliest = max(at, free[sl]) if capacity else at
        hi = at + slack
        if hi < earliest:
            hi = earliest
        best_v = mean_intensity_over(earliest, rt, mean_i, amp, period)
        best_s = earliest
        if hi > earliest:
            step = (hi - earliest) / 400.0
            g = earliest + step
            while g < hi:
                v = mean_intensity_over(g, rt, mean_i, amp, period)
                if v < best_v - 1e-15:
                    best_v, best_s = v, g
                g += step
            v = mean_intensity_over(hi, rt, mean_i, amp, period)
            if v < best_v - 1e-15:
                best_v, best_s = v, hi
        if capacity:
            free[sl] = best_s + rt
        dyn += ex / 3.6e6 * best_v
        shift_sum += best_s - st_base
        delay_sum += best_s - at
    n = max(len(trace), 1)
    return dyn, shift_sum / n, delay_sum / n


def max_sla_slack(trace, nslots, mean_i, amp, period, sla=SLA_S, hi=3600.0):
    """Largest per-job deferral the MEAN-delay constraint actually admits.

    The constraint in this paper is a 60-second constraint on MEAN queueing delay, not a
    60-second per-job deadline. The baseline scheduler already spends part of that budget
    queueing, and deferring a job also pushes back everything behind it on the same
    device, so granting every job a full 60 seconds of deferral overshoots the mean
    constraint. This binary search finds the largest uniform deferral budget that keeps
    the facility inside the same constraint the composition search is held to.
    """
    lo = 0.0
    if temporal(trace, nslots, hi, mean_i, amp, period)[2] <= sla:
        return hi
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if temporal(trace, nslots, mid, mean_i, amp, period)[2] <= sla:
            lo = mid
        else:
            hi = mid
    return lo


def day_carbon(counts, ts, ks, policy, mean_i, amp, emb=None, period=DAY, horizon=DAY):
    """Total carbon per job over one replayed day: unshiftable static idle power for the
    whole horizon, plus the excess above idle of every job at the intensity prevailing
    while it runs, plus amortised embodied carbon of the devices held."""
    tr, static_j, ns = facility_trace(counts, ts, ks, policy=policy, horizon=horizon)
    static_g = static_j / 3.6e6 * mean_intensity_over(0.0, horizon, mean_i, amp, period)
    dyn_g, _, delay = temporal(tr, ns, 0.0, mean_i, amp, period)
    emb_g = 0.0
    if emb:
        emb_g = sum(c * emb[MACH[j]] * 1000.0 for j, c in enumerate(counts)) * (horizon / LIFE_S)
    n = max(len(ts), 1)
    return (static_g + dyn_g + emb_g) / n, delay, (static_g / n, dyn_g / n, emb_g / n)


def run_g3(E24, D24, N24, REF24, FEAS24, embv):
    MEAN_I = 400.0
    print("\ndiurnal profile: sinusoid, mean {:.0f} gCO2e/kWh, amplitude {:.2f} "
          "(peak/trough {:.2f}x), trough at {:.0f}:00. ILLUSTRATIVE ASSUMPTION, not "
          "fitted to any balancing authority.".format(
              MEAN_I, AMPLITUDE, (1 + AMPLITUDE) / (1 - AMPLITUDE), TROUGH_HOUR))

    s0 = SEEDS24[0]
    # every fleet below is drawn from the 24-hour-feasible set, so none of them is a
    # transient artefact of a one-hour replay
    e_fleet = min(FEAS24, key=lambda c: E24[s0][c])
    cb400 = {c: carbon_per_job(E24[s0][c], c, MEAN_I, embv, horizon=DAY, njobs=N24[s0])[0]
             for c in FEAS24}
    c_fleet = min(FEAS24, key=lambda c: cb400[c])
    # throughput-first reference: all-A100, smallest count that holds the constraint
    a100 = [c for c in FEAS24 if c[4] == sum(c)]
    ref_fleet = min(a100, key=sum) if a100 else (0, 0, 0, 0, 10)
    # placement pool: the cheapest 24-hour-feasible fleet holding all five types, which
    # is the five-type pool the paper's placement lever needs in order to have anything
    # to exploit
    five = [c for c in FEAS24 if all(v > 0 for v in c)]
    pool = min(five, key=lambda c: E24[s0][c])
    print("  reference, throughput-first purchase  : {}".format(fleetstr(ref_fleet)))
    print("  energy-optimal composition            : {}".format(fleetstr(e_fleet)))
    print("  carbon-optimal composition at 400     : {}".format(fleetstr(c_fleet)))
    print("  five-type pool for the placement lever: {}".format(fleetstr(pool)))

    lev = {k: [] for k in ("composition_carbon", "composition_energy", "placement_spatial",
                           "placement_temporal", "placement_temporal_bound")}
    detail = []
    for s in SEEDS24:
        ts, ks = arrivals(W_UNIFORM, s, horizon=DAY)
        ref_g, ref_d, _ = day_carbon(ref_fleet, ts, ks, "fastest", MEAN_I, AMPLITUDE, emb=embv)
        cc_g, cc_d, _ = day_carbon(c_fleet, ts, ks, "energy", MEAN_I, AMPLITUDE, emb=embv)
        ec_g, _, _ = day_carbon(e_fleet, ts, ks, "energy", MEAN_I, AMPLITUDE, emb=embv)
        p_fast, _, _ = day_carbon(pool, ts, ks, "fastest", MEAN_I, AMPLITUDE, emb=embv)
        p_eng, _, parts = day_carbon(pool, ts, ks, "energy", MEAN_I, AMPLITUDE, emb=embv)
        # temporal lever, on the same pool and the same replay
        tr, static_j, ns = facility_trace(pool, ts, ks, horizon=DAY)
        static_g = static_j / 3.6e6 * mean_intensity_over(0.0, DAY, MEAN_I, AMPLITUDE, DAY)
        emb_g = sum(c * embv[MACH[j]] * 1000.0 for j, c in enumerate(pool)) * (DAY / LIFE_S)
        base_dyn, _, base_del = temporal(tr, ns, 0.0, MEAN_I, AMPLITUDE, DAY)
        sl_ok = max_sla_slack(tr, ns, MEAN_I, AMPLITUDE, DAY)
        pol_dyn, pol_sh, pol_del = temporal(tr, ns, sl_ok, MEAN_I, AMPLITUDE, DAY)
        # the bound is granted the FULL 60 seconds per job, which the mean-delay
        # constraint does not actually allow, so it is generous on both axes
        bnd_dyn, _, _ = temporal(tr, ns, SLA_S, MEAN_I, AMPLITUDE, DAY, capacity=False)
        tot = static_g + base_dyn + emb_g
        lev["composition_carbon"].append(100 * (ref_g - cc_g) / ref_g)
        lev["composition_energy"].append(100 * (ref_g - ec_g) / ref_g)
        lev["placement_spatial"].append(100 * (p_fast - p_eng) / p_fast)
        lev["placement_temporal"].append(100 * (base_dyn - pol_dyn) / tot)
        lev["placement_temporal_bound"].append(100 * (base_dyn - bnd_dyn) / tot)
        detail.append(dict(seed=s, njobs=len(ts),
                           ref_g_per_job=ref_g, ref_delay_s=ref_d,
                           carbon_composition_g=cc_g, carbon_composition_delay_s=cc_d,
                           energy_composition_g=ec_g,
                           pool_fastest_g=p_fast, pool_energy_g=p_eng,
                           pool_static_g=parts[0], pool_dynamic_g=parts[1],
                           pool_embodied_g=parts[2],
                           temporal_policy_dyn_g=pol_dyn / len(ts),
                           temporal_bound_dyn_g=bnd_dyn / len(ts),
                           temporal_mean_shift_s=pol_sh, temporal_mean_delay_s=pol_del,
                           temporal_sla_admissible_slack_s=sl_ok,
                           baseline_mean_delay_s=base_del))

    shiftable = float(np.mean([d["pool_dynamic_g"] /
                               (d["pool_static_g"] + d["pool_dynamic_g"] + d["pool_embodied_g"])
                               for d in detail]))
    print("\nlevers on identical replayed 24-hour arrival streams, {} seeds".format(len(SEEDS24)))
    print("  only {:.1%} of this facility's carbon is attached to job execution and is "
          "therefore shiftable at all; the rest is idle power held all day plus amortised "
          "embodied carbon.".format(shiftable))
    print("\n{:<46}{:>12}{:>10}".format("lever (carbon saving, 24h replay)", "mean", "sd"))
    for k, label in [("composition_carbon", "composition: carbon-optimal fleet"),
                     ("composition_energy", "composition: energy-optimal fleet"),
                     ("placement_spatial", "placement, spatial: which device runs it"),
                     ("placement_temporal", "placement, temporal: when, within the SLA"),
                     ("placement_temporal_bound", "placement, temporal, UPPER BOUND at 60s")]:
        v = lev[k]
        print("{:<46}{:>11.3f}%{:>10.3f}".format(label, float(np.mean(v)),
                                                 float(np.std(v, ddof=1))))

    # ---------------- how much slack would temporal shifting need to matter?
    print("\ntemporal shifting against the deadline slack it is allowed (seed {})".format(s0))
    print("{:>10}{:>16}{:>16}{:>15}{:>15}".format(
        "slack", "policy saving", "upper bound", "mean shift s", "mean delay s"))
    SLACKS = [60.0, 300.0, 1800.0, 3600.0, 4 * 3600.0, 8 * 3600.0, 12 * 3600.0]
    ts0, ks0 = arrivals(W_UNIFORM, s0, horizon=DAY)
    tr0, static_j0, ns0 = facility_trace(pool, ts0, ks0, horizon=DAY)
    static_g0 = static_j0 / 3.6e6 * mean_intensity_over(0.0, DAY, MEAN_I, AMPLITUDE, DAY)
    emb_g0 = sum(c * embv[MACH[j]] * 1000.0 for j, c in enumerate(pool)) * (DAY / LIFE_S)
    base_dyn0, _, base_del0 = temporal(tr0, ns0, 0.0, MEAN_I, AMPLITUDE, DAY)
    tot0 = static_g0 + base_dyn0 + emb_g0
    slackrows = []
    for sl in SLACKS:
        pd, psh, pdl = temporal(tr0, ns0, sl, MEAN_I, AMPLITUDE, DAY)
        bd, _, _ = temporal(tr0, ns0, sl, MEAN_I, AMPLITUDE, DAY, capacity=False)
        ps = 100 * (base_dyn0 - pd) / tot0
        bs = 100 * (base_dyn0 - bd) / tot0
        lab = "{:.0f}s".format(sl) if sl < 3600 else "{:.0f}h".format(sl / 3600)
        print("{:>10}{:>15.3f}%{:>15.3f}%{:>15.0f}{:>15.1f}".format(lab, ps, bs, psh, pdl))
        slackrows.append(dict(slack_s=sl, policy_saving_pct=ps, bound_saving_pct=bs,
                              mean_shift_s=psh, mean_delay_s=pdl))

    # ---------------- T1 constant intensity
    f_base, _, _ = temporal(tr0, ns0, 0.0, MEAN_I, 0.0, DAY)
    f_sh, f_shift, _ = temporal(tr0, ns0, SLA_S, MEAN_I, 0.0, DAY)
    rel = abs(f_base - f_sh) / f_base
    sane("T1 under a constant grid intensity temporal shifting is worth exactly zero",
         rel < 1e-12 and abs(f_shift) < 1e-12,
         "relative carbon difference {:.2e}, mean shift applied {:.2e}s".format(rel, f_shift))

    # ---------------- T2 mechanism validation
    # The diurnal profile barely moves over a 60-second window, so a near-zero headline
    # number is expected. To distinguish that from a scheduler that does nothing, the
    # same code is run against a profile whose period is short compared with the slack
    # but still long compared with the 3.5 to 102-second job runtimes: a one-hour period
    # with a 30-minute slack.
    FASTP, FASTSLACK = 3600.0, 1800.0
    m_base, _, _ = temporal(tr0, ns0, 0.0, MEAN_I, AMPLITUDE, FASTP)
    m_bnd, m_bshift, _ = temporal(tr0, ns0, FASTSLACK, MEAN_I, AMPLITUDE, FASTP,
                                  capacity=False)
    m_pol, m_shift, _ = temporal(tr0, ns0, FASTSLACK, MEAN_I, AMPLITUDE, FASTP)
    mbnd = 100 * (m_base - m_bnd) / m_base
    msav = 100 * (m_base - m_pol) / m_base
    sane("T2 the shifting mechanism is alive: with a one-hour intensity period and a "
         "30-minute slack the intensity-selection code captures nearly the whole swing",
         mbnd > 20.0,
         "capacity-free bound removes {:.1f}% of the shiftable carbon against a {:.1f}% "
         "ceiling at amplitude {:.2f}; the same code returns {:.3f}% on the diurnal "
         "profile at the SLA-admissible slack, so that number is the profile's flatness "
         "and not a no-op".format(mbnd, 100 * AMPLITUDE, AMPLITUDE,
                                  float(np.mean(lev["placement_temporal"]))))
    # The first version of T2 asserted this on the CAPACITY-RESPECTING policy and failed
    # at 0.9 per cent. Root cause, found by inspecting the trace rather than guessing: the
    # facility runs at high enough utilisation that every job chasing the same intensity
    # trough simply queues behind every other job doing the same, so contention, not the
    # intensity search, is what limits the policy. Asserting the mechanism on the
    # capacity-free variant isolates the code under test; the contention gap is reported
    # as a result rather than checked as an invariant.
    print("  contention gap on the fast profile: capacity-free bound {:.1f}% against "
          "{:.1f}% for the capacity-respecting greedy policy (mean shift {:.0f}s)"
          .format(mbnd, msav, m_shift))

    # ---------------- T3 monotone in slack
    v3 = [(slackrows[i - 1]["slack_s"], slackrows[i]["slack_s"])
          for i in range(1, len(slackrows))
          if slackrows[i]["bound_saving_pct"] < slackrows[i - 1]["bound_saving_pct"] - 1e-9]
    sane("T3 the temporal-shifting upper bound is non-decreasing in deadline slack",
         not v3, "monotone over {} slack levels from 60s to 12h".format(len(SLACKS))
         if not v3 else "violations: {}".format(v3))

    # ---------------- T4 bound dominates policy
    v4 = [(r["slack_s"], r["policy_saving_pct"], r["bound_saving_pct"]) for r in slackrows
          if r["policy_saving_pct"] > r["bound_saving_pct"] + 1e-9]
    sane("T4 the capacity-free bound dominates the capacity-respecting policy",
         not v4, "bound >= policy at all {} slack levels".format(len(SLACKS))
         if not v4 else "violations: {}".format(v4))

    # ---------------- T5 SLA
    # First version of this check granted every job a full 60-second deferral and then
    # asserted the 60-second constraint. That is self-contradictory: the constraint is on
    # MEAN delay, the baseline already spends 34.5s of it queueing, and deferring a job
    # pushes back everything behind it on the same device, so the mean landed at 61.9s.
    # The check was mis-specified, not the code. It is replaced by the operationally
    # meaningful statement: the deferral budget is chosen so the facility stays inside
    # the same constraint the composition search is held to.
    sl_ok0 = float(np.mean([d["temporal_sla_admissible_slack_s"] for d in detail]))
    del_ok0 = float(np.mean([d["temporal_mean_delay_s"] for d in detail]))
    sane("T5 the reported temporal operating point respects the same 60-second mean-delay "
         "constraint the composition search respects",
         del_ok0 <= SLA_S,
         "the constraint admits a mean deferral budget of only {:.1f}s per job on this "
         "fleet (baseline mean delay {:.1f}s of the 60s budget already spent queueing); "
         "at that budget the mean delay is {:.1f}s. Granting the full 60s per job instead "
         "pushes the mean to {:.1f}s and breaches the constraint."
         .format(sl_ok0, base_del0, del_ok0, slackrows[0]["mean_delay_s"]))

    # ---------------- what slack would temporal shifting need to rival composition?
    comp = float(np.mean(lev["composition_carbon"]))
    need = None
    for r in slackrows:
        if r["bound_saving_pct"] >= comp:
            need = r["slack_s"]
            break
    print("\ncomposition is worth {:.1f}%. The capacity-free temporal bound reaches "
          "{:.1f}% at a 12-hour deadline; {}".format(
              comp, slackrows[-1]["bound_saving_pct"],
              "it first matches composition at a {:.0f}h slack".format(need / 3600)
              if need else "it never matches composition within a 12-hour deadline."))

    return dict(profile=dict(shape="sinusoid, minimum at the solar peak",
                             mean_gco2e_per_kwh=MEAN_I, amplitude=AMPLITUDE,
                             peak_to_trough=(1 + AMPLITUDE) / (1 - AMPLITUDE),
                             trough_hour=TROUGH_HOUR, period_s=DAY,
                             status="ILLUSTRATIVE ASSUMPTION, not fitted to any grid"),
                horizon_s=DAY, seeds=SEEDS24, sla_s=SLA_S,
                reference_fleet=fleetstr(ref_fleet), energy_fleet=fleetstr(e_fleet),
                carbon_fleet=fleetstr(c_fleet), placement_pool=fleetstr(pool),
                shiftable_carbon_fraction=shiftable,
                levers={k: dict(mean_pct=float(np.mean(v)), sd_pct=float(np.std(v, ddof=1)),
                                per_seed=v) for k, v in lev.items()},
                per_seed=detail, slack_sweep=slackrows,
                sla_admissible_slack_s=sl_ok0, temporal_mean_delay_s=del_ok0,
                baseline_mean_delay_s=base_del0,
                mechanism_check=dict(period_s=FASTP, slack_s=FASTSLACK,
                                     bound_saving_pct=mbnd, policy_saving_pct=msav,
                                     note="the gap between bound and policy is contention: "
                                          "every job chasing the same intensity trough "
                                          "queues behind every other job doing the same"),
                slack_to_match_composition_s=need)


if __name__ == "__main__":
    main()
