# -*- coding: utf-8 -*-
"""Verify that every headline figure/table number is reproducible from the CORRECT SUBSET of the consolidated
master table (data/processed/elf_master.csv). For each claim we state the subset it should come from, recompute
it straight from the master, and compare to the value the paper builder produces. A PASS means the figure/table
uses the same data subset as the single source of truth."""
import numpy as np
import pandas as pd

M = pd.read_csv("data/processed/elf_master.csv")
import build_gptenergy as B   # the paper's built values

TDP = {"T4": 70, "L4": 72, "A10G": 300, "A100": 400}
checks = []


def chk(name, got, exp, tol, subset):
    ok = abs(got - exp) <= tol
    checks.append((ok, name, got, exp, subset))


def joule_pct(sub):
    """median energy-optimal cap (%TDP) over workloads, from a subset of the master (one row per cap level)."""
    js = []
    for (c, w), g in sub.groupby(["card", "workload"]):
        g = g.sort_values("cap_w")
        js.append(g.loc[g.energy_J_per_step.idxmin(), "cap_pct_tdp"])
    return float(np.median(js))


def fix_penalty(sub, card):
    """mean energy penalty (%) of holding every workload at the per-card median Joule cap vs its own optimum."""
    oc = []
    for w, g in sub.groupby("workload"):
        g = g.sort_values("cap_w"); oc.append(g.loc[g.energy_J_per_step.idxmin(), "cap_pct_tdp"])
    fixcap = float(np.median(oc)); fixw = fixcap / 100 * TDP[card]; pen = []
    for w, g in sub.groupby("workload"):
        g = g.sort_values("cap_w"); cw = g.cap_w.values; E = g.energy_J_per_step.values
        k = int(np.argmin(np.abs(cw - fixw))); pen.append(100 * (E[k] - E.min()) / E.min())
    return fixcap, float(np.mean(pen)), float(np.std(oc))


# --- 1. Joule point per card: loaded_primary subset (Fig 4/5, FIX, section 6) ---
for card, exp in [("A100", B.FIX["A100"]["cap"]), ("A10G", B.FIX["A10G"]["cap"])]:
    sub = M[(M.card == card) & (M.loaded_primary)]
    fc, pen, sd = fix_penalty(sub, card)
    chk(f"Joule cap {card}", fc, exp, 2.0, "loaded_primary")
    chk(f"FIX penalty {card}", pen, B.FIX[card]["pen_mean"], 0.6, "loaded_primary")
    chk(f"Fig5 sd {card}", sd, B.FIX[card]["sd"], 0.6, "loaded_primary")

# --- 2. beta medians per card: control_primary fitted law (Fig 1, section 5, BETA) ---
for card in ["T4", "L4", "A10G", "A100"]:
    sub = M[(M.card == card) & (M.control_primary)].drop_duplicates(["workload"])
    got = float(sub.fit_beta.median())
    chk(f"beta median {card}", got, B.BETA["card"][card][0], 0.3, "control_primary fit_beta")

# --- 3. Fig 2 same-rate spread: control_primary A10G, 20-24 inf/s band ---
sr = M[(M.card == "A10G") & (M.control_primary) & (M.throughput >= 20) & (M.throughput <= 24)]
chk("Fig2 pmin", float(sr.power_w.min()), B.SR["pmin"], 2.0, "A10G control_primary, 20-24 inf/s")
chk("Fig2 pmax", float(sr.power_w.max()), B.SR["pmax"], 2.0, "A10G control_primary, 20-24 inf/s")

# --- 4. clock/DVFS savings: CLOCK rows (section 10, clk) ---
for card, key in [("T4", "T4_med"), ("L4", "L4_med")]:
    sub = M[(M.card == card) & (M.actuator == "clock")]
    sv = []
    for w, g in sub.groupby("workload"):
        g = g.sort_values("clock_mhz"); E = g.energy_J_per_step.values
        sv.append(100 * (1 - E.min() / E[-1]))
    chk(f"clock saving {card}", float(np.median(sv)), B.clk[key], 1.5, "actuator==clock")

# --- 5. cap floor as %TDP: lowest reachable cap where draw still <= cap (section 10, clk floor) ---
for card, key in [("T4", "T4_floor"), ("L4", "L4_floor")]:
    sub = M[(M.card == card) & (M.actuator == "power_cap")]
    chk(f"cap floor {card}", float(sub.cap_pct_tdp.min()), B.clk[key], 1.0, "power_cap min cap_pct_tdp")

# --- report ---
print(f"{'claim':22} {'master':>9} {'paper':>9} {'tol':>5}  subset")
print("-" * 78)
npass = 0
for ok, name, got, exp, subset in checks:
    npass += ok
    print(f"{'PASS' if ok else 'FAIL'} {name:17} {got:>9.2f} {exp:>9.2f}  ok?   {subset}")
print("-" * 78)
print(f"{npass}/{len(checks)} checks pass")
