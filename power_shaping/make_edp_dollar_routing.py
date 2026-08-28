# -*- coding: utf-8 -*-
"""Three new energy-based metrics on the cap-sweep data (control mode, batch 32), for A100/A10G (+T4/L4):
  1. EDP / ED2P optimal cap  -- energy*latency and energy*latency^2 minima (the 'balanced' operating point).
  2. Dollar-optimal vs joule-optimal -- under per-hour cloud billing, cost/inference ~ 1/throughput, so the
     cost-optimal cap is UNCAPPED (max throughput) = the energy-WORST point. Quantify the divergence.
  3. Iso-latency cheapest-energy routing across GPUs -- for a target latency, which (GPU,cap) is greenest.
Saves data/raw/edp_dollar.csv. Prices: per-hour on-demand (A100 = p4d.24xlarge/8).
"""
import os
import numpy as np
import pandas as pd
from predict_power import RAW, saturating_set

TDP = {"T4": 70, "L4": 72, "A10G": 300, "A100": 400}
PRICE_HR = {"T4": 0.526, "L4": 0.8048, "A10G": 1.006, "A100": 32.77 / 8}   # USD/hr per GPU
KWH = 0.10                                                                  # datacenter $/kWh for operator power


def load():
    fr = [pd.read_csv(os.path.join(RAW, "aws_sweep_batch.csv")),
          pd.read_csv(os.path.join(RAW, "aws_sweep_a100_batch.csv"))]
    df = pd.concat(fr, ignore_index=True)
    df["card"] = df.gpu.str.replace("NVIDIA ", "", regex=False).str.replace("Tesla ", "", regex=False)
    df["card"] = df.card.str.replace("A100-SXM4-40GB", "A100", regex=False).str.strip()
    df = saturating_set(df)                                   # loaded (production-relevant) regime
    # Mean the repetitions per operating point before any argmin (see make_efficiency_table.load).
    df = df.groupby(["card", "workload", "cap_w"], as_index=False).agg(
        power_w=("power_w", "mean"), throughput=("throughput", "mean"))
    df["E_J"] = df.power_w / df.throughput
    df["lat_ms"] = 1000.0 / df.throughput
    return df


def per_model(df):
    rows = []
    for (card, wl), g in df.groupby(["card", "workload"]):
        g = g.sort_values("cap_w")
        mx = g.loc[g.throughput.idxmax()]                                   # $-optimal cap = max throughput (computed, not assumed)
        E = g.E_J.values; L = g.lat_ms.values
        ke = int(np.argmin(E)); kdp = int(np.argmin(E * L)); kd2 = int(np.argmin(E * L * L))
        rows.append(dict(card=card, model=wl,
                         cap_Emin=100 * g.cap_w.values[ke] / TDP[card],
                         cap_EDP=100 * g.cap_w.values[kdp] / TDP[card],
                         cap_ED2P=100 * g.cap_w.values[kd2] / TDP[card],
                         cap_dollar=100 * mx.cap_w / TDP[card],
                         E_at_dollar=mx.E_J, E_at_Emin=E[ke],
                         dollar_penalty_of_green=mx.throughput / g.throughput.values[ke]))  # =cards mult
    return pd.DataFrame(rows)


def iso_latency(df, model, targets_ms):
    # across ALL cards+caps, cheapest energy that meets latency <= target
    d = df[df.workload == model]
    out = []
    for L in targets_ms:
        ok = d[d.lat_ms <= L]
        if len(ok) == 0:
            out.append((L, None, None, None, None)); continue
        r = ok.loc[ok.E_J.idxmin()]
        out.append((L, r.card, 100 * r.cap_w / TDP[r.card], r.E_J, r.lat_ms))
    return out


def main():
    df = load()
    t = per_model(df); t.to_csv(os.path.join(RAW, "edp_dollar.csv"), index=False)

    print("== 1. Balanced optima: median optimal cap (% TDP) per card ==")
    print(f"  {'card':5} {'E-min':>6} {'EDP':>6} {'ED2P':>6} {'$-opt(=max perf)':>16}")
    for card in ["T4", "L4", "A10G", "A100"]:
        s = t[t.card == card]
        print(f"  {card:5} {s.cap_Emin.median():>5.0f}% {s.cap_EDP.median():>5.0f}% {s.cap_ED2P.median():>5.0f}%"
              f" {s.cap_dollar.median():>15.0f}%")

    print("\n== 2. Dollar-optimal vs joule-optimal (per-hour billing) ==")
    for card in ["A100", "A10G"]:
        s = t[t.card == card]
        green_dollar = s.dollar_penalty_of_green.median()          # extra $ to run green (=cards multiplier)
        # operator power-cost angle: energy saved at E-min vs uncapped, valued at $/kWh over the fleet
        esave = 100 * (1 - (s.E_at_Emin / s.E_at_dollar)).median()
        print(f"  {card}: $-optimal cap = {s.cap_dollar.median():.0f}% TDP (uncapped, max throughput) = the energy-WORST point.")
        print(f"        Running at the joule-optimal cap costs +{100*(green_dollar-1):.0f}% dollars (renter, per-hour) "
              f"to save {esave:.0f}% energy (operator, power bill).")

    print("\n== 3. Iso-latency cheapest-energy routing (llm_decode) ==")
    print(f"  {'target latency':>14} -> {'greenest GPU':>12} {'@cap':>6} {'energy/step':>11} {'actual lat':>10}")
    for L, card, cap, E, lat in iso_latency(df, "llm_decode", [40, 50, 60, 80, 120]):
        if card is None:
            print(f"  {L:>12}ms -> (no option this fast)")
        else:
            print(f"  {L:>12}ms -> {card:>12} {cap:>4.0f}% {E:>9.2f}J {lat:>8.1f}ms")
    print("\nsaved -> data/raw/edp_dollar.csv")


if __name__ == "__main__":
    main()
