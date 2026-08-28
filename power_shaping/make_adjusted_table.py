# -*- coding: utf-8 -*-
"""Throughput-adjusted comparison: serve the SAME aggregate throughput at the energy-efficient cap by adding
cards. Because energy/inference = power/throughput is already work-normalized, total energy to serve a fixed
throughput at EFF vs MAX is exactly the energy/step ratio -- the card count cancels. So the trade to run at
EFF is: (1) cards multiplier = T_max/T_eff (capex), (2) latency ratio EFF/MAX (per-request, cards don't fix
it), (3) energy ratio EFF/MAX (already throughput-adjusted; <1 = real saving). Reads efficiency_table.csv.
"""
import os
import pandas as pd
from predict_power import RAW


def main():
    t = pd.read_csv(os.path.join(RAW, "efficiency_table.csv"))
    t["cards"] = t.max_thr / t.eff_thr                     # cards needed at EFF to match one MAX card's throughput
    t["lat_ratio"] = t.eff_lat / t.max_lat                 # per-request latency EFF/MAX (>1 = slower)
    t["energy_ratio"] = t.eff_E / t.max_E                  # throughput-adjusted energy EFF/MAX (<1 = saving)
    t.to_csv(os.path.join(RAW, "adjusted_table.csv"), index=False)
    for card in ["A100", "A10G"]:
        s = t[t.card == card].sort_values("energy_ratio")
        print(f"\n===== {card}: serve SAME throughput at the energy-efficient cap (add cards) =====")
        print(f"  {'model':20} | {'cards x':>7} | {'latency ratio':>13} | {'energy ratio':>12}")
        for _, r in s.iterrows():
            print(f"  {r.model:20} | {r.cards:6.2f}x | {r.lat_ratio:11.2f}x | {r.energy_ratio:10.2f}x")
        print(f"  ---- {card} medians: cards {s.cards.median():.2f}x   latency {s.lat_ratio.median():.2f}x   "
              f"energy {s.energy_ratio.median():.2f}x  (i.e. {100*(1-s.energy_ratio.median()):.0f}% less energy, "
              f"{100*(s.lat_ratio.median()-1):.0f}% more latency, {100*(s.cards.median()-1):.0f}% more cards)")
    print("\nsaved -> data/raw/adjusted_table.csv")


if __name__ == "__main__":
    main()
