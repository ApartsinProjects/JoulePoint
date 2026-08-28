# -*- coding: utf-8 -*-
"""
Turn the Modal pilot measurements into the preliminary-results artefacts:
  1. energy-per-sample matrix (load x machine)
  2. the reversal test: does the energy-optimal machine depend on the load?
  3. an HTML fragment for section 2.5.5.8 and a figure for the appendix
"""
import io, json, os, re, sys
from collections import defaultdict

LOG = "experiments/pilot_run.log"
OUT_JSON = "experiments/pilot_results.json"

def load_results():
    if os.path.exists(OUT_JSON):
        return json.load(io.open(OUT_JSON, encoding="utf-8"))
    raw = io.open(LOG, encoding="utf-8", errors="replace").read()
    m = re.search(r"===RESULTS_JSON_START===\s*(.*?)\s*===RESULTS_JSON_END===", raw, re.S)
    if not m:
        sys.exit("no results block in log; job may still be running or failed")
    data = json.loads(m.group(1))
    json.dump(data, io.open(OUT_JSON, "w", encoding="utf-8"), indent=1)
    return data

def main():
    data = load_results()
    rows = [r for blk in data for r in blk["rows"]]
    ok = [r for r in rows if r.get("status") == "ok" and r.get("energy_per_sample_mj")]
    oom = [r for r in rows if r.get("status") == "oom"]
    machines = [b["machine"] for b in data]
    caps = {b["machine"]: b.get("power_cap_w") for b in data}
    devs = {b["machine"]: b.get("device") for b in data}

    print(f"cells: {len(rows)} total, {len(ok)} measured, {len(oom)} memory-infeasible")
    print(f"machines: {machines}")

    # energy per sample at a fixed configuration
    for prec in ["fp16", "fp32"]:
        for bs in [32, 128, 8]:
            sub = [r for r in ok if r["precision"] == prec and r["batch"] == bs]
            if len(sub) < 6:
                continue
            tab = defaultdict(dict)
            for r in sub:
                tab[r["load"]][r["machine"]] = r["energy_per_sample_mj"]
            print(f"\n=== energy per sample (mJ), {prec}, batch {bs} ===")
            hdr = [m for m in machines if any(m in v for v in tab.values())]
            print("load".ljust(13) + "".join(m.ljust(12) for m in hdr) + "best")
            winners = {}
            for load, d in tab.items():
                best = min(d, key=d.get) if d else None
                winners[load] = best
                line = load.ljust(13)
                for m in hdr:
                    v = d.get(m)
                    line += (f"{v:.3f}" if v else "-").ljust(12)
                print(line + str(best))
            distinct = set(winners.values())
            print(f"  energy-optimal machine per load: {winners}")
            print(f"  --> {len(distinct)} distinct winner(s) across {len(winners)} loads"
                  f"{'  ** REVERSAL PRESENT **' if len(distinct) > 1 else '  (no reversal at this config)'}")

    # reversal across configurations for a single load
    print("\n=== does the optimal machine change with configuration? ===")
    byload = defaultdict(lambda: defaultdict(dict))
    for r in ok:
        byload[r["load"]][(r["precision"], r["batch"])][r["machine"]] = r["energy_per_sample_mj"]
    for load, cfgs in byload.items():
        w = {}
        for cfg, d in cfgs.items():
            if d:
                w[cfg] = min(d, key=d.get)
        if w and len(set(w.values())) > 1:
            print(f"  {load}: winner varies by config -> {w}")
        elif w:
            print(f"  {load}: {list(set(w.values()))[0]} wins at every measured config")

    # spread: how much is at stake
    print("\n=== spread between best and worst feasible machine ===")
    for load, cfgs in byload.items():
        for cfg, d in cfgs.items():
            if len(d) >= 3:
                lo, hi = min(d.values()), max(d.values())
                print(f"  {load:12} {str(cfg):16} best={lo:8.3f}  worst={hi:8.3f}  ratio={hi/lo:5.2f}x")
                break

    print("\n=== power caps / devices ===")
    for m in machines:
        print(f"  {m:12} {devs[m]:28} cap={caps[m]}W")

if __name__ == "__main__":
    main()
