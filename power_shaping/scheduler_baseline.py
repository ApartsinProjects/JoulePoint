# -*- coding: utf-8 -*-
"""
Dedicated priority-scheduler baselines under the SAME strict grid cap (answers the "the 44%->0% win is
just priority scheduling" objection). Under STRICT every controller is clamped to the same cap_hard, so
we can isolate the SCHEDULING discipline: we run the class-blind controller (no deferral, gate = admit
all) with four serving disciplines and compare, then show what our full controller (priority ladder +
deferral) adds ON TOP.

  fifo            first-come-first-served (the naive default)
  strict-priority serve the critical class first
  edf             earliest-deadline-first across classes
  wfq             weighted fair queueing (serve the class most starved relative to its weight)
  ours            our controller: priority serving PLUS deferral of the low-priority classes

Honest expectation (pre-registered): strict-priority, EDF and WFQ should all protect the critical class
about as well as our serving order, so the critical-SLO win is standard priority scheduling; our controller's
ADDED value is the deferral that lowers weighted service cost, on top of the grid-compliance guarantee and
rebound control that a scheduler does not provide.
"""
from __future__ import annotations
import os, json
import pocb_sim as S


def run_at(dip_frac, seed=0):
    reqs = S.load_requests(seed=seed)
    T = int(3600 / S.DT)
    ref = S.uncontrolled_peak(reqs, T=T)
    C = S.envelope(T, ref, dip_frac=dip_frac)

    def one(ctrl, serve):
        m, _ = S.simulate(reqs, C, ctrl, seed=seed, serve=serve)
        cr = m["critical"]["slo_violation_pct"]; it = m["interactive"]["slo_violation_pct"]
        return {"crit_slo": round(cr, 1), "inter_slo": round(it, 1),
                "wcost": round(m["weighted_service_cost"], 0),
                "compliance": round(m["envelope"]["compliance_pct"], 1)}

    rec = {"dip_frac": dip_frac}
    # pure schedulers: class-blind controller (gate admits all, no deferral), varying discipline
    rec["fifo"] = one(S.ctrl_uniform, "fifo")
    rec["strict_priority"] = one(S.ctrl_uniform, "priority")
    rec["edf"] = one(S.ctrl_uniform, "edf")
    rec["wfq"] = one(S.ctrl_uniform, "wfq")
    # our full controller: priority serving + deferral of low-priority classes
    rec["ours_priority_plus_defer"] = one(S.ctrl_priority, "priority")
    return rec


def main():
    rows = [run_at(d) for d in (0.5, 0.45, 0.4)]
    out = {"rows": rows}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/scheduler_baseline.json", "w"), indent=2)

    print("== Scheduler baselines under the same strict cap ==")
    hdr = f"{'dip':>5} {'policy':>26} {'crit%':>7} {'inter%':>7} {'wcost':>10} {'compl%':>7}"
    print(hdr)
    for r in rows:
        for k in ["fifo", "strict_priority", "edf", "wfq", "ours_priority_plus_defer"]:
            v = r[k]
            print(f"{r['dip_frac']:>5} {k:>26} {v['crit_slo']:>6.1f} {v['inter_slo']:>6.1f} {v['wcost']:>10.0f} {v['compliance']:>6.1f}")
        print("-" * len(hdr))
    print("written -> results/scheduler_baseline.json")


if __name__ == "__main__":
    main()
