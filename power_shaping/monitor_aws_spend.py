# -*- coding: utf-8 -*-
"""
Independent AWS spend watchdog for the power-shaping data-collection campaign. Polls the
tagged instances every 90 s, estimates cumulative cost from each instance's runtime x hourly
rate, and enforces hard limits:

  * HARD STOP at $BUDGET      -> terminate every tagged instance, then keep sweeping.
  * DEAD-MAN OVERRIDE 55 min  -> terminate any single instance older than the 45-min on-box
                                 dead-man (i.e. one that failed to self-terminate).
  * CONCURRENCY > 1           -> the campaign runs one instance at a time; more = anomaly.

Exits (one clear notification) when: a breach was auto-handled, OR the campaign is finished
(no tagged instances for 3 straight polls after having seen at least one), OR a 2.5 h safety
timeout. On exit it prints the estimated total spend and confirms no orphans remain.
"""
from __future__ import annotations
import time, sys, argparse
from datetime import datetime, timezone
import boto3
from aws_powercap_probe import cleanup, TAG

import os
REGION = os.environ.get("WATCH_REGION", "us-east-1")
BUDGET = 10.0          # hard ceiling (user-authorized); override with --budget
ALERT = 6.0            # warn above this
DEADMAN_MIN = 55.0     # terminate an instance older than this (on-box dead-man is 45 min)
POLL = 90
RATES = {"g5.xlarge": 1.006, "g5.2xlarge": 1.212, "g6.xlarge": 0.8048, "g6e.xlarge": 1.861,
         "g4dn.xlarge": 0.526, "p3.2xlarge": 3.06, "a10g": 1.0,
         "p4d.24xlarge": 41.0, "p4de.24xlarge": 47.0, "p5.48xlarge": 98.32}  # p4d Tokyo rate (over-est = safer)


def main():
    global BUDGET, ALERT
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=BUDGET)
    a = ap.parse_args()
    BUDGET = a.budget; ALERT = min(ALERT, 0.6 * BUDGET)
    ec2 = boto3.client("ec2", REGION)
    seen = {}                      # iid -> {type, launch, last}
    empty_polls = 0; ever_seen = False; t_start = time.time()
    print(f"[watchdog] start; budget ${BUDGET:.0f}, alert ${ALERT:.0f}, dead-man {DEADMAN_MIN:.0f}min", flush=True)
    while True:
        try:
            r = ec2.describe_instances(Filters=[{"Name": "tag:purpose", "Values": [TAG]}])
        except Exception as e:
            print(f"[watchdog] describe error: {str(e)[:80]}", flush=True); time.sleep(POLL); continue
        insts = [i for res in r["Reservations"] for i in res["Instances"]]
        now = datetime.now(timezone.utc)
        running = []
        for i in insts:
            iid = i["InstanceId"]; st = i["State"]["Name"]; typ = i["InstanceType"]; lt = i["LaunchTime"]
            rec = seen.setdefault(iid, {"type": typ, "launch": lt, "last": lt})
            if st in ("pending", "running", "shutting-down"):
                rec["last"] = now
                running.append((iid, typ, st, (now - lt).total_seconds() / 60.0))
        cum = sum(max(0.0, (rec["last"] - rec["launch"]).total_seconds() / 3600.0) * RATES.get(rec["type"], 1.0)
                  for rec in seen.values())
        if running:
            ever_seen = True; empty_polls = 0
        else:
            empty_polls += 1

        # --- enforcement ---
        breach = None
        old = [(iid, age) for iid, _, _, age in running if age > DEADMAN_MIN]
        if cum >= BUDGET:
            ids = [iid for iid, _, _, _ in running]
            if ids:
                ec2.terminate_instances(InstanceIds=ids)
            breach = f"HARD STOP: cumulative ${cum:.2f} >= ${BUDGET:.0f}; terminated {ids}"
        elif old:
            ids = [iid for iid, _ in old]
            ec2.terminate_instances(InstanceIds=ids)
            breach = f"DEAD-MAN OVERRIDE: {old} older than {DEADMAN_MIN:.0f}min; terminated {ids}"
        elif len(running) > 1:
            print(f"[watchdog] WARN concurrency={len(running)} (expected 1): "
                  f"{[(t, round(a)) for _, t, _, a in running]}", flush=True)

        status = f"[watchdog] cum≈${cum:.2f} | running={[(t, round(a)) for _, t, _, a in running]}"
        if breach:
            print(f"{status}\n[watchdog] !!! {breach}", flush=True)
            print("[watchdog] sweeping all tagged resources ...", flush=True)
            cleanup(ec2, tag_sweep=True); print(f"[watchdog] EXIT on breach. est. total ${cum:.2f}", flush=True); return
        if cum >= ALERT:
            print(f"{status}  <-- above alert ${ALERT:.0f}", flush=True)
        elif running:
            print(status, flush=True)

        if ever_seen and empty_polls >= 3:
            ok = cleanup(ec2, tag_sweep=True)
            print(f"[watchdog] campaign complete. est. total spend ${cum:.2f}. "
                  f"orphans: {'NONE' if ok else 'STRAGGLERS REMAIN'}", flush=True)
            return
        if time.time() - t_start > 9000:
            print(f"[watchdog] 2.5h safety timeout; est. ${cum:.2f}; final sweep", flush=True)
            cleanup(ec2, tag_sweep=True); return
        time.sleep(POLL)


if __name__ == "__main__":
    main()
