# -*- coding: utf-8 -*-
"""Dedup the AWS sweep CSVs: the serial console echoes each line twice (tee + cloud-init),
so measurement rows are doubled and a header echo leaks in. Drop the stray header row, coerce
numerics, and remove exact-duplicate measurements. Idempotent."""
import os
import pandas as pd

RAW = os.path.join(os.path.dirname(__file__), "data", "raw")
NUM = ["cap_frac", "cap_w", "rep", "power_w", "throughput"]


def clean(path, keys):
    if not os.path.exists(path):
        print(f"  {os.path.basename(path)}: absent, skip"); return
    d = pd.read_csv(path)
    d = d[d["workload"] != "workload"]                       # drop echoed header row
    for c in NUM:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["power_w", "throughput"]).drop_duplicates(subset=keys)
    d.to_csv(path, index=False)
    nwl = d["workload"].nunique()
    print(f"  {os.path.basename(path)}: {len(d)} rows, {nwl} workloads"
          + (f", {d['gpu'].nunique()} GPUs" if "gpu" in d.columns else ""))


def main():
    print("== cleaning AWS sweep CSVs ==")
    clean(os.path.join(RAW, "own_aws_sweep.csv"), ["workload", "cap_frac", "rep", "power_w", "throughput"])
    clean(os.path.join(RAW, "aws_sweep_multi.csv"), ["gpu", "workload", "cap_frac", "rep", "power_w", "throughput"])


if __name__ == "__main__":
    main()
