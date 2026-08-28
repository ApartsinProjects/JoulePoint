# -*- coding: utf-8 -*-
"""Splice the sharded A10G 3-rep sweep into data/raw/aws_sweep_batch.csv, replacing the old
A10G 1-rep block. Validates schema/coverage BEFORE touching the target and backs it up first.
Run after aws_sweep_a10g_shard.py has produced data/raw/aws_sweep_a10g_rep3.csv (or the
per-shard files, which it will concatenate itself if the merged file is absent)."""
import glob, os, shutil
import pandas as pd

RAW = "E:/Projects/DataCenterEnergy/power_shaping/data/raw"
TARGET = f"{RAW}/aws_sweep_batch.csv"
MERGED = f"{RAW}/aws_sweep_a10g_rep3.csv"

if os.path.exists(MERGED):
    new = pd.read_csv(MERGED)
else:
    parts = sorted(glob.glob(f"{RAW}/aws_sweep_a10g_shard*.csv"))
    assert parts, "no shard CSVs found"
    new = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    print(f"concatenated {len(parts)} shard files")

# ---- validate the new A10G block before splicing ----
assert set(new["gpu"]) == {"NVIDIA A10G"}, set(new["gpu"])
assert sorted(new["rep"].unique()) == [0, 1, 2], sorted(new["rep"].unique())
assert new["workload"].nunique() == 20, sorted(new["workload"].unique())
assert new["cap_w"].nunique() == 8, sorted(new["cap_w"].unique())        # 100..300 W on A10G
cnt = new.groupby(["workload", "mode"]).size()
bad = cnt[cnt != 24]                                                      # 8 caps x 3 reps per (workload, mode)
assert bad.empty, f"incomplete (workload,mode) cells:\n{bad}"
per_wl_modes = new.groupby("workload")["mode"].nunique()
print(f"A10G new block: {len(new)} rows; "
      f"{int((per_wl_modes == 2).sum())} workloads with control+saturate, "
      f"{int((per_wl_modes == 1).sum())} control-only (saturating batch == 32)")

# ---- splice: drop the old A10G block, append the new one ----
base = pd.read_csv(TARGET)
old_a10g = (base["gpu"] == "NVIDIA A10G").sum()
keep = base[base["gpu"] != "NVIDIA A10G"]
assert keep["gpu"].nunique() == 2 and len(keep) == len(base) - old_a10g   # T4 + L4 stay untouched
out = pd.concat([keep, new[base.columns]], ignore_index=True)

bak = TARGET.replace(".csv", "_a10g1rep.bak.csv")
shutil.copy2(TARGET, bak)
out.to_csv(TARGET, index=False)
print(f"replaced {old_a10g} A10G 1-rep rows with {len(new)} A10G 3-rep rows")
print(f"{TARGET}: {len(base)} -> {len(out)} rows (backup: {bak})")
