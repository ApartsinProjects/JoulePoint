# -*- coding: utf-8 -*-
"""A10G (g5.xlarge) saturating-batch 3-rep sweep, MODEL-SHARDED across N single-GPU g5 instances.

Why: the un-sharded 20-model x 8-cap x 3-rep x 2-mode A10G run needs ~55-65 min of instance time,
past both the 55-min on-instance dead-man and the 50-min orchestrator poll deadline baked into
aws_sweep_expand. A g5.xlarge has ONE A10G, so the p4d trick (shard across 8 local GPUs) is not
available; instead we shard the MODEL LIST across N parallel g5.xlarge instances. Each instance
runs models[SHARD::NSHARDS] with the exact same worker as the A100 run (aws_sweep_a100_batch.
WORKER_PY: auto-calibrate saturating batch, then control(batch32)+saturate cap sweep, REPS=3,
8 cap levels, reps grouped per model in one thermal session). Per-shard wall-clock at N=4 is
~13 min of sweep + boot, comfortably inside the UNCHANGED 50/55-min timers.

Output: data/raw/aws_sweep_a10g_shard{i}.csv per shard, merged data/raw/aws_sweep_a10g_rep3.csv.
S3-exfil / throwaway-VPC / dead-man / self-clean scaffold: reused verbatim via run_one.
NOTE: each parallel shard creates its own throwaway VPC; the default EC2 quota is 5 VPCs/region,
so N=4 plus the account's default VPC sits exactly at quota. If any other VPC exists in the
region, run with --nshards 3 (7/7/6 models per shard, ~18 min sweep each, still ample margin).
"""
from __future__ import annotations
import argparse, os, csv, base64
import boto3
from aws_sweep_expand import run_one, ensure_bucket
from aws_sweep_a100_batch import WORKER_PY
from aws_powercap_probe import cleanup
from aws_powercap_sweep import latest_pytorch_dlami

HEADER = "gpu,workload,mode,batch,cap_frac,cap_w,rep,power_w,util,t_compute_ms,throughput"
FIELDS = HEADER.split(",")

# Single-worker variant of the a100 USER_DATA: same dead-man, S3 PUT, console-blob fallback and
# self-shutdown as aws_sweep_expand.USER_DATA_TMPL; adds persistence mode + the SHARD env vars.
# {B64},{SHARD},{NSHARDS} are substituted here; {PUT_URL} is substituted inside run_one.
USER_DATA_TMPL = r"""#!/bin/bash
exec > >(tee -a /dev/console) 2>&1
( sleep 3300; shutdown -h now ) &      # 55-min dead-man; a shard needs ~15-20 min
sleep 25
echo "SWEEP_BOOT_OK shard={SHARD}/{NSHARDS}"
echo "{B64}" | base64 -d > /root/worker.py
nvidia-smi -pm 1
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null && conda activate pytorch 2>/dev/null
PY=python3
for cand in "$(which python)" /opt/conda/envs/pytorch/bin/python /opt/conda/bin/python /opt/pytorch/bin/python /usr/bin/python3; do
  if [ -n "$cand" ] && $cand -c "import torch, torchvision" 2>/dev/null; then PY=$cand; break; fi
done
echo "SWEEP_PYTHON=$PY"
$PY -c "import torch, torchvision" 2>/dev/null || { echo "SWEEP_FATAL_NO_TORCH_PY=$PY"; sleep 3; shutdown -h now; }
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 GPU_PHYS=0 SHARD={SHARD} NSHARDS={NSHARDS} $PY /root/worker.py > /root/body.csv 2>/root/err.log; echo "SWEEP_PY_EXIT=$?"
{ echo "gpu,workload,mode,batch,cap_frac,cap_w,rep,power_w,util,t_compute_ms,throughput"; cat /root/body.csv; } > /root/out.csv
echo "SWEEP_ROWS=$(wc -l < /root/out.csv)"
gzip -c /root/out.csv > /root/out.csv.gz
if [ "{PUT_URL}" != NONE ]; then
  for a in 1 2 3 4 5; do
    if curl -sSf --connect-timeout 15 --max-time 300 -X PUT --upload-file /root/out.csv.gz "{PUT_URL}"; then echo "SWEEP_UPLOADED bytes=$(stat -c%s /root/out.csv.gz)"; break; else echo "SWEEP_UPLOAD_RETRY=$a"; sleep 5; fi
  done
fi
echo "SWEEP_ERR_HEAD"; head -c 1500 /root/err.log; echo
echo "SWEEP_B64_START"
gzip -c /root/out.csv | base64 -w 76 | awk '{printf "B64[%05d]%s\n", NR, $0}'
echo "SWEEP_B64_END"
sleep 3; shutdown -h now
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nshards", type=int, default=4, help="parallel g5.xlarge instances; models split [i::N]")
    ap.add_argument("--instance-type", default="g5.xlarge")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--no-final-cleanup", action="store_true")
    args = ap.parse_args()
    region, n = args.region, args.nshards
    ec2 = boto3.client("ec2", region_name=region)
    ami = latest_pytorch_dlami(ec2)
    bucket, _ = ensure_bucket(region)
    b64 = base64.b64encode(WORKER_PY.encode()).decode()
    print(f"[aws] A10G SHARDED batch sweep ami={ami} type={args.instance_type} nshards={n} bucket=s3://{bucket}")
    os.makedirs("data/raw", exist_ok=True)

    def do_shard(i):
        c = boto3.client("ec2", region_name=region); s3c = boto3.client("s3", region_name=region)
        ud = (USER_DATA_TMPL.replace("{B64}", b64)
              .replace("{SHARD}", str(i)).replace("{NSHARDS}", str(n)))
        rows, gpu = run_one(c, args.instance_type, ami, ud, bucket=bucket, s3=s3c,
                            s3_prefix=f"sweeps/a10g-shard{i}")
        for d in rows:
            d.setdefault("gpu", gpu)
        p = f"data/raw/aws_sweep_a10g_shard{i}.csv"
        if rows:
            with open(p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                w.writeheader(); [w.writerow(d) for d in rows]
        print(f"[aws] shard {i}: {len(rows)} rows -> {p}")
        return rows

    all_rows = []
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=n) as ex:
            futs = {ex.submit(do_shard, i): i for i in range(n)}
            for fut in as_completed(futs):
                try:
                    all_rows += fut.result()
                except Exception as e:
                    print(f"[aws] shard {futs[fut]} FAILED: {e}")
        out_path = "data/raw/aws_sweep_a10g_rep3.csv"
        if all_rows:
            all_rows.sort(key=lambda d: (d["workload"], d["mode"], float(d["cap_w"]), int(d["rep"])))
            with open(out_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                w.writeheader(); [w.writerow(d) for d in all_rows]
        wls = {d["workload"] for d in all_rows}
        reps = sorted({d["rep"] for d in all_rows})
        print(f"\n[aws] A10G SHARDED TOTAL {len(all_rows)} rows, {len(wls)}/20 workloads, reps={reps} -> {out_path}")
        missing = 20 - len(wls)
        if missing:
            print(f"[aws] WARNING: {missing} workloads missing -- re-run only the failed shard(s) "
                  f"(shard i covers models [i::{n}] of the MAKERS order)")
    finally:
        if not args.no_final_cleanup:
            print("[aws] final orphan sweep ..."); cleanup(ec2, tag_sweep=True)


if __name__ == "__main__":
    main()
