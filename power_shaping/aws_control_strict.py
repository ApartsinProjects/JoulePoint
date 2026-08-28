# -*- coding: utf-8 -*-
"""
STRENGTHENED measured closed-loop control on real GPUs (addresses the review: several GPUs,
several target trajectories, strict-envelope telemetry). The grid ceiling is enforced by the
HARDWARE power cap (nvidia-smi -pl), so realized power is guaranteed <= target -- this grounds
the paper's hardware lower-bound guarantee (section 4) and strict-compliance controller on real
silicon. A latency-critical stream and a deferrable throughput job share the GPU; a duty-cycle
controller protects critical p95 latency by yielding the GPU to it when the cap is tight.

For each of three target trajectories (single step, staircase, linear ramp), repeated, we log a
1 Hz trace and compute per-segment telemetry: fraction of time above target (envelope violations),
max exceedance, cap actuation latency, and critical p95/p99 latency. Runs across A10G / L4 / T4.
Self-cleaning (throwaway VPC/subnet, AZ-retry, per-GPU teardown, final tag sweep). Nothing left.

Outputs: data/raw/aws_control_strict.csv (traces), results/aws_control_strict.json (telemetry).
"""
from __future__ import annotations
import argparse, time, re, os, base64, io, csv, json
import boto3
from botocore.exceptions import ClientError
from aws_powercap_probe import cleanup, TAG, TAGS
from aws_powercap_sweep import latest_pytorch_dlami

CONTROL_PY = r'''
import time, subprocess, threading, sys
import numpy as np, torch
import pynvml
pynvml.nvmlInit(); H = pynvml.nvmlDeviceGetHandleByIndex(0)
def power(): return pynvml.nvmlDeviceGetPowerUsage(H)/1000.0
name = pynvml.nvmlDeviceGetName(H)
name = name.decode() if isinstance(name, bytes) else name
defW = int(pynvml.nvmlDeviceGetPowerManagementDefaultLimit(H)/1000)
lo, hi = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(H); minW = int(lo/1000)
subprocess.run(["sudo","nvidia-smi","-pm","1"], capture_output=True)
def set_cap(w):
    w = int(max(minW, min(defW, w)))
    subprocess.run(["sudo","nvidia-smi","-pl",str(w)], capture_output=True); return w
print(f"CTRL_META gpu={name} default={defW} min={minW}", flush=True)

dev = torch.device("cuda")
crit_a = torch.randn(256, 2048, device=dev, dtype=torch.float16)
crit_b = torch.randn(2048, 2048, device=dev, dtype=torch.float16)
big_a  = torch.randn(8192, 8192, device=dev, dtype=torch.float16)
big_b  = torch.randn(8192, 8192, device=dev, dtype=torch.float16)

st = {"run": True, "duty": 1.0, "lat": [], "dcount": 0}
def defer_loop():
    while st["run"]:
        d = st["duty"]
        if d > 0.03:
            t0 = time.time(); c = big_a @ big_b; torch.cuda.synchronize(); st["dcount"] += 1
            el = time.time() - t0; time.sleep(max(0.0, el * (1.0/d - 1.0)))
        else:
            time.sleep(0.02)
def crit_loop():
    while st["run"]:
        t0 = time.time(); c = crit_a @ crit_b; torch.cuda.synchronize()
        st["lat"].append((time.time(), (time.time()-t0)*1000.0)); time.sleep(0.004)

# power sampler at ~25 Hz
samp = {"on": True, "v": []}
def samp_loop():
    while samp["on"]:
        samp["v"].append((time.time(), power())); time.sleep(0.04)

def p95(xs, q=95):
    return float(np.percentile(xs, q)) if xs else float("nan")

# trajectories: list of (duration_s, target_fraction_of_default). ramp is expanded to 1s steps.
def traj_step():      return [(12,1.0),(30,0.55),(12,1.0)]
def traj_staircase(): return [(9,1.0),(10,0.80),(10,0.65),(10,0.50),(9,1.0)]
def traj_ramp():
    segs=[(8,1.0)]
    for k in range(12): segs.append((1.5, 1.0 - 0.5*(k+1)/12))   # ramp down to 0.5
    for k in range(12): segs.append((1.5, 0.5 + 0.5*(k+1)/12))   # ramp back up
    segs.append((8,1.0)); return segs
TRAJ = {"step": traj_step(), "staircase": traj_staircase(), "ramp": traj_ramp()}
REPS = 3

def run(segs, base_p95):
    st["run"]=True; st["duty"]=1.0; st["lat"]=[]; st["dcount"]=0
    th=[threading.Thread(target=defer_loop,daemon=True), threading.Thread(target=crit_loop,daemon=True)]
    [x.start() for x in th]; time.sleep(3)
    samp["on"]=True; samp["v"]=[]; sth=threading.Thread(target=samp_loop,daemon=True); sth.start()
    t0=time.time(); rows=[]; seg_marks=[]; last_dc=st["dcount"]; last_t=t0; tcur=0.0
    for dur, frac in segs:
        tgt=set_cap(frac*defW); seg_start=time.time(); seg_marks.append((seg_start-t0, tgt, dur))
        while time.time()-seg_start < dur:
            time.sleep(1.0)
            now=time.time(); t=now-t0
            recent=[l for (ts,l) in st["lat"] if ts>now-1.0]
            dthru=(st["dcount"]-last_dc)/max(1e-6,now-last_t); last_dc=st["dcount"]; last_t=now
            rows.append((round(t,1), tgt, round(power(),1), round(p95(recent),2), round(dthru,2)))
            # controller: protect critical p95 by yielding deferrable when latency inflates
            cp=p95(recent)
            if cp==cp and base_p95>0:
                if cp > 1.6*base_p95: st["duty"]=max(0.03, st["duty"]*0.6)
                elif cp < 1.2*base_p95: st["duty"]=min(1.0, st["duty"]*1.15+0.02)
    st["run"]=False; samp["on"]=False; time.sleep(0.2)
    set_cap(defW)
    return rows, list(samp["v"]), seg_marks, list(st["lat"])

# warmup baseline: uncapped critical p95
set_cap(defW); st["run"]=True; st["duty"]=0.0
wth=threading.Thread(target=crit_loop,daemon=True); wth.start(); time.sleep(4)
base_p95 = p95([l for (ts,l) in st["lat"]]); st["run"]=False; time.sleep(0.2)
print(f"CTRL_BASE crit_p95_ms={base_p95:.3f}", flush=True)

print("CTRL_TRACE_START", flush=True)
print("gpu,trajectory,rep,t,target_w,power_w,crit_p95_ms,defer_thru", flush=True)
telem=[]
for tname, segs in TRAJ.items():
    for rep in range(REPS):
        rows, sv, marks, lats = run(segs, base_p95)
        if rep==0:
            for (t,tgt,pw,cp,dt) in rows:
                print(f"{name},{tname},{rep},{t},{tgt},{pw},{cp},{dt}", flush=True)
        # per-segment telemetry from the high-rate samples
        t0abs = sv[0][0] if sv else 0
        for (sstart, tgt, dur) in marks:
            seg = [(ts-t0abs, p) for (ts,p) in sv if sstart <= ts-t0abs < sstart+dur]
            if not seg: continue
            ps = [p for (_,p) in seg]
            exceed = [max(0.0, p-tgt) for p in ps]
            frac_above = 100.0*sum(1 for e in exceed if e>2.0)/len(ps)      # >2W over target
            # actuation latency: time from seg start until power first <= target+2W (for step-downs)
            act = float("nan")
            for (rt, p) in seg:
                if p <= tgt+2.0: act = rt-sstart; break
            # steady-state (skip first 2s after a step) power stats
            steady=[p for (rt,p) in seg if rt-sstart>2.0] or ps
            seglat=[l for (ts,l) in lats if sstart <= ts-t0abs < sstart+dur]
            telem.append((name,tname,rep,round(sstart,1),tgt,round(np.mean(steady),1),
                          round(np.percentile(ps,95),1),round(max(exceed),1),round(frac_above,1),
                          round(act,2) if act==act else -1,
                          round(np.percentile(seglat,95),2) if seglat else -1,
                          round(np.percentile(seglat,99),2) if seglat else -1))
print("CTRL_TRACE_END", flush=True)
print("CTRL_TELEM_START", flush=True)
print("gpu,trajectory,rep,seg_t,target_w,mean_power_w,p95_power_w,max_exceed_w,frac_above_pct,actuation_s,crit_p95_ms,crit_p99_ms", flush=True)
for r in telem: print(",".join(str(x) for x in r), flush=True)
print("CTRL_TELEM_END", flush=True)
'''

USER_DATA_TMPL = r"""#!/bin/bash
exec > >(tee -a /dev/console) 2>&1
( sleep 1500; shutdown -h now ) &
sleep 25
echo "STRICT_BOOT_OK"
echo "{B64}" | base64 -d > /root/ctrl.py
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null && conda activate pytorch 2>/dev/null
PY=python3
for cand in "$(which python)" /opt/conda/envs/pytorch/bin/python /opt/conda/bin/python /opt/pytorch/bin/python /usr/bin/python3; do
  if [ -n "$cand" ] && $cand -c "import torch, pynvml" 2>/dev/null; then PY=$cand; break; fi
done
$PY -c "import pynvml" 2>/dev/null || $PY -m pip install -q nvidia-ml-py 2>/dev/null
echo "STRICT_PY=$PY"
$PY /root/ctrl.py 2>&1 || echo "STRICT_EXIT=$?"
echo "STRICT_DONE"
sleep 300
"""


def run_one(ec2, itype, ami, user_data):
    azs = sorted(o["Location"] for o in ec2.describe_instance_type_offerings(
        LocationType="availability-zone",
        Filters=[{"Name": "instance-type", "Values": [itype]}])["InstanceTypeOfferings"])
    if not azs:
        print(f"[aws] {itype}: not offered, skip"); return "", None
    vpc_id = subnet_id = iid = None
    try:
        vpc_id = ec2.create_vpc(CidrBlock="10.0.0.0/16",
                    TagSpecifications=[{"ResourceType": "vpc", "Tags": TAGS}])["Vpc"]["VpcId"]
        ec2.get_waiter("vpc_available").wait(VpcIds=[vpc_id])
        for i, az in enumerate(azs):
            subnet_id = ec2.create_subnet(VpcId=vpc_id, CidrBlock=f"10.0.{i}.0/24", AvailabilityZone=az,
                        TagSpecifications=[{"ResourceType": "subnet", "Tags": TAGS}])["Subnet"]["SubnetId"]
            try:
                r = ec2.run_instances(ImageId=ami, InstanceType=itype, MinCount=1, MaxCount=1,
                    SubnetId=subnet_id, UserData=user_data, InstanceInitiatedShutdownBehavior="terminate",
                    BlockDeviceMappings=[{"DeviceName": "/dev/sda1", "Ebs": {"DeleteOnTermination": True, "VolumeSize": 100}}],
                    TagSpecifications=[{"ResourceType": "instance", "Tags": TAGS}])
                iid = r["Instances"][0]["InstanceId"]; print(f"[aws] {itype} launched in {az} ({iid})"); break
            except ClientError as e:
                if "InsufficientInstanceCapacity" in str(e) and i < len(azs) - 1:
                    print(f"[aws] {itype}: no capacity in {az}, next"); ec2.delete_subnet(SubnetId=subnet_id); subnet_id=None; continue
                raise
        if iid is None:
            return "", None
        text = None; deadline = time.time() + 1200
        while time.time() < deadline:
            time.sleep(40)
            try:
                out = ec2.get_console_output(InstanceId=iid, Latest=True).get("Output", "") or ""
            except ClientError:
                out = ""
            stt = ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]["State"]["Name"]
            done = "CTRL_TELEM_END" in out or "STRICT_EXIT" in out
            print(f"  ... {itype} {int(deadline-time.time())}s left, state={stt}, {len(out)}B, done={done}")
            if out: text = out
            if done or stt in ("shutting-down", "terminated"):
                break
        return text or "", itype
    finally:
        print(f"[aws] tearing down {itype} ..."); cleanup(ec2, vpc=vpc_id, subnet=subnet_id, iid=iid)


def parse_block(text, start, end):
    if not text or start not in text or end not in text:
        return []
    block = text.split(start, 1)[1].split(end, 1)[0]
    block = re.sub(r"(?m)^\[[^\]]*\]\s*cloud-init\[\d+\]:\s*", "", block)
    seen = set(); out = []
    rdr = csv.DictReader(io.StringIO(block.strip().replace("\r", "")))
    for d in rdr:
        key = tuple(d.values())
        if key in seen or d.get("gpu") in (None, "gpu"):
            continue
        seen.add(key); out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-types", default="g5.xlarge,g6.xlarge,g4dn.xlarge")
    ap.add_argument("--region", default="us-east-1")
    args = ap.parse_args()
    ec2 = boto3.client("ec2", args.region)
    ami = latest_pytorch_dlami(ec2)
    user_data = USER_DATA_TMPL.replace("{B64}", base64.b64encode(CONTROL_PY.encode()).decode())
    types = [t.strip() for t in args.instance_types.split(",") if t.strip()]
    print(f"[aws] strict-control ami={ami} types={types}")

    os.makedirs("data/raw", exist_ok=True); os.makedirs("results", exist_ok=True)
    traces = []; telem = []
    try:
        for itype in types:
            text, _ = run_one(ec2, itype, ami, user_data)
            base = re.search(r"CTRL_BASE crit_p95_ms=([\d.]+)", text or "")
            tr = parse_block(text, "CTRL_TRACE_START", "CTRL_TRACE_END")
            te = parse_block(text, "CTRL_TELEM_START", "CTRL_TELEM_END")
            print(f"[aws] {itype}: {len(tr)} trace rows, {len(te)} telem rows"
                  + (f", base p95 {base.group(1)}ms" if base else ""))
            traces += tr; telem += te
            if not tr and not te:
                print(f"[aws] {itype} tail:\n{(text or '')[-1200:]}")
            if traces:
                with open("data/raw/aws_control_strict.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(traces[0].keys())); w.writeheader(); [w.writerow(x) for x in traces]
            if telem:
                json.dump(telem, open("results/aws_control_strict.json", "w"), indent=2)
        ng = len({t["gpu"] for t in telem}) if telem else 0
        print(f"\n[aws] TOTAL {len(traces)} trace rows, {len(telem)} telem rows across {ng} GPUs")
    finally:
        print("\n[aws] final orphan sweep ...")
        print("[aws] clean." if cleanup(ec2, tag_sweep=True) else "[aws] STRAGGLERS REMAIN")


if __name__ == "__main__":
    main()
