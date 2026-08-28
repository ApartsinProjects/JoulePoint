# -*- coding: utf-8 -*-
"""Reaction-time / actuation-dynamics experiment (Stage 2/4 gap): how fast does the operating point respond
when we STEP the power cap? Runs a workload continuously on one GPU, steps the cap up/down on a schedule, and
logs power draw + recent per-iteration latency at ~50 Hz. From the trace we read the settling time (time for
power to reach the new steady value after a cap change) and any latency transient. Single g5.xlarge (A10G,
300 W TDP, wide range so steps are visible). S3 exfil via the existing scaffold.
"""
from __future__ import annotations
import argparse, os, base64, csv
import boto3
from aws_sweep_expand import run_one, USER_DATA_TMPL, ensure_bucket
from aws_powercap_probe import cleanup
from aws_powercap_sweep import latest_pytorch_dlami

SWEEP_PY = r'''
import time, subprocess, threading, sys
import torch, torch.nn as nn
import torchvision.models as tvm
dev = torch.device("cuda"); DT = torch.float16
import pynvml; pynvml.nvmlInit(); _h = pynvml.nvmlDeviceGetHandleByIndex(0)
def draw(): return pynvml.nvmlDeviceGetPowerUsage(_h) / 1000.0
def util(): return pynvml.nvmlDeviceGetUtilizationRates(_h).gpu
name = torch.cuda.get_device_name(0)
defW = int(round(pynvml.nvmlDeviceGetPowerManagementDefaultLimit(_h) / 1000.0))
minW = int(round(pynvml.nvmlDeviceGetPowerManagementLimitConstraints(_h)[0] / 1000.0))
def set_cap(w): subprocess.run(["sudo", "nvidia-smi", "-pl", str(int(w))], capture_output=True)

def llm_decode():
    H=4096; nH=32; nL=6; L=2048; hd=H//nH; B=32
    qkv=[nn.Linear(H,3*H,bias=False).to(dev,DT) for _ in range(nL)]
    proj=[nn.Linear(H,H,bias=False).to(dev,DT) for _ in range(nL)]
    m1=[nn.Linear(H,4*H,bias=False).to(dev,DT) for _ in range(nL)]
    m2=[nn.Linear(4*H,H,bias=False).to(dev,DT) for _ in range(nL)]
    Kc=[torch.randn(B,nH,L,hd,device=dev,dtype=DT) for _ in range(nL)]
    Vc=[torch.randn(B,nH,L,hd,device=dev,dtype=DT) for _ in range(nL)]
    def fwd(x):
        h=x
        for i in range(nL):
            q=qkv[i](h)[:, :H].view(B,nH,1,hd)
            a=torch.softmax((q@Kc[i].transpose(-1,-2))/8.0,dim=-1)@Vc[i]
            h=h+proj[i](a.reshape(B,H)); h=h+m2[i](torch.relu(m1[i](h)))
        return h
    return fwd, torch.randn(B,H,device=dev,dtype=DT)

def vision(nm, bs=32):
    m=getattr(tvm,nm)(weights=None).to(dev,DT).eval()
    x=torch.randn(bs,3,224,224,device=dev,dtype=DT)
    def fwd(z):
        with torch.no_grad(): return m(z)
    return fwd, x

WL = {"resnet50": lambda: vision("resnet50"), "llm_decode": llm_decode}

# schedule of (hold_seconds, cap_fraction_of_TDP); a step every 5 s, alternating deep/full to see both directions
SCHED = [(4, 1.0), (5, 0.5), (5, 1.0), (5, 0.35), (5, 1.0), (5, 0.65), (4, 1.0)]

print("gpu,workload,t_ms,cap_set_w,power_w,util,iter_ms", flush=True)
for wname, make in WL.items():
    try:
        set_cap(defW); fwd, x = make()
        for _ in range(5): fwd(x)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"# {wname} build failed: {str(e)[:80]}", flush=True); continue
    state = {"cap": defW, "last_iter_ms": float("nan"), "stop": False}
    def sampler():
        t0 = time.time()
        while not state["stop"]:
            print(f"{name},{wname},{(time.time()-t0)*1000:.0f},{state['cap']},{draw():.1f},{util()},{state['last_iter_ms']:.3f}", flush=True)
            time.sleep(0.02)                                   # ~50 Hz
    th = threading.Thread(target=sampler, daemon=True); th.start()
    for hold, frac in SCHED:
        w = max(minW, int(round(frac * defW))); set_cap(w); state["cap"] = w
        t_end = time.time() + hold
        while time.time() < t_end:
            ts = time.time(); fwd(x); torch.cuda.synchronize()
            state["last_iter_ms"] = (time.time() - ts) * 1000.0
    state["stop"] = True; time.sleep(0.1)
    set_cap(defW); del fwd, x; torch.cuda.empty_cache()
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-type", default="g5.xlarge")     # A10G, wide power range
    ap.add_argument("--region", default="us-east-1")
    args = ap.parse_args()
    region = args.region
    ec2 = boto3.client("ec2", region_name=region); ami = latest_pytorch_dlami(ec2)
    bucket, _ = ensure_bucket(region)
    user_data = USER_DATA_TMPL.replace("{B64}", base64.b64encode(SWEEP_PY.encode()).decode())
    print(f"[aws] REACTION-TIME probe ami={ami} type={args.instance_type} bucket=s3://{bucket}")
    os.makedirs("data/raw", exist_ok=True)
    out_path = "data/raw/reaction_time.csv"
    fields = ["gpu", "workload", "t_ms", "cap_set_w", "power_w", "util", "iter_ms"]
    try:
        c = boto3.client("ec2", region_name=region); s3c = boto3.client("s3", region_name=region)
        rows, gpu = run_one(c, args.instance_type, ami, user_data, bucket=bucket, s3=s3c)
        for d in rows:
            d.setdefault("gpu", gpu)
        if rows:
            with open(out_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader(); [w.writerow(d) for d in rows]
        print(f"[aws] REACTION TOTAL {len(rows)} rows -> {out_path}")
    finally:
        print("[aws] final orphan sweep ...")
        cleanup(boto3.client("ec2", region_name=region), tag_sweep=True)


if __name__ == "__main__":
    main()
