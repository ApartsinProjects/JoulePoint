# -*- coding: utf-8 -*-
"""
Two-actuator hardware probe on a real GPU (self-cleaning), for the paper's actuator-portfolio
section. On one A10G it measures:
  #4 DVFS-vs-cap crossover: sweep SM clock (nvidia-smi -lgc) and, separately, power cap (-pl) on a
     fixed steady workload; report power + throughput + energy-per-op at each, so DVFS and the cap
     can be compared at matched mean power (does clock-locking beat the cap on energy/op?).
  #5 model-size ladder: measure energy-per-inference for a ladder of vision models (mobilenet ->
     resnet50 -> densenet121 -> vit_b16 -> vit_l16), the measured-energy axis of a model-cascade
     power-vs-quality frontier (quality axis is published ImageNet accuracy, cited in the paper).
Synthetic / torchvision(weights=None) workloads only -> no internet -> minimal throwaway VPC.
Outputs data/raw/aws_actuator.csv + results/aws_actuator.json.
"""
from __future__ import annotations
import argparse, time, re, os, base64, io, csv, json
import boto3
from botocore.exceptions import ClientError
from aws_powercap_probe import cleanup, TAG, TAGS
from aws_powercap_sweep import latest_pytorch_dlami

PROBE_PY = r'''
import time, subprocess, threading
import numpy as np, torch
import pynvml
pynvml.nvmlInit(); H = pynvml.nvmlDeviceGetHandleByIndex(0)
def power(): return pynvml.nvmlDeviceGetPowerUsage(H)/1000.0
name = pynvml.nvmlDeviceGetName(H); name = name.decode() if isinstance(name,bytes) else name
defW = int(pynvml.nvmlDeviceGetPowerManagementDefaultLimit(H)/1000)
lo,hi = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(H); minW=int(lo/1000)
maxclk = max(pynvml.nvmlDeviceGetSupportedGraphicsClocks(H, pynvml.nvmlDeviceGetSupportedMemoryClocks(H)[0]))
subprocess.run(["sudo","nvidia-smi","-pm","1"], capture_output=True)
print(f"PROBE_META gpu={name} default={defW} min={minW} maxclk={maxclk}", flush=True)
def set_cap(w): subprocess.run(["sudo","nvidia-smi","-pl",str(int(max(minW,min(defW,w))))], capture_output=True)
def set_clk(c): subprocess.run(["sudo","nvidia-smi","-lgc",f"{int(c)},{int(c)}"], capture_output=True)
def rst_clk(): subprocess.run(["sudo","nvidia-smi","-rgc"], capture_output=True)

dev = torch.device("cuda")
A = torch.randn(8192,8192,device=dev,dtype=torch.float16); B = torch.randn(8192,8192,device=dev,dtype=torch.float16)
def gemm(): c=A@B; torch.cuda.synchronize(); return c
def measure(step, secs=4.0):
    for _ in range(3): step()
    torch.cuda.synchronize(); v=[]; st=True
    def samp():
        while st: v.append(power()); time.sleep(0.05)
    th=threading.Thread(target=samp,daemon=True); th.start()
    t0=time.time(); n=0
    while time.time()-t0<secs: step(); n+=1
    torch.cuda.synchronize(); el=time.time()-t0; st=False; time.sleep(0.1)
    p=float(np.mean(v)) if v else float("nan")
    return n/el, p

print("PROBE_CSV_START", flush=True)
print("kind,setting,power_w,throughput,energy_per_op_j", flush=True)
# #4a clock sweep (cap at default)
set_cap(defW)
clocks = sorted({int(maxclk*f) for f in (1.0,0.85,0.7,0.55,0.4)}, reverse=True)
for c in clocks:
    set_clk(c); time.sleep(1.5)
    thr,pw = measure(gemm)
    print(f"clock,{c},{pw:.1f},{thr:.4f},{pw/max(thr,1e-6):.2f}", flush=True)
rst_clk()
# #4b cap sweep (clock unlocked)
for fr in (1.0,0.9,0.8,0.7,0.6,0.5):
    set_cap(fr*defW); time.sleep(1.5)
    thr,pw = measure(gemm)
    print(f"cap,{fr:.2f},{pw:.1f},{thr:.4f},{pw/max(thr,1e-6):.2f}", flush=True)
set_cap(defW)
# #5 model ladder (energy per inference at full power)
import torchvision.models as tvm
LADDER = [("mobilenet_v3_small",tvm.mobilenet_v3_small),("resnet50",tvm.resnet50),
          ("densenet121",tvm.densenet121),("vit_b_16",tvm.vit_b_16),("vit_l_16",tvm.vit_l_16)]
for mname, builder in LADDER:
    try:
        m = builder(weights=None).to(dev).half().eval()
        x = torch.randn(32,3,224,224,device=dev,dtype=torch.float16)
        def step():
            with torch.no_grad(): return m(x).sum()
        thr,pw = measure(step, secs=4.0)
        print(f"model,{mname},{pw:.1f},{thr:.4f},{pw/max(thr,1e-6):.2f}", flush=True)
        del m; torch.cuda.empty_cache()
    except Exception as e:
        print(f"# model {mname} failed: {str(e)[:60]}", flush=True)
print("PROBE_CSV_END", flush=True)
'''

USER_DATA_TMPL = r"""#!/bin/bash
exec > >(tee -a /dev/console) 2>&1
( sleep 1200; shutdown -h now ) &
sleep 25
echo "PROBE_BOOT_OK"
echo "{B64}" | base64 -d > /root/probe.py
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null && conda activate pytorch 2>/dev/null
PY=python3
for cand in "$(which python)" /opt/conda/envs/pytorch/bin/python /opt/conda/bin/python /opt/pytorch/bin/python /usr/bin/python3; do
  if [ -n "$cand" ] && $cand -c "import torch, torchvision, pynvml" 2>/dev/null; then PY=$cand; break; fi
done
$PY -c "import pynvml" 2>/dev/null || $PY -m pip install -q nvidia-ml-py 2>/dev/null
echo "PROBE_PY=$PY"
$PY /root/probe.py 2>&1 || echo "PROBE_EXIT=$?"
echo "PROBE_DONE"
sleep 240
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-type", default="g5.xlarge")   # A10G, wide power+clock range
    ap.add_argument("--region", default="us-east-1")
    args = ap.parse_args()
    ec2 = boto3.client("ec2", args.region)
    ami = latest_pytorch_dlami(ec2)
    user_data = USER_DATA_TMPL.replace("{B64}", base64.b64encode(PROBE_PY.encode()).decode())
    azs = sorted(o["Location"] for o in ec2.describe_instance_type_offerings(
        LocationType="availability-zone",
        Filters=[{"Name": "instance-type", "Values": [args.instance_type]}])["InstanceTypeOfferings"])
    print(f"[aws] actuator probe type={args.instance_type} ami={ami}")
    vpc_id = subnet_id = iid = None
    try:
        vpc_id = ec2.create_vpc(CidrBlock="10.0.0.0/16", TagSpecifications=[{"ResourceType": "vpc", "Tags": TAGS}])["Vpc"]["VpcId"]
        ec2.get_waiter("vpc_available").wait(VpcIds=[vpc_id])
        for i, az in enumerate(azs):
            subnet_id = ec2.create_subnet(VpcId=vpc_id, CidrBlock=f"10.0.{i}.0/24", AvailabilityZone=az,
                        TagSpecifications=[{"ResourceType": "subnet", "Tags": TAGS}])["Subnet"]["SubnetId"]
            try:
                r = ec2.run_instances(ImageId=ami, InstanceType=args.instance_type, MinCount=1, MaxCount=1,
                    SubnetId=subnet_id, UserData=user_data, InstanceInitiatedShutdownBehavior="terminate",
                    BlockDeviceMappings=[{"DeviceName": "/dev/sda1", "Ebs": {"DeleteOnTermination": True, "VolumeSize": 100}}],
                    TagSpecifications=[{"ResourceType": "instance", "Tags": TAGS}])
                iid = r["Instances"][0]["InstanceId"]; print(f"[aws] launched in {az} ({iid})"); break
            except ClientError as e:
                if "InsufficientInstanceCapacity" in str(e) and i < len(azs) - 1:
                    ec2.delete_subnet(SubnetId=subnet_id); subnet_id = None; continue
                raise
        text = None; deadline = time.time() + 1000
        while time.time() < deadline:
            time.sleep(40)
            try:
                out = ec2.get_console_output(InstanceId=iid, Latest=True).get("Output", "") or ""
            except ClientError:
                out = ""
            stt = ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]["State"]["Name"]
            done = "PROBE_CSV_END" in out or "PROBE_EXIT" in out
            print(f"  ... {int(deadline-time.time())}s left, state={stt}, {len(out)}B, done={done}")
            if out: text = out
            if done or stt in ("shutting-down", "terminated"):
                break
        rows = []
        if text and "PROBE_CSV_START" in text:
            block = text.split("PROBE_CSV_START", 1)[1].split("PROBE_CSV_END", 1)[0]
            block = re.sub(r"(?m)^\[[^\]]*\]\s*cloud-init\[\d+\]:\s*", "", block)
            seen = set()
            for d in csv.DictReader(io.StringIO(block.strip().replace("\r", ""))):
                k = (d.get("kind"), d.get("setting"))
                if d.get("kind") in ("clock", "cap", "model") and k not in seen:
                    seen.add(k); rows.append(d)
        meta = re.search(r"PROBE_META (.+)", text or "")
        os.makedirs("data/raw", exist_ok=True); os.makedirs("results", exist_ok=True)
        if rows:
            with open("data/raw/aws_actuator.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(x) for x in rows]
            json.dump({"meta": meta.group(1) if meta else "", "rows": rows},
                      open("results/aws_actuator.json", "w"), indent=2)
            print(f"[aws] captured {len(rows)} rows ({meta.group(1) if meta else ''}) -> data/raw/aws_actuator.csv")
        else:
            print(f"[aws] NO CSV. tail:\n{(text or '')[-1500:]}")
    finally:
        print("[aws] tearing down..."); print("clean" if cleanup(ec2, vpc=vpc_id, subnet=subnet_id, iid=iid) else "STRAGGLERS")


if __name__ == "__main__":
    main()
