# -*- coding: utf-8 -*-
"""
AWS EC2 GPU power-cap ELASTICITY SWEEP with full self-cleanup (reuses the probe lifecycle).

Runs a set of heterogeneous workloads (compute-bound vs memory-bound) on a real g5.xlarge
(A10G) at power caps {100,90,80,70,60}% of TDP, logging measured NVML power + throughput at
each cap. Produces the real GPU-inference power-cap elasticity data the public datasets lack,
in the same schema the kill-test/learned-model loaders read.

No model downloads (torchvision weights=None, synthetic kernels) -> no internet -> the same
minimal throwaway VPC+subnet as the probe; the instance self-terminates and everything is
torn down (instance -> subnet -> VPC) with a tag-based orphan sweep.

Retrieval: the on-instance script prints a compact CSV between markers to the serial console;
we fetch it with get_console_output (CSV is a few KB, well under the 64 KB cap).
"""
from __future__ import annotations
import argparse, time, json, re, os, base64, io, csv
import boto3
from botocore.exceptions import ClientError
from aws_powercap_probe import az_for_type, cleanup, TAG, TAGS


def latest_pytorch_dlami(ec2):
    """Latest AWS Deep Learning AMI that ships PyTorch + torchvision preinstalled."""
    for pat in ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Ubuntu 22.04)*",
                "Deep Learning AMI GPU PyTorch * (Ubuntu 20.04)*"]:
        imgs = ec2.describe_images(Owners=["amazon"], Filters=[
            {"Name": "name", "Values": [pat]}, {"Name": "state", "Values": ["available"]}])["Images"]
        if imgs:
            return sorted(imgs, key=lambda x: x["CreationDate"], reverse=True)[0]["ImageId"]
    raise RuntimeError("no PyTorch Deep Learning AMI found")

# ---------------- on-instance sweep program ----------------
SWEEP_PY = r'''
import time, subprocess, threading, sys
import torch

def smi(q):
    return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip().splitlines()

subprocess.run(["nvidia-smi", "-pm", "1"], capture_output=True)
name = smi("name")[0]
defW = int(float(smi("power.default_limit")[0])); minW = int(float(smi("power.min_limit")[0]))
print(f"SWEEP_GPU={name} default={defW} min={minW}", flush=True)

# power sampler (pynvml if available, else nvidia-smi)
try:
    import pynvml; pynvml.nvmlInit(); _h = pynvml.nvmlDeviceGetHandleByIndex(0)
    def draw(): return pynvml.nvmlDeviceGetPowerUsage(_h)/1000.0
except Exception:
    def draw():
        try: return float(smi("power.draw")[0])
        except Exception: return float("nan")

class Sampler(threading.Thread):
    def __init__(s): super().__init__(daemon=True); s.on=True; s.v=[]
    def run(s):
        while s.on:
            s.v.append(draw()); time.sleep(0.05)
    def stop(s): s.on=False; return s.v

dev = torch.device("cuda"); torch.backends.cudnn.benchmark = True

# ---- workloads: each returns a step() doing one unit of work ----
def wl_gemm(dtype):
    a = torch.randn(8192, 8192, device=dev, dtype=dtype); b = torch.randn(8192, 8192, device=dev, dtype=dtype)
    def step():
        c = a @ b; return c.sum()
    return step
def wl_membw():
    n = 1<<26  # 64M elems fp32 = 256MB
    x = torch.randn(n, device=dev); y = torch.randn(n, device=dev)
    def step():
        z = x*1.0001 + y; return z[0]
    return step
def wl_embed():
    emb = torch.nn.Embedding(4_000_000, 128).to(dev)
    idx = torch.randint(0, 4_000_000, (65536,), device=dev)
    def step():
        return emb(idx).sum()
    return step
def wl_vision(builder):
    import torchvision.models as tvm
    m = builder(weights=None).to(dev).half().eval()
    x = torch.randn(32, 3, 224, 224, device=dev, dtype=torch.float16)
    def step():
        with torch.no_grad(): return m(x).sum()
    return step
def wl_decode():
    # small sequential matmuls with per-step sync (latency-bound, decode-like)
    a = torch.randn(512, 4096, device=dev, dtype=torch.float16); b = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    def step():
        c = a @ b; torch.cuda.synchronize(); return c.sum()
    return step

import torchvision.models as tvm
WORKLOADS = {
    "gemm_fp16":  wl_gemm(torch.float16),
    "gemm_fp32":  wl_gemm(torch.float32),
    "membw":      wl_membw(),
    "embed_gather": wl_embed(),
    "resnet50":   wl_vision(tvm.resnet50),
    "vit_b16":    wl_vision(tvm.vit_b_16),
    "decode_like": wl_decode(),
}

CAPS = [1.0, 0.9, 0.8, 0.7, 0.6]
REPS = 3
def set_cap(fr):
    w = max(minW, int(round(fr*defW)))
    subprocess.run(["sudo","nvidia-smi","-pl",str(w)], capture_output=True); return w

def measure(step, secs=4.0):
    torch.cuda.synchronize(); s = Sampler(); s.start()
    t0 = time.time(); n = 0
    while time.time()-t0 < secs:
        step(); n += 1
    torch.cuda.synchronize(); el = time.time()-t0
    v = s.stop(); v = [x for x in v if x==x]
    p = sum(v)/len(v) if v else float("nan")
    return n/el, p

print("SWEEP_CSV_START", flush=True)
print("workload,cap_frac,cap_w,rep,power_w,throughput", flush=True)
for wname, step in WORKLOADS.items():
    try:
        for _ in range(2): step()          # warm
        torch.cuda.synchronize()
    except Exception as e:
        print(f"# {wname} build/warm failed: {str(e)[:80]}", flush=True); continue
    for fr in CAPS:
        w = set_cap(fr); time.sleep(1.5)   # settle the cap
        for r in range(REPS):
            try:
                thr, pw = measure(step)
                print(f"{wname},{fr:.2f},{w},{r},{pw:.1f},{thr:.4f}", flush=True)
            except Exception as e:
                print(f"# {wname},{fr},{w},{r} err {str(e)[:60]}", flush=True)
    set_cap(1.0)
print("SWEEP_CSV_END", flush=True)
'''

USER_DATA_TMPL = r"""#!/bin/bash
exec > >(tee -a /dev/console) 2>&1
( sleep 2400; shutdown -h now ) &      # 40-min dead-man switch
sleep 25
echo "SWEEP_BOOT_OK"
echo "{B64}" | base64 -d > /root/sweep.py
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null && conda activate pytorch 2>/dev/null
PY=python3
for cand in "$(which python)" /opt/conda/envs/pytorch/bin/python /opt/conda/bin/python /opt/pytorch/bin/python /usr/bin/python3; do
  if [ -n "$cand" ] && $cand -c "import torch, torchvision" 2>/dev/null; then PY=$cand; break; fi
done
echo "SWEEP_PYTHON=$PY"
$PY /root/sweep.py || echo "SWEEP_PY_EXIT=$?"
echo "SWEEP_DONE"
sleep 3; shutdown -h now
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-type", default="g5.xlarge")
    ap.add_argument("--region", default="us-east-1")
    args = ap.parse_args()
    ec2 = boto3.client("ec2", region_name=args.region)
    ami = latest_pytorch_dlami(ec2); az = az_for_type(ec2, args.instance_type)
    b64 = base64.b64encode(SWEEP_PY.encode()).decode()
    user_data = USER_DATA_TMPL.replace("{B64}", b64)
    print(f"[aws] sweep region={args.region} type={args.instance_type} az={az} ami={ami}")

    vpc_id = subnet_id = iid = None
    try:
        vpc_id = ec2.create_vpc(CidrBlock="10.0.0.0/16",
                    TagSpecifications=[{"ResourceType": "vpc", "Tags": TAGS}])["Vpc"]["VpcId"]
        ec2.get_waiter("vpc_available").wait(VpcIds=[vpc_id])
        subnet_id = ec2.create_subnet(VpcId=vpc_id, CidrBlock="10.0.0.0/24", AvailabilityZone=az,
                    TagSpecifications=[{"ResourceType": "subnet", "Tags": TAGS}])["Subnet"]["SubnetId"]
        print(f"[aws] vpc={vpc_id} subnet={subnet_id}")
        r = ec2.run_instances(ImageId=ami, InstanceType=args.instance_type, MinCount=1, MaxCount=1,
            SubnetId=subnet_id, UserData=user_data, InstanceInitiatedShutdownBehavior="terminate",
            BlockDeviceMappings=[{"DeviceName": "/dev/sda1", "Ebs": {"DeleteOnTermination": True, "VolumeSize": 100}}],
            TagSpecifications=[{"ResourceType": "instance", "Tags": TAGS}])
        iid = r["Instances"][0]["InstanceId"]
        print(f"[aws] instance {iid}; sweeping (poll up to ~35 min)...")

        text = None; deadline = time.time() + 2100
        while time.time() < deadline:
            time.sleep(45)
            try:
                out = ec2.get_console_output(InstanceId=iid, Latest=True).get("Output", "") or ""
            except ClientError:
                out = ""
            st = ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]["State"]["Name"]
            got = "SWEEP_CSV_END" in out
            print(f"  ... {int(deadline-time.time())}s left, state={st}, console={len(out)}B, csv_end={got}")
            if got or st == "terminated":
                text = out; break

        rows = []
        if text and "SWEEP_CSV_START" in text:
            block = text.split("SWEEP_CSV_START", 1)[1].split("SWEEP_CSV_END", 1)[0]
            # strip the serial-console line prefix "[   12.34] cloud-init[NNN]: " from each line
            block = re.sub(r"(?m)^\[[^\]]*\]\s*cloud-init\[\d+\]:\s*", "", block)
            rdr = csv.DictReader(io.StringIO(block.strip().replace("\r", "")))
            for d in rdr:
                if d.get("workload") and not d["workload"].startswith("#"):
                    rows.append(d)
        os.makedirs("data/raw", exist_ok=True)
        gpu = re.search(r"SWEEP_GPU=(.*)", text or "")
        if rows:
            with open("data/raw/own_aws_sweep.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["workload","cap_frac","cap_w","rep","power_w","throughput"])
                w.writeheader(); [w.writerow(x) for x in rows]
            print(f"\n[aws] captured {len(rows)} rows on {gpu.group(1) if gpu else '?'} "
                  f"-> data/raw/own_aws_sweep.csv")
        else:
            print(f"\n[aws] NO CSV captured. console tail:\n{(text or '')[-1500:]}")
    except ClientError as e:
        print(f"[aws] LAUNCH ERROR: {e}")
    finally:
        print("\n[aws] tearing down all created resources...")
        cleanup(ec2, vpc=vpc_id, subnet=subnet_id, iid=iid)


if __name__ == "__main__":
    main()
