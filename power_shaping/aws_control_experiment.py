# -*- coding: utf-8 -*-
"""
MEASURED closed-loop power-shaping control on a real AWS GPU (converts the control claim
from Simulated to Measured). Reuses the self-cleaning VPC->subnet->instance lifecycle.

On a real A10G, two workloads share the GPU:
  * critical  -- a latency-sensitive small-matmul inference stream (measured per-request p95);
  * deferrable -- a throughput matmul whose DUTY CYCLE the controller modulates.
A dynamic power target steps down during a constrained window. The closed-loop controller
regulates the deferrable duty cycle so MEASURED NVML power tracks the target (shed only what
is required -- the fix for the over-curtailment critique), protecting critical latency.

Two runs are measured: UNCONTROLLED (duty=1, no control -> power violates the target) and
CONTROLLED (regulated -> power tracks target, critical latency preserved, deferrable throughput
deferred). Compact CSV returned via serial console. Nothing left behind (tag-based sweep).
"""
from __future__ import annotations
import argparse, time, json, re, os, base64, io, csv
import boto3
from botocore.exceptions import ClientError
from aws_powercap_probe import az_for_type, cleanup, TAG, TAGS
from aws_powercap_sweep import latest_pytorch_dlami

CONTROL_PY = r'''
import time, subprocess, threading, random, traceback, sys
try:
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    dev = torch.device("cuda")
except Exception:
    print("CTRL_IMPORT_FAIL", flush=True); traceback.print_exc(); sys.exit(3)
def smi(q):
    return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip().splitlines()
subprocess.run(["sudo", "nvidia-smi", "-pm", "1"], capture_output=True)
defW = int(float(smi("power.default_limit")[0]))
try:
    import pynvml; pynvml.nvmlInit(); _h = pynvml.nvmlDeviceGetHandleByIndex(0)
    def power(): return pynvml.nvmlDeviceGetPowerUsage(_h)/1000.0
except Exception:
    def power():
        try: return float(smi("power.draw")[0])
        except Exception: return float("nan")
print(f"CTRL_GPU={smi('name')[0]} default={defW}", flush=True)

# workloads
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
            el = time.time() - t0
            time.sleep(max(0.0, el * (1.0/d - 1.0)))   # busy fraction ~ duty
        else:
            time.sleep(0.02)
def crit_loop():
    while st["run"]:
        t0 = time.time(); c = crit_a @ crit_b; torch.cuda.synchronize()
        st["lat"].append((time.time(), (time.time()-t0)*1000.0)); time.sleep(0.004)

def target(t):
    return 0.55*defW if 20 <= t < 55 else 1.02*defW      # constrained window 20-55 s

def run(controlled, dur=75):
    st["run"] = True; st["duty"] = 1.0; st["lat"] = []; st["dcount"] = 0
    th = [threading.Thread(target=defer_loop, daemon=True), threading.Thread(target=crit_loop, daemon=True)]
    [x.start() for x in th]
    time.sleep(4)                                        # warm up
    t0 = time.time(); last_dc = st["dcount"]; last_t = time.time(); rows = []
    while True:
        t = time.time() - t0
        if t > dur: break
        tgt = target(t); p = power()
        if controlled:                                   # regulate duty so power tracks target
            st["duty"] = min(1.0, max(0.03, st["duty"] + 0.20 * (tgt - p) / defW))
        else:
            st["duty"] = 1.0
        now = time.time()
        recent = [l for (ts, l) in st["lat"] if ts > now - 1.0]
        p95 = sorted(recent)[int(0.95*(len(recent)-1))] if recent else float("nan")
        dthru = (st["dcount"] - last_dc) / max(1e-6, now - last_t); last_dc = st["dcount"]; last_t = now
        rows.append((("ctrl" if controlled else "base"), round(t,1), round(tgt,0),
                     round(p,1), round(p95,2), round(dthru,2)))
        time.sleep(1.0)
    st["run"] = False; time.sleep(0.3)
    return rows

try:
    print("CTRL_CSV_START", flush=True)
    print("run,t,target_w,power_w,crit_p95_ms,defer_thru", flush=True)
    for controlled in (False, True):
        for r in run(controlled):
            print(",".join(str(x) for x in r), flush=True)
        subprocess.run(["sudo","nvidia-smi","-pl",str(defW)], capture_output=True)
    print("CTRL_CSV_END", flush=True)
except Exception:
    print("CTRL_RUN_FAIL", flush=True); traceback.print_exc()
'''

USER_DATA_TMPL = r"""#!/bin/bash
exec > >(tee -a /dev/console) 2>&1
# dead-man ONLY (10 min). We do NOT self-terminate after the run so the serial console
# stays readable; the launcher's cleanup terminates the instance once it has the CSV.
( sleep 600; shutdown -h now ) &
sleep 25
echo "CTRL_BOOT_OK"
echo "{B64}" | base64 -d > /root/ctrl.py
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null && conda activate pytorch 2>/dev/null
PY=python3
for cand in "$(which python)" /opt/conda/envs/pytorch/bin/python /opt/conda/bin/python /opt/pytorch/bin/python /usr/bin/python3; do
  if [ -n "$cand" ] && $cand -c "import torch" 2>/dev/null; then PY=$cand; break; fi
done
echo "CTRL_PYTHON=$PY"
$PY /root/ctrl.py 2>&1 || echo "CTRL_PY_EXIT=$?"
echo "CTRL_DONE"
# stay up so the console remains fetchable; dead-man will reap if the launcher dies.
sleep 400
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-type", default="g5.xlarge")
    ap.add_argument("--region", default="us-east-1")
    args = ap.parse_args()
    ec2 = boto3.client("ec2", region_name=args.region)
    ami = latest_pytorch_dlami(ec2)
    azs = sorted(o["Location"] for o in ec2.describe_instance_type_offerings(
        LocationType="availability-zone",
        Filters=[{"Name": "instance-type", "Values": [args.instance_type]}])["InstanceTypeOfferings"])
    user_data = USER_DATA_TMPL.replace("{B64}", base64.b64encode(CONTROL_PY.encode()).decode())
    print(f"[aws] control experiment type={args.instance_type} ami={ami} azs={azs}")

    vpc_id = subnet_id = iid = None
    try:
        vpc_id = ec2.create_vpc(CidrBlock="10.0.0.0/16",
                    TagSpecifications=[{"ResourceType": "vpc", "Tags": TAGS}])["Vpc"]["VpcId"]
        ec2.get_waiter("vpc_available").wait(VpcIds=[vpc_id])
        for i, az in enumerate(azs):          # retry across AZs on InsufficientInstanceCapacity
            subnet_id = ec2.create_subnet(VpcId=vpc_id, CidrBlock=f"10.0.{i}.0/24", AvailabilityZone=az,
                        TagSpecifications=[{"ResourceType": "subnet", "Tags": TAGS}])["Subnet"]["SubnetId"]
            try:
                r = ec2.run_instances(ImageId=ami, InstanceType=args.instance_type, MinCount=1, MaxCount=1,
                    SubnetId=subnet_id, UserData=user_data, InstanceInitiatedShutdownBehavior="terminate",
                    BlockDeviceMappings=[{"DeviceName": "/dev/sda1", "Ebs": {"DeleteOnTermination": True, "VolumeSize": 100}}],
                    TagSpecifications=[{"ResourceType": "instance", "Tags": TAGS}])
                iid = r["Instances"][0]["InstanceId"]; print(f"[aws] launched in {az}"); break
            except ClientError as e:
                if "InsufficientInstanceCapacity" in str(e) and i < len(azs) - 1:
                    print(f"[aws] no capacity in {az}, trying next AZ")
                    ec2.delete_subnet(SubnetId=subnet_id); subnet_id = None; continue
                raise
        if iid is None:
            raise RuntimeError("no capacity in any AZ")
        print(f"[aws] instance {iid}; running measured control (poll up to ~12 min)...")
        text = None; deadline = time.time() + 720
        while time.time() < deadline:
            time.sleep(30)
            try:
                out = ec2.get_console_output(InstanceId=iid, Latest=True).get("Output", "") or ""
            except ClientError:
                out = ""
            st = ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]["State"]["Name"]
            done = any(m in out for m in ("CTRL_CSV_END", "CTRL_RUN_FAIL", "CTRL_IMPORT_FAIL"))
            print(f"  ... {int(deadline-time.time())}s left, state={st}, console={len(out)}B, "
                  f"got_csv={'CTRL_CSV_END' in out} fail={('CTRL_RUN_FAIL' in out or 'CTRL_IMPORT_FAIL' in out)}")
            if out:
                text = out                      # keep the freshest non-empty console
            if done or st in ("shutting-down", "terminated"):
                break
        rows = []
        if text and "CTRL_CSV_START" in text:
            block = text.split("CTRL_CSV_START", 1)[1].split("CTRL_CSV_END", 1)[0]
            block = re.sub(r"(?m)^\[[^\]]*\]\s*cloud-init\[\d+\]:\s*", "", block)
            for d in csv.DictReader(io.StringIO(block.strip().replace("\r", ""))):
                if d.get("run") in ("ctrl", "base"):
                    rows.append(d)
        os.makedirs("data/raw", exist_ok=True)
        if rows:
            with open("data/raw/aws_control.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["run","t","target_w","power_w","crit_p95_ms","defer_thru"])
                w.writeheader(); [w.writerow(x) for x in rows]
            print(f"\n[aws] captured {len(rows)} rows -> data/raw/aws_control.csv")
        else:
            print(f"\n[aws] NO CSV. tail:\n{(text or '')[-1500:]}")
    except ClientError as e:
        print(f"[aws] LAUNCH ERROR: {e}")
    finally:
        print("\n[aws] tearing down..."); cleanup(ec2, vpc=vpc_id, subnet=subnet_id, iid=iid)


if __name__ == "__main__":
    main()
