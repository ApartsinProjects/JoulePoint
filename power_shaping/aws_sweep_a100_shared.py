# -*- coding: utf-8 -*-
"""
A100 inference-cost sweep on ONE p4d.24xlarge (8x A100 40GB), SHARDED across the 8 GPUs.

AWS sells A100 only in 8-GPU instances, so instead of paying for 8 and using 1, we split the ~20 models
across all 8 GPUs (model j -> GPU j % NG) and run the full cap sweep on each GPU in parallel. All 8 A100s are
identical, so every (model, A100, cap) curve is valid; the 8-way parallelism just cuts wall-clock ~8x
(~4-5 min compute vs ~30), which is what makes A100 affordable (~one instance-time instead of eight).

Each per-GPU worker: pinned via CUDA_VISIBLE_DEVICES (torch sees cuda:0 = its physical GPU), caps ITS gpu via
`nvidia-smi -i <phys> -pl`, reads NVML power for <phys>, runs its shard of models, writes /root/out_<i>.csv.
The boot script waits for all workers then emits the combined CSV between SWEEP_CSV_START/END for the poller.

Output: data/raw/aws_sweep_a100.csv (own file). Same 11-column schema as aws_sweep_spectrum.csv, so
analyze_inference_cost.py reads it directly (merge with the A10G/L4/T4 spectrum for the full card set).

GATES before launching: (1) on-demand P-instance vCPU quota >= 96 (p4d = 96 vCPU) -- often 0 by default,
needs a Service Quota increase; a launch with 0 quota fails instantly (~$0) with VcpuLimitExceeded.
(2) raise the spend watchdog above $8 (p4d ~ $32.77/hr; a ~12-15 min run ~ $6-8).
Launch only after the A10G/L4/T4 sweep has finished (shared cleanup tag).
"""
from __future__ import annotations
import argparse, os, base64, csv
import boto3
from aws_sweep_expand import run_one, ensure_bucket
from aws_powercap_probe import cleanup
from aws_powercap_sweep import latest_pytorch_dlami

# ---- per-GPU worker: single-GPU sharded sweep (no header/markers; boot script adds them once) ----
WORKER_PY = r'''
import os, time, subprocess, threading
import torch, torch.nn as nn, torch.nn.functional as F

PHYS = int(os.environ.get("GPU_PHYS", "0"))       # physical index for nvidia-smi -i / NVML
SHARD = int(os.environ.get("SHARD", "0"))
NSHARDS = int(os.environ.get("NSHARDS", "1"))

def smi(q):
    return subprocess.run(["nvidia-smi", "-i", str(PHYS), f"--query-gpu={q}",
                           "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip().splitlines()

name = smi("name")[0].replace(",", " ")
defW = int(float(smi("power.default_limit")[0])); minW = int(float(smi("power.min_limit")[0]))

try:
    import pynvml; pynvml.nvmlInit(); _h = pynvml.nvmlDeviceGetHandleByIndex(PHYS)
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

dev = torch.device("cuda:0")                        # CUDA_VISIBLE_DEVICES pins this to PHYS
torch.cuda.set_device(0); torch.backends.cudnn.benchmark = True
DT = torch.float16
import torchvision.models as tvm

def _pin(t):
    try: return t.pin_memory()
    except Exception: return t

def llm_decode():
    H=4096; nH=32; nL=6; L=2048; hd=H//nH
    qkv=[nn.Linear(H,3*H,bias=False).to(dev,DT) for _ in range(nL)]
    proj=[nn.Linear(H,H,bias=False).to(dev,DT) for _ in range(nL)]
    m1=[nn.Linear(H,4*H,bias=False).to(dev,DT) for _ in range(nL)]
    m2=[nn.Linear(4*H,H,bias=False).to(dev,DT) for _ in range(nL)]
    Kc=[torch.randn(1,nH,L,hd,device=dev,dtype=DT) for _ in range(nL)]
    Vc=[torch.randn(1,nH,L,hd,device=dev,dtype=DT) for _ in range(nL)]
    cin=_pin(torch.randn(1,1,H,dtype=DT))
    def fwd(x):
        with torch.no_grad():
            for i in range(nL):
                q,k,v=qkv[i](x).chunk(3,dim=-1)
                q=q.view(1,1,nH,hd).transpose(1,2);k=k.view(1,1,nH,hd).transpose(1,2);v=v.view(1,1,nH,hd).transpose(1,2)
                K=torch.cat([Kc[i],k],2);V=torch.cat([Vc[i],v],2)
                a=F.scaled_dot_product_attention(q,K,V).transpose(1,2).reshape(1,1,H)
                x=x+proj[i](a); x=x+m2[i](F.gelu(m1[i](x)))
            return x
    return fwd,cin

class _Attn2d(nn.Module):
    def __init__(s,c):
        super().__init__();s.q=nn.Conv2d(c,c,1);s.k=nn.Conv2d(c,c,1);s.v=nn.Conv2d(c,c,1);s.o=nn.Conv2d(c,c,1)
    def forward(s,x):
        b,c,h,w=x.shape
        q=s.q(x).flatten(2).transpose(1,2);k=s.k(x).flatten(2).transpose(1,2);v=s.v(x).flatten(2).transpose(1,2)
        a=F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(b,c,h,w)
        return x+s.o(a)

def sd_unet():
    base=320; bs=4; res=64
    class UNet(nn.Module):
        def __init__(s):
            super().__init__()
            s.inc=nn.Conv2d(4,base,3,padding=1)
            s.d1=nn.Sequential(nn.Conv2d(base,base,3,padding=1),nn.SiLU(),nn.Conv2d(base,2*base,3,stride=2,padding=1),nn.SiLU())
            s.d2=nn.Sequential(nn.Conv2d(2*base,2*base,3,padding=1),nn.SiLU(),nn.Conv2d(2*base,4*base,3,stride=2,padding=1),nn.SiLU())
            s.mid=nn.Sequential(nn.Conv2d(4*base,4*base,3,padding=1),nn.SiLU(),_Attn2d(4*base),nn.Conv2d(4*base,4*base,3,padding=1),nn.SiLU())
            s.u2=nn.Sequential(nn.Upsample(scale_factor=2),nn.Conv2d(4*base,2*base,3,padding=1),nn.SiLU())
            s.u1=nn.Sequential(nn.Upsample(scale_factor=2),nn.Conv2d(2*base,base,3,padding=1),nn.SiLU())
            s.outc=nn.Conv2d(base,4,3,padding=1)
        def forward(s,x):
            h=s.inc(x);h=s.d1(h);h=s.d2(h);h=s.mid(h);h=s.u2(h);h=s.u1(h);return s.outc(h)
    m=UNet().to(dev,DT).eval(); cin=_pin(torch.randn(bs,4,res,res,dtype=DT))
    def fwd(x):
        with torch.no_grad(): return m(x)
    return fwd,cin

def vision(builder,bs=32):
    m=builder(weights=None).to(dev,DT).eval(); cin=_pin(torch.randn(bs,3,224,224,dtype=DT))
    def fwd(x):
        with torch.no_grad(): return m(x)
    return fwd,cin

_TV = ["resnet50","resnet152","resnext50_32x4d","wide_resnet50_2","vgg16","densenet121",
       "efficientnet_b0","efficientnet_v2_s","convnext_tiny","convnext_small",
       "mobilenet_v3_large","mnasnet1_0","regnet_y_1_6gf","regnet_y_8gf","shufflenet_v2_x1_0",
       "vit_b_16","vit_b_32","swin_t"]
BUILDERS={"llm_decode":llm_decode, "sd_unet":sd_unet}
for _n in _TV:
    BUILDERS[_n]=(lambda nm: (lambda: vision(getattr(tvm, nm))))(_n)

ITEMS=list(BUILDERS.items())[SHARD::NSHARDS]         # this GPU's shard of the models

NP=12
CAPS_W=sorted(set(int(round(minW+i*(defW-minW)/(NP-1))) for i in range(NP)))
REPS=2

def set_cap(w):
    subprocess.run(["sudo","nvidia-smi","-i",str(PHYS),"-pl",str(w)],capture_output=True)

def ev(): return torch.cuda.Event(enable_timing=True)
def one(fn,*a):
    torch.cuda.synchronize();e0,e1=ev(),ev();e0.record();r=fn(*a);e1.record();torch.cuda.synchronize()
    return e0.elapsed_time(e1),r

def measure_compute(fwd,x,secs=2.5):
    torch.cuda.synchronize();s=Sampler();s.start();t0=time.time();n=0
    while time.time()-t0<secs:
        fwd(x);n+=1
    torch.cuda.synchronize();el=time.time()-t0
    v=[z for z in s.stop() if z==z];p=sum(v)/len(v) if v else float("nan")
    return n/el,p

for wname,builder in ITEMS:
    try:
        t0=time.time(); fwd,cin=builder(); torch.cuda.synchronize(); t_load=(time.time()-t0)*1000
        x=cin.to(dev,non_blocking=True)
        for _ in range(3): fwd(x)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"# {wname} build/warm failed: {str(e)[:80]}", flush=True); continue
    for w in CAPS_W:
        set_cap(w); time.sleep(1.5); fr=w/defW
        for r in range(REPS):
            try:
                t_h2d,_=one(lambda: cin.to(dev,non_blocking=True))
                thr,pw=measure_compute(fwd,x)
                t_cmp=1000.0/thr if thr>0 else float("nan")
                _,out=one(fwd,x)
                t_d2h,_=one(lambda: out.detach().to("cpu"))
                print(f"{name},{wname},{fr:.3f},{w},{r},{pw:.1f},{t_load:.1f},{t_h2d:.3f},{t_cmp:.3f},{t_d2h:.3f},{thr:.5f}", flush=True)
            except Exception as e:
                print(f"# {wname},{w},{r} err {str(e)[:50]}", flush=True)
    set_cap(defW); del fwd; torch.cuda.empty_cache()
'''

USER_DATA_A100 = r"""#!/bin/bash
exec > >(tee -a /dev/console) 2>&1     # forces late output to the serial console (fallback blob path)
( sleep 1800; shutdown -h now ) &      # 30-min dead-man (caps p4d overspend at ~$16 even if it hangs)
sleep 25
echo "SWEEP_BOOT_OK"
echo "{B64}" | base64 -d > /root/worker.py
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null && conda activate pytorch 2>/dev/null
PY=python3
for cand in "$(which python)" /opt/pytorch/bin/python /opt/conda/envs/pytorch/bin/python /opt/conda/bin/python /usr/bin/python3; do
  if [ -n "$cand" ] && $cand -c "import torch, torchvision" 2>/dev/null; then PY=$cand; break; fi
done
echo "SWEEP_PYTHON=$PY"
# FAIL LOUD + CHEAP: if no torch-capable python was found, abort in ~2 min instead of a 30-min $16 burn.
$PY -c "import torch, torchvision" 2>/dev/null || { echo "SWEEP_FATAL_NO_TORCH_PY=$PY"; sleep 3; shutdown -h now; }
NG=$(nvidia-smi -L | wc -l)
echo "SWEEP_NGPU=$NG"
PIDS=""
for i in $(seq 0 $((NG-1))); do
  ( CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$i GPU_PHYS=$i SHARD=$i NSHARDS=$NG $PY /root/worker.py > /root/out_$i.csv 2>/root/err_$i.log; echo "SWEEP_W${i}_EXIT=$?" ) &
  PIDS="$PIDS $!"
done
wait $PIDS      # wait ONLY on the workers, NOT the dead-man subshell (bare `wait` would block the full 30 min)
# Combine workers, and fold each worker's stderr head in as #-comment rows so it RIDES to S3 (parser skips #).
{ echo "gpu,workload,cap_frac,cap_w,rep,power_w,t_load_ms,t_h2d_ms,t_compute_ms,t_d2h_ms,throughput"; cat /root/out_*.csv;
  for i in $(seq 0 $((NG-1))); do echo "#ERR w$i: $(head -c 300 /root/err_$i.log 2>/dev/null | tr '\n' ' ')"; done; } > /root/all.csv
echo "SWEEP_ROWS=$(wc -l < /root/all.csv)"
# PRIMARY: upload the combined results to S3 via presigned PUT (through the VPC's S3 gateway endpoint)
gzip -c /root/all.csv > /root/all.csv.gz
if [ "{PUT_URL}" != NONE ]; then
  for a in 1 2 3 4 5; do
    if curl -sSf --connect-timeout 15 --max-time 300 -X PUT --upload-file /root/all.csv.gz "{PUT_URL}"; then echo "SWEEP_UPLOADED bytes=$(stat -c%s /root/all.csv.gz)"; break; else echo "SWEEP_UPLOAD_RETRY=$a"; sleep 5; fi
  done
fi
# FALLBACK: all-worker stderr heads (BEFORE the blob so they survive the console window) + console blob
for i in $(seq 0 $((NG-1))); do echo "SWEEP_ERR_${i}:"; head -c 300 /root/err_$i.log 2>/dev/null; echo; done
echo "SWEEP_B64_START"
gzip -c /root/all.csv | base64 -w 76 | awk '{printf "B64[%05d]%s\n", NR, $0}'
echo "SWEEP_B64_END"
sleep 3; shutdown -h now
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-type", default="p4d.24xlarge", help="p4d.24xlarge=8xA100 40GB; p4de=80GB")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--no-final-cleanup", action="store_true",
                    help="skip the tag-wide orphan sweep (use when ANOTHER tagged sweep runs concurrently; "
                         "run_one still tears down this instance's own VPC/subnet/instance by id)")
    args = ap.parse_args()
    ec2 = boto3.client("ec2", region_name=args.region)
    s3c = boto3.client("s3", region_name=args.region)
    ami = latest_pytorch_dlami(ec2)
    bucket, _ = ensure_bucket(args.region)
    user_data = USER_DATA_A100.replace("{B64}", base64.b64encode(WORKER_PY.encode()).decode())
    print(f"[aws] A100 SHARDED sweep ami={ami} type={args.instance_type} bucket=s3://{bucket}")

    os.makedirs("data/raw", exist_ok=True)
    out_path = "data/raw/aws_sweep_a100.csv"
    fields = ["gpu", "workload", "cap_frac", "cap_w", "rep", "power_w",
              "t_load_ms", "t_h2d_ms", "t_compute_ms", "t_d2h_ms", "throughput"]
    try:
        rows, gpu_name = run_one(ec2, args.instance_type, ami, user_data, bucket=bucket, s3=s3c)
        for d in rows:
            d.setdefault("gpu", gpu_name)
        if rows:
            with open(out_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader(); [w.writerow(d) for d in rows]
        nwl = len({d["workload"] for d in rows})
        print(f"\n[aws] A100 TOTAL {len(rows)} rows, {nwl} workloads -> {out_path}")
    finally:
        if args.no_final_cleanup:
            print("\n[aws] skipping tag-wide orphan sweep (--no-final-cleanup); per-instance teardown already ran.")
        else:
            print("\n[aws] final orphan sweep ...")
            ok = cleanup(ec2, tag_sweep=True)
            print("[aws] all clear." if ok else "[aws] WARNING: stragglers remain -- check console!")


if __name__ == "__main__":
    main()
