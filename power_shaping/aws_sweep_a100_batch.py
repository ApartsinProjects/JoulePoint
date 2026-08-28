# -*- coding: utf-8 -*-
"""A100 saturating-batch sweep, SHARDED across the 8 A100s of one p4d. Each GPU auto-calibrates the batch to
near-TDP (uncapped), then sweeps power caps at that batch, for its shard of the models. Combines the sharding
infra of aws_sweep_a100_shared.py with the batch-calibration of aws_sweep_batch.py. Output: data/raw/
aws_sweep_a100_batch.csv via S3.
"""
from __future__ import annotations
import argparse, os, base64, csv
import boto3
from aws_sweep_expand import run_one, ensure_bucket
from aws_powercap_probe import cleanup
from aws_powercap_sweep import latest_pytorch_dlami

WORKER_PY = r'''
import os, time, subprocess, threading, sys
import torch, torch.nn as nn, torch.nn.functional as F
PHYS=int(os.environ.get("GPU_PHYS","0")); SHARD=int(os.environ.get("SHARD","0")); NSHARDS=int(os.environ.get("NSHARDS","1"))
def smi(q):
    return subprocess.run(["nvidia-smi","-i",str(PHYS),f"--query-gpu={q}","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout.strip().splitlines()
name=smi("name")[0].replace(",", " ")
defW=int(float(smi("power.default_limit")[0])); minW=int(float(smi("power.min_limit")[0]))
try:
    import pynvml; pynvml.nvmlInit(); _h=pynvml.nvmlDeviceGetHandleByIndex(PHYS)
    def draw(): return pynvml.nvmlDeviceGetPowerUsage(_h)/1000.0
    def util(): return pynvml.nvmlDeviceGetUtilizationRates(_h).gpu
except Exception:
    def draw():
        try: return float(smi("power.draw")[0])
        except Exception: return float("nan")
    def util(): return 0
class Sampler(threading.Thread):
    def __init__(s): super().__init__(daemon=True); s.on=True; s.v=[]; s.u=[]
    def run(s):
        while s.on: s.v.append(draw()); s.u.append(util()); time.sleep(0.05)
    def stop(s): s.on=False; return s.v, s.u
dev=torch.device("cuda:0"); torch.cuda.set_device(0); torch.backends.cudnn.benchmark=True
DT=torch.float16; import torchvision.models as tvm
def _pin(t):
    try: return t.pin_memory()
    except Exception: return t
def vision(nm):
    m=getattr(tvm,nm)(weights=None).to(dev,DT).eval(); res=224
    def make(bs):
        cin=_pin(torch.randn(bs,3,res,res,dtype=DT))
        def fwd(x):
            with torch.no_grad(): return m(x)
        return fwd,cin
    return make
def sd_unet():
    base=320; res=64
    class A2(nn.Module):
        def __init__(s,c): super().__init__(); s.q=nn.Conv2d(c,c,1);s.k=nn.Conv2d(c,c,1);s.v=nn.Conv2d(c,c,1);s.o=nn.Conv2d(c,c,1)
        def forward(s,x):
            b,c,h,w=x.shape; q=s.q(x).flatten(2).transpose(1,2);k=s.k(x).flatten(2).transpose(1,2);v=s.v(x).flatten(2).transpose(1,2)
            a=F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(b,c,h,w); return x+s.o(a)
    class U(nn.Module):
        def __init__(s):
            super().__init__(); s.inc=nn.Conv2d(4,base,3,padding=1)
            s.d1=nn.Sequential(nn.Conv2d(base,base,3,padding=1),nn.SiLU(),nn.Conv2d(base,2*base,3,stride=2,padding=1),nn.SiLU())
            s.d2=nn.Sequential(nn.Conv2d(2*base,2*base,3,padding=1),nn.SiLU(),nn.Conv2d(2*base,4*base,3,stride=2,padding=1),nn.SiLU())
            s.mid=nn.Sequential(nn.Conv2d(4*base,4*base,3,padding=1),nn.SiLU(),A2(4*base),nn.Conv2d(4*base,4*base,3,padding=1),nn.SiLU())
            s.u2=nn.Sequential(nn.Upsample(scale_factor=2),nn.Conv2d(4*base,2*base,3,padding=1),nn.SiLU())
            s.u1=nn.Sequential(nn.Upsample(scale_factor=2),nn.Conv2d(2*base,base,3,padding=1),nn.SiLU())
            s.outc=nn.Conv2d(base,4,3,padding=1)
        def forward(s,x):
            h=s.inc(x);h=s.d1(h);h=s.d2(h);h=s.mid(h);h=s.u2(h);h=s.u1(h);return s.outc(h)
    m=U().to(dev,DT).eval()
    def make(bs):
        cin=_pin(torch.randn(bs,4,res,res,dtype=DT))
        def fwd(x):
            with torch.no_grad(): return m(x)
        return fwd,cin
    return make
def llm_decode():
    H=4096; nH=32; nL=6; L=2048; hd=H//nH
    qkv=[nn.Linear(H,3*H,bias=False).to(dev,DT) for _ in range(nL)]; proj=[nn.Linear(H,H,bias=False).to(dev,DT) for _ in range(nL)]
    m1=[nn.Linear(H,4*H,bias=False).to(dev,DT) for _ in range(nL)]; m2=[nn.Linear(4*H,H,bias=False).to(dev,DT) for _ in range(nL)]
    def make(B):
        Kc=[torch.randn(B,nH,L,hd,device=dev,dtype=DT) for _ in range(nL)]; Vc=[torch.randn(B,nH,L,hd,device=dev,dtype=DT) for _ in range(nL)]
        cin=_pin(torch.randn(B,1,H,dtype=DT))
        def fwd(x):
            with torch.no_grad():
                for i in range(nL):
                    q,k,v=qkv[i](x).chunk(3,dim=-1)
                    q=q.view(B,1,nH,hd).transpose(1,2);k=k.view(B,1,nH,hd).transpose(1,2);v=v.view(B,1,nH,hd).transpose(1,2)
                    K=torch.cat([Kc[i],k],2);V=torch.cat([Vc[i],v],2)
                    a=F.scaled_dot_product_attention(q,K,V).transpose(1,2).reshape(B,1,H)
                    x=x+proj[i](a); x=x+m2[i](F.gelu(m1[i](x)))
                return x
        return fwd,cin
    return make
_TV=["resnet50","resnet152","resnext50_32x4d","wide_resnet50_2","vgg16","densenet121","efficientnet_b0","efficientnet_v2_s","convnext_tiny","convnext_small","mobilenet_v3_large","mnasnet1_0","regnet_y_1_6gf","regnet_y_8gf","shufflenet_v2_x1_0","vit_b_16","vit_b_32","swin_t"]
MAKERS={"llm_decode":llm_decode(),"sd_unet":sd_unet()}
for _n in _TV: MAKERS[_n]=vision(_n)
ITEMS=list(MAKERS.items())[SHARD::NSHARDS]
BATCHES=[1,2,4,8,16,32,64,128,256,512]
def set_cap(w): subprocess.run(["sudo","nvidia-smi","-i",str(PHYS),"-pl",str(w)],capture_output=True)
def measure(fwd,x,secs=2.0):
    torch.cuda.synchronize();s=Sampler();s.start();t0=time.time();n=0
    while time.time()-t0<secs: fwd(x);n+=1
    torch.cuda.synchronize();el=time.time()-t0; v,u=s.stop(); v=[z for z in v if z==z]
    return n/el,(sum(v)/len(v) if v else float("nan")),(sum(u)/len(u) if u else 0)
def calibrate(make):
    # argmax draw over the whole batch grid (OOM-safe); no plateau early-exit. Stop only on real saturation.
    set_cap(defW); best_b=BATCHES[0]; best_p=-1.0; best_u=0.0
    for b in BATCHES:
        try:
            fwd,cin=make(b); x=cin.to(dev,non_blocking=True)
            for _ in range(3): fwd(x)
            torch.cuda.synchronize(); thr,p,ut=measure(fwd,x,secs=1.5)
        except RuntimeError as e:
            if "out of memory" in str(e).lower(): torch.cuda.empty_cache(); break
            raise
        finally:
            try: del fwd,cin,x; torch.cuda.empty_cache()
            except Exception: pass
        if p>best_p: best_p,best_b,best_u=p,b,ut
        if p>=0.90*defW: break
    return best_b,best_p,best_u
NP=8; CAPS_W=sorted(set(int(round(minW+i*(defW-minW)/(NP-1))) for i in range(NP))); REPS=3
def sweep_caps(fwd,x,wname,mode,bs):
    for w in CAPS_W:
        set_cap(w); time.sleep(1.5); fr=w/defW
        for r in range(REPS):
            try:
                thr,pw,ut=measure(fwd,x); tc=1000.0/thr if thr>0 else float("nan")
                print(f"{name},{wname},{mode},{bs},{fr:.3f},{w},{r},{pw:.1f},{ut:.0f},{tc:.3f},{thr:.5f}",flush=True)
            except Exception as e: sys.stderr.write(f"{wname},{mode},{w} err {str(e)[:50]}\n")
for wname,make in ITEMS:
    try:
        bstar,cdraw,cutil=calibrate(make)
    except Exception as e:
        sys.stderr.write(f"{wname} cal failed: {str(e)[:80]}\n"); continue
    sys.stderr.write(f"SATURATE {wname}: batch={bstar} draw={cdraw:.0f}W ({cdraw/defW:.2f}xTDP)\n")
    plan=[("control",32)]
    if bstar!=32: plan.append(("saturate",bstar))
    for mode,bs in plan:
        try:
            set_cap(defW); fwd,cin=make(bs); x=cin.to(dev,non_blocking=True)
            for _ in range(3): fwd(x)
            torch.cuda.synchronize(); sweep_caps(fwd,x,wname,mode,bs)
        except Exception as e:
            sys.stderr.write(f"{wname} {mode} b={bs} failed: {str(e)[:80]}\n")
        finally:
            try: set_cap(defW); del fwd,cin,x; torch.cuda.empty_cache()
            except Exception: pass
'''

USER_DATA = r"""#!/bin/bash
exec > >(tee -a /dev/console) 2>&1
( sleep 2400; shutdown -h now ) &      # 40-min dead-man (calibration is heavier)
sleep 25
echo "SWEEP_BOOT_OK"
echo "{B64}" | base64 -d > /root/worker.py
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null && conda activate pytorch 2>/dev/null
PY=python3
for cand in "$(which python)" /opt/pytorch/bin/python /opt/conda/envs/pytorch/bin/python /opt/conda/bin/python /usr/bin/python3; do
  if [ -n "$cand" ] && $cand -c "import torch, torchvision" 2>/dev/null; then PY=$cand; break; fi
done
echo "SWEEP_PYTHON=$PY"
$PY -c "import torch, torchvision" 2>/dev/null || { echo "SWEEP_FATAL_NO_TORCH_PY=$PY"; sleep 3; shutdown -h now; }
NG=$(nvidia-smi -L | wc -l); echo "SWEEP_NGPU=$NG"
PIDS=""
for i in $(seq 0 $((NG-1))); do
  ( CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$i GPU_PHYS=$i SHARD=$i NSHARDS=$NG $PY /root/worker.py > /root/out_$i.csv 2>/root/err_$i.log; echo "SWEEP_W${i}_EXIT=$?" ) &
  PIDS="$PIDS $!"
done
wait $PIDS
{ echo "gpu,workload,mode,batch,cap_frac,cap_w,rep,power_w,util,t_compute_ms,throughput"; cat /root/out_*.csv;
  for i in $(seq 0 $((NG-1))); do echo "#ERR w$i: $(head -c 300 /root/err_$i.log 2>/dev/null | tr '\n' ' ')"; done; } > /root/all.csv
echo "SWEEP_ROWS=$(wc -l < /root/all.csv)"
gzip -c /root/all.csv > /root/all.csv.gz
if [ "{PUT_URL}" != NONE ]; then
  for a in 1 2 3 4 5; do
    if curl -sSf --connect-timeout 15 --max-time 300 -X PUT --upload-file /root/all.csv.gz "{PUT_URL}"; then echo "SWEEP_UPLOADED"; break; else echo "SWEEP_UPLOAD_RETRY=$a"; sleep 5; fi
  done
fi
for i in $(seq 0 $((NG-1))); do echo "SWEEP_ERR_${i}:"; head -c 300 /root/err_$i.log 2>/dev/null; echo; done
echo "SWEEP_B64_START"; gzip -c /root/all.csv | base64 -w 76 | awk '{printf "B64[%05d]%s\n", NR, $0}'; echo "SWEEP_B64_END"
sleep 3; shutdown -h now
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-type", default="p4d.24xlarge")
    ap.add_argument("--region", default="ap-northeast-1")
    args = ap.parse_args()
    ec2 = boto3.client("ec2", region_name=args.region); s3c = boto3.client("s3", region_name=args.region)
    ami = latest_pytorch_dlami(ec2); bucket, _ = ensure_bucket(args.region)
    user_data = USER_DATA.replace("{B64}", base64.b64encode(WORKER_PY.encode()).decode())
    print(f"[aws] A100 BATCH-SATURATE sharded ami={ami} type={args.instance_type} bucket=s3://{bucket}")
    os.makedirs("data/raw", exist_ok=True)
    out_path = "data/raw/aws_sweep_a100_batch_rep3.csv"
    fields = ["gpu", "workload", "mode", "batch", "cap_frac", "cap_w", "rep", "power_w", "util", "t_compute_ms", "throughput"]
    try:
        rows, gpu = run_one(ec2, args.instance_type, ami, user_data, bucket=bucket, s3=s3c)
        for d in rows:
            d.setdefault("gpu", gpu)
        if rows:
            with open(out_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); [w.writerow(d) for d in rows]
        print(f"\n[aws] A100 BATCH TOTAL {len(rows)} rows -> {out_path}")
    finally:
        print("[aws] final orphan sweep ..."); cleanup(ec2, tag_sweep=True)


if __name__ == "__main__":
    main()
