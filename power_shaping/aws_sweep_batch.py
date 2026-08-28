# -*- coding: utf-8 -*-
"""Saturating-batch sweep: for each model on each GPU, AUTO-CALIBRATE the batch size to drive the GPU near
its TDP (real compute saturation), OOM-safe, then sweep power caps at that batch. This removes the low-drain
bias -- jobs now actually work the GPU, so the cap bites and elasticity shows. Memory-bound jobs that cannot
saturate (LLM decode) plateau below TDP and act as the 'stays inert' control.

Output: data/raw/aws_sweep_batch.csv (schema adds a `batch` column and `util`), via the robust S3 path.
Calibration target: power draw >= 0.90 * default_power_limit at max cap, else the largest batch that fits
(OOM) or where power plateaus.
"""
from __future__ import annotations
import argparse, os, base64, csv
import boto3
from aws_sweep_expand import run_one, USER_DATA_TMPL, ensure_bucket
from aws_powercap_probe import cleanup
from aws_powercap_sweep import latest_pytorch_dlami

SWEEP_PY = r'''
SMOKE_FLAG = False
import time, subprocess, threading, sys
import torch, torch.nn as nn, torch.nn.functional as F

def smi(q):
    return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip().splitlines()
subprocess.run(["nvidia-smi", "-pm", "1"], capture_output=True)
name = smi("name")[0].replace(",", " ")
defW = int(float(smi("power.default_limit")[0])); minW = int(float(smi("power.min_limit")[0]))
print(f"SWEEP_GPU={name} default={defW} min={minW}", file=sys.stderr, flush=True)

try:
    import pynvml; pynvml.nvmlInit(); _h = pynvml.nvmlDeviceGetHandleByIndex(0)
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
        while s.on:
            s.v.append(draw()); s.u.append(util()); time.sleep(0.05)
    def stop(s): s.on=False; return s.v, s.u

dev = torch.device("cuda:0"); torch.cuda.set_device(0); torch.backends.cudnn.benchmark = True
DT = torch.float16
import torchvision.models as tvm

def _pin(t):
    try: return t.pin_memory()
    except Exception: return t

def vision(nm):
    m = getattr(tvm, nm)(weights=None).to(dev, DT).eval()
    res = 224
    def make(bs):
        cin = _pin(torch.randn(bs, 3, res, res, dtype=DT))
        def fwd(x):
            with torch.no_grad(): return m(x)
        return fwd, cin
    return make

def sd_unet():
    base=320; res=64
    class A2(nn.Module):
        def __init__(s,c): super().__init__(); s.q=nn.Conv2d(c,c,1);s.k=nn.Conv2d(c,c,1);s.v=nn.Conv2d(c,c,1);s.o=nn.Conv2d(c,c,1)
        def forward(s,x):
            b,c,h,w=x.shape
            q=s.q(x).flatten(2).transpose(1,2);k=s.k(x).flatten(2).transpose(1,2);v=s.v(x).flatten(2).transpose(1,2)
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
    qkv=[nn.Linear(H,3*H,bias=False).to(dev,DT) for _ in range(nL)]
    proj=[nn.Linear(H,H,bias=False).to(dev,DT) for _ in range(nL)]
    m1=[nn.Linear(H,4*H,bias=False).to(dev,DT) for _ in range(nL)]
    m2=[nn.Linear(4*H,H,bias=False).to(dev,DT) for _ in range(nL)]
    def make(B):
        Kc=[torch.randn(B,nH,L,hd,device=dev,dtype=DT) for _ in range(nL)]
        Vc=[torch.randn(B,nH,L,hd,device=dev,dtype=DT) for _ in range(nL)]
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

_TV = ["resnet50","resnet152","resnext50_32x4d","wide_resnet50_2","vgg16","densenet121",
       "efficientnet_b0","efficientnet_v2_s","convnext_tiny","convnext_small",
       "mobilenet_v3_large","mnasnet1_0","regnet_y_1_6gf","regnet_y_8gf","shufflenet_v2_x1_0",
       "vit_b_16","vit_b_32","swin_t"]
MAKERS={"llm_decode":llm_decode(), "sd_unet":sd_unet()}
for _n in _TV:
    MAKERS[_n]=vision(_n)

BATCHES=[1,2,4,8,16,32,64,128,256,512]
if SMOKE_FLAG:
    MAKERS={k:MAKERS[k] for k in ["resnet50","llm_decode"]}; BATCHES=[1,4,16]

def set_cap(w):
    subprocess.run(["sudo","nvidia-smi","-pl",str(w)],capture_output=True)

def measure(fwd,x,secs=2.0):
    torch.cuda.synchronize();s=Sampler();s.start();t0=time.time();n=0
    while time.time()-t0<secs:
        fwd(x);n+=1
    torch.cuda.synchronize();el=time.time()-t0
    v,u=s.stop(); v=[z for z in v if z==z]
    p=sum(v)/len(v) if v else float("nan"); ut=sum(u)/len(u) if u else 0
    return n/el, p, ut

def calibrate(make):
    # ramp batch at max (uncapped) power; pick the batch with the HIGHEST steady draw that fits in memory.
    # No plateau early-exit: power-vs-batch is not monotone-smooth, so we ramp the whole grid (OOM-safe) and
    # take the argmax. Stop early only if we truly saturate the envelope (>=0.90*TDP).
    set_cap(defW); best_b=BATCHES[0]; best_p=-1.0; best_u=0.0
    for b in BATCHES:
        try:
            fwd,cin=make(b); x=cin.to(dev,non_blocking=True)
            for _ in range(3): fwd(x)
            torch.cuda.synchronize()
            thr,p,ut=measure(fwd,x,secs=1.5)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache(); break
            raise
        finally:
            try: del fwd,cin,x; torch.cuda.empty_cache()
            except Exception: pass
        print(f"# cal b={b} draw={p:.0f}W util={ut:.0f}% ({p/defW:.2f}xTDP)", flush=True)
        if p>best_p: best_p,best_b,best_u=p,b,ut
        if p>=0.90*defW: break                    # genuinely saturated the power envelope
    return best_b,best_p,best_u

NP=8
CAPS_W=sorted(set(int(round(minW+i*(defW-minW)/(NP-1))) for i in range(NP)))
REPS=3

def sweep_caps(fwd,x,wname,mode,bs):
    # cap sweep at a fixed (fwd,batch); emit one row per (cap,rep). mode in {control,saturate}.
    for w in CAPS_W:
        set_cap(w); time.sleep(1.5); fr=w/defW
        for r in range(REPS):
            try:
                thr,pw,ut=measure(fwd,x)
                tc=1000.0/thr if thr>0 else float("nan")
                print(f"{name},{wname},{mode},{bs},{fr:.3f},{w},{r},{pw:.1f},{ut:.0f},{tc:.3f},{thr:.5f}", flush=True)
            except Exception as e:
                print(f"# {wname},{mode},{w},{r} err {str(e)[:50]}", flush=True)

print("gpu,workload,mode,batch,cap_frac,cap_w,rep,power_w,util,t_compute_ms,throughput", flush=True)
for wname,make in MAKERS.items():
    try:
        bstar,cdraw,cutil=calibrate(make)
    except Exception as e:
        print(f"# {wname} calibrate failed: {str(e)[:80]}", flush=True); continue
    print(f"# SATURATE {wname}: batch={bstar} draw={cdraw:.0f}W ({cdraw/defW:.2f}xTDP) util={cutil:.0f}%", flush=True)
    # confound-free control vs saturate: both cap-swept back-to-back in the SAME (hot) session, same code.
    plan=[("control",32)]
    if bstar!=32: plan.append(("saturate",bstar))
    else: print(f"# {wname}: saturating batch == 32; control and saturate coincide", flush=True)
    for mode,bs in plan:
        try:
            set_cap(defW)
            fwd,cin=make(bs); x=cin.to(dev,non_blocking=True)
            for _ in range(3): fwd(x)
            torch.cuda.synchronize()
            sweep_caps(fwd,x,wname,mode,bs)
        except Exception as e:
            print(f"# {wname} {mode} b={bs} failed: {str(e)[:80]}", flush=True)
        finally:
            try: set_cap(defW); del fwd,cin,x; torch.cuda.empty_cache()
            except Exception: pass
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-types", default="g5.xlarge,g6.xlarge,g4dn.xlarge")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--no-final-cleanup", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--np", type=int, default=None, help="number of cap levels (default 8); use a larger value for a fine-grid validation sweep")
    ap.add_argument("--reps", type=int, default=None, help="reps per operating point (default 3)")
    ap.add_argument("--out", default=None, help="output csv path")
    args = ap.parse_args()
    region = args.region
    ec2 = boto3.client("ec2", region_name=region); ami = latest_pytorch_dlami(ec2)
    bucket, _ = ensure_bucket(region)
    src = SWEEP_PY.replace("SMOKE_FLAG = False", "SMOKE_FLAG = True") if args.smoke else SWEEP_PY
    if args.np:
        src = src.replace("NP=8", f"NP={args.np}")
    if args.reps:
        src = src.replace("REPS=3", f"REPS={args.reps}")
    user_data = USER_DATA_TMPL.replace("{B64}", base64.b64encode(src.encode()).decode())
    types = ["g4dn.xlarge"] if args.smoke else [t.strip() for t in args.instance_types.split(",") if t.strip()]
    print(f"[aws] BATCH-SATURATE sweep ami={ami} types={types} bucket=s3://{bucket} smoke={args.smoke} np={args.np} reps={args.reps}")
    os.makedirs("data/raw", exist_ok=True)
    out_path = args.out or ("data/raw/aws_sweep_batch_smoke.csv" if args.smoke else "data/raw/aws_sweep_batch_rep3.csv")
    fields = ["gpu", "workload", "mode", "batch", "cap_frac", "cap_w", "rep", "power_w", "util", "t_compute_ms", "throughput"]
    all_rows = []

    def write():
        if all_rows:
            with open(out_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader(); [w.writerow(d) for d in all_rows]

    def do_one(itype):
        c = boto3.client("ec2", region_name=region); s3c = boto3.client("s3", region_name=region)
        rows, gpu = run_one(c, itype, ami, user_data, bucket=bucket, s3=s3c)
        for d in rows:
            d.setdefault("gpu", gpu)
        return rows

    try:
        if args.parallel:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=len(types)) as ex:
                futs = {ex.submit(do_one, t): t for t in types}
                for fut in as_completed(futs):
                    try:
                        all_rows += fut.result()
                    except Exception as e:
                        print(f"[aws] {futs[fut]} failed: {e}")
                    write()
        else:
            for t in types:
                try:
                    all_rows += do_one(t)
                except Exception as e:
                    print(f"[aws] {t} failed: {e}")
                write()
        print(f"\n[aws] BATCH TOTAL {len(all_rows)} rows -> {out_path}")
    finally:
        if not args.no_final_cleanup:
            print("[aws] final orphan sweep ..."); cleanup(ec2, tag_sweep=True)


if __name__ == "__main__":
    main()
