# -*- coding: utf-8 -*-
"""
FULL-SPECTRUM cap sweep for 4 REAL inference workloads, with a 4-phase time split.

Motivation (from the cap/drain analysis):
  * We want each card's cap curve traced all the way to the DRIVER FLOOR, not stopped at 0.5xTDP. This script
    reads power.min_limit on-device and sweeps a DENSE grid from minW to defW (NP points), so the true minimum
    power limit and the full curve shape come out per card.
  * We split every measurement into  t_load / t_h2d / t_compute / t_d2h  because a power cap only stretches the
    COMPUTE term; load and transfer are cap-invariant. t_compute is what we sweep; t_load/t_h2d are recorded at
    every cap so their cap-invariance is demonstrated, not assumed.

Workloads (real inference, random weights, timed at steady state -- NO load time in the throughput):
  * llm_decode : batch-1 autoregressive decode step with a pre-filled KV cache (Llama-shaped) -> memory-bound
  * sd_unet    : SD-1.5-shaped denoising UNet (conv down/up + bottleneck self-attention)      -> compute-bound
  * vit_b16    : torchvision ViT-B/16 forward, bs=32  (SAME definition as the existing data)
  * resnet152  : torchvision ResNet-152 forward, bs=32 (SAME definition as the existing data)
vit_b16/resnet152 reuse the existing builders so their 0.5-1.0 points are directly comparable; llm_decode and
sd_unet are new, collected across the whole range.

Output: data/raw/aws_sweep_spectrum.csv  (own file -- does NOT touch aws_sweep_multi.csv).
Self-cleaning via the shared run_one/cleanup infra. Launch only AFTER any other tagged sweep has finished
(they share a cleanup tag; the final orphan sweep would otherwise terminate a concurrent run's instance).
"""
from __future__ import annotations
import argparse, os, base64, csv
import boto3
from aws_sweep_expand import run_one, USER_DATA_TMPL, ensure_bucket
from aws_powercap_probe import cleanup
from aws_powercap_sweep import latest_pytorch_dlami

SWEEP_PY = r'''
SMOKE_FLAG = False       # main() flips this to True for a tiny end-to-end validation run
import time, subprocess, threading, sys
import torch, torch.nn as nn, torch.nn.functional as F

def smi(q):
    return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip().splitlines()

subprocess.run(["nvidia-smi", "-pm", "1"], capture_output=True)
name = smi("name")[0].replace(",", " ")
defW = int(float(smi("power.default_limit")[0])); minW = int(float(smi("power.min_limit")[0]))
print(f"SWEEP_GPU={name} default={defW} min={minW}", file=sys.stderr, flush=True)  # stderr, NOT out.csv

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

# ~20 real inference models on ONE provisioning, spanning the arithmetic-intensity spectrum:
# memory-bound (depthwise/shuffle) -> mid CNN -> heavy CNN/transformer -> compute-bound matmul.
_TV = ["resnet50","resnet152","resnext50_32x4d","wide_resnet50_2","vgg16","densenet121",
       "efficientnet_b0","efficientnet_v2_s","convnext_tiny","convnext_small",
       "mobilenet_v3_large","mnasnet1_0","regnet_y_1_6gf","regnet_y_8gf","shufflenet_v2_x1_0",
       "vit_b_16","vit_b_32","swin_t"]
BUILDERS={"llm_decode":llm_decode, "sd_unet":sd_unet}
for _n in _TV:
    BUILDERS[_n]=(lambda nm: (lambda: vision(getattr(tvm, nm))))(_n)

NP=12
CAPS_W=sorted(set(int(round(minW+i*(defW-minW)/(NP-1))) for i in range(NP)))
REPS=2
ITEMS=list(BUILDERS.items())
if SMOKE_FLAG:                                       # tiny run: 2 models, 3 caps, 1 rep (~3 min)
    ITEMS=ITEMS[:2]; CAPS_W=[CAPS_W[0],CAPS_W[len(CAPS_W)//2],CAPS_W[-1]]; REPS=1

def set_cap(w):
    subprocess.run(["sudo","nvidia-smi","-pl",str(w)],capture_output=True)

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

print("gpu,workload,cap_frac,cap_w,rep,power_w,t_load_ms,t_h2d_ms,t_compute_ms,t_d2h_ms,throughput",flush=True)
for wname,builder in ITEMS:
    try:
        t0=time.time(); fwd,cin=builder(); torch.cuda.synchronize(); t_load=(time.time()-t0)*1000
        x=cin.to(dev,non_blocking=True)
        for _ in range(3): fwd(x)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"# {wname} build/warm failed: {str(e)[:80]}",flush=True); continue
    for w in CAPS_W:
        set_cap(w); time.sleep(1.5); fr=w/defW
        for r in range(REPS):
            try:
                t_h2d,_=one(lambda: cin.to(dev,non_blocking=True))
                thr,pw=measure_compute(fwd,x)
                t_cmp=1000.0/thr if thr>0 else float("nan")
                _,out=one(fwd,x)
                t_d2h,_=one(lambda: out.detach().to("cpu"))
                print(f"{name},{wname},{fr:.3f},{w},{r},{pw:.1f},{t_load:.1f},{t_h2d:.3f},{t_cmp:.3f},{t_d2h:.3f},{thr:.5f}",flush=True)
            except Exception as e:
                print(f"# {wname},{w},{r} err {str(e)[:50]}",flush=True)
    set_cap(defW); del fwd; torch.cuda.empty_cache()
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-types", default="g5.xlarge,g6.xlarge,g4dn.xlarge",
                    help="comma-separated; g5=A10G, g6=L4, g4dn=T4")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--parallel", action="store_true",
                    help="launch all instance types concurrently (each run_one is self-contained; ~3x faster)")
    ap.add_argument("--no-final-cleanup", action="store_true",
                    help="skip the tag-wide orphan sweep at the end (use when ANOTHER tagged sweep is running "
                         "concurrently, so we don't terminate its instance; run_one still tears down its own)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end validation: ONE g4dn (T4), 2 models x 3 caps x 1 rep (~3 min, ~$0.05), "
                         "writes data/raw/aws_sweep_smoke.csv. Proves the gzip-blob pipeline produces rows.")
    args = ap.parse_args()
    region = args.region
    ec2 = boto3.client("ec2", region_name=region)
    ami = latest_pytorch_dlami(ec2)
    sweep_src = SWEEP_PY.replace("SMOKE_FLAG = False", "SMOKE_FLAG = True") if args.smoke else SWEEP_PY
    user_data = USER_DATA_TMPL.replace("{B64}", base64.b64encode(sweep_src.encode()).decode())
    types = ["g4dn.xlarge"] if args.smoke else [t.strip() for t in args.instance_types.split(",") if t.strip()]
    print(f"[aws] SPECTRUM sweep ami={ami} types={types} parallel={args.parallel} smoke={args.smoke}")

    os.makedirs("data/raw", exist_ok=True)
    bucket, _ = ensure_bucket(region)                      # PRIMARY data path: per-region S3 results bucket
    print(f"[aws] results bucket: s3://{bucket}")
    out_path = "data/raw/aws_sweep_smoke.csv" if args.smoke else "data/raw/aws_sweep_spectrum.csv"
    fields = ["gpu", "workload", "cap_frac", "cap_w", "rep", "power_w",
              "t_load_ms", "t_h2d_ms", "t_compute_ms", "t_d2h_ms", "throughput"]
    all_rows = []

    def write():
        if all_rows:
            with open(out_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader(); [w.writerow(d) for d in all_rows]

    def do_one(itype):
        c = boto3.client("ec2", region_name=region)       # own client per thread (run_one tears down its own VPC)
        s3c = boto3.client("s3", region_name=region)
        rows, gpu_name = run_one(c, itype, ami, user_data, bucket=bucket, s3=s3c)
        for d in rows:
            d.setdefault("gpu", gpu_name)
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
                        print(f"[aws] {futs[fut]} failed: {type(e).__name__}: {e}")
                    write()
        else:
            for itype in types:
                try:
                    all_rows += do_one(itype)
                except Exception as e:
                    print(f"[aws] {itype} failed: {type(e).__name__}: {e}")
                write()
        nwl = len({d["workload"] for d in all_rows}); ngpu = len({d.get("gpu") for d in all_rows})
        print(f"\n[aws] SPECTRUM TOTAL {len(all_rows)} rows, {nwl} workloads, {ngpu} GPUs -> {out_path}")
    finally:
        if args.no_final_cleanup:
            print("\n[aws] skipping tag-wide orphan sweep (--no-final-cleanup); per-instance teardown already ran.")
        else:
            print("\n[aws] final orphan sweep ...")
            ok = cleanup(ec2, tag_sweep=True)
            print("[aws] all clear." if ok else "[aws] WARNING: stragglers remain -- check console!")


if __name__ == "__main__":
    main()
