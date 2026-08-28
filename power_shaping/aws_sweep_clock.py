# -*- coding: utf-8 -*-
"""Clock-frequency (DVFS) sweep on the SMALL cards (T4, L4), the actuator that reaches BELOW the power-cap
floor. `nvidia-smi -pl` refuses to cap a small card below a high firmware floor (T4 ~86% TDP), so its Joule
point is unobservable with the power cap. `nvidia-smi -lgc` locks the graphics clock directly, dropping
voltage and power past that floor, exposing the sub-floor part of the U. For each of the 20 models, at its
loaded (max-draw) batch, we sweep the supported graphics clocks and record power, utilization, and throughput.
Companion to the power-cap sweeps; run on g4dn (T4) and g6 (L4). Output: data/raw/aws_sweep_clock.csv."""
import argparse, base64, csv, os
import boto3
from aws_sweep_expand import run_one, USER_DATA_TMPL, ensure_bucket
from aws_powercap_sweep import latest_pytorch_dlami

CLOCK_PY = r'''
SMOKE_FLAG = False
FOCUS_LOW = False
BATCH_MAP = {}
import time, subprocess, threading, sys
import torch, torch.nn as nn, torch.nn.functional as F

def smi(q):
    return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip().splitlines()
subprocess.run(["nvidia-smi", "-pm", "1"], capture_output=True)
name = smi("name")[0].replace(",", " ")
defW = int(float(smi("power.default_limit")[0]))
print(f"CLOCK_GPU={name} default={defW}", file=sys.stderr, flush=True)

def supported_clocks():
    out = subprocess.run(["nvidia-smi", "--query-supported-clocks=graphics", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    vals = sorted(set(int(l.strip().split()[0]) for l in out.splitlines()
                      if l.strip() and l.strip().split()[0].isdigit()))
    return vals
CLKS_ALL = supported_clocks()
NP = 8
if CLKS_ALL and FOCUS_LOW:
    # A card whose power-cap sweep already covers the high-power region only needs NEW points at low power:
    # sample densely across the bottom ~65% of the clock range, plus keep the top clock as a high anchor so
    # the clock-derived curve overlaps the power-cap-derived curve for cross-actuator validation.
    cut = CLKS_ALL[:max(3, int(round(0.65 * len(CLKS_ALL))))]
    idx = sorted(set(int(round(i * (len(cut) - 1) / (NP - 2))) for i in range(NP - 1)))
    CLKS = sorted(set([cut[i] for i in idx] + [CLKS_ALL[-1]]))
elif CLKS_ALL:
    idx = sorted(set(int(round(i * (len(CLKS_ALL) - 1) / (NP - 1))) for i in range(NP)))
    CLKS = [CLKS_ALL[i] for i in idx]
else:
    CLKS = []
print(f"SUPPORTED_CLOCKS n={len(CLKS_ALL)} span={CLKS_ALL[:1]}..{CLKS_ALL[-1:]} sweep={CLKS}", file=sys.stderr, flush=True)

try:
    import pynvml; pynvml.nvmlInit(); _h = pynvml.nvmlDeviceGetHandleByIndex(0)
    def draw(): return pynvml.nvmlDeviceGetPowerUsage(_h) / 1000.0
    def util(): return pynvml.nvmlDeviceGetUtilizationRates(_h).gpu
except Exception:
    def draw():
        try: return float(smi("power.draw")[0])
        except Exception: return float("nan")
    def util(): return 0

class Sampler(threading.Thread):
    def __init__(s): super().__init__(daemon=True); s.on = True; s.v = []; s.u = []
    def run(s):
        while s.on:
            s.v.append(draw()); s.u.append(util()); time.sleep(0.05)
    def stop(s): s.on = False; return s.v, s.u

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
    base = 320; res = 64
    class A2(nn.Module):
        def __init__(s, c): super().__init__(); s.q = nn.Conv2d(c, c, 1); s.k = nn.Conv2d(c, c, 1); s.v = nn.Conv2d(c, c, 1); s.o = nn.Conv2d(c, c, 1)
        def forward(s, x):
            b, c, h, w = x.shape
            q = s.q(x).flatten(2).transpose(1, 2); k = s.k(x).flatten(2).transpose(1, 2); v = s.v(x).flatten(2).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, c, h, w); return x + s.o(a)
    class U(nn.Module):
        def __init__(s):
            super().__init__(); s.inc = nn.Conv2d(4, base, 3, padding=1)
            s.d1 = nn.Sequential(nn.Conv2d(base, base, 3, padding=1), nn.SiLU(), nn.Conv2d(base, 2 * base, 3, stride=2, padding=1), nn.SiLU())
            s.d2 = nn.Sequential(nn.Conv2d(2 * base, 2 * base, 3, padding=1), nn.SiLU(), nn.Conv2d(2 * base, 4 * base, 3, stride=2, padding=1), nn.SiLU())
            s.mid = nn.Sequential(nn.Conv2d(4 * base, 4 * base, 3, padding=1), nn.SiLU(), A2(4 * base), nn.Conv2d(4 * base, 4 * base, 3, padding=1), nn.SiLU())
            s.u2 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(4 * base, 2 * base, 3, padding=1), nn.SiLU())
            s.u1 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(2 * base, base, 3, padding=1), nn.SiLU())
            s.outc = nn.Conv2d(base, 4, 3, padding=1)
        def forward(s, x):
            h = s.inc(x); h = s.d1(h); h = s.d2(h); h = s.mid(h); h = s.u2(h); h = s.u1(h); return s.outc(h)
    m = U().to(dev, DT).eval()
    def make(bs):
        cin = _pin(torch.randn(bs, 4, res, res, dtype=DT))
        def fwd(x):
            with torch.no_grad(): return m(x)
        return fwd, cin
    return make

def llm_decode():
    H = 4096; nH = 32; nL = 6; L = 2048; hd = H // nH
    qkv = [nn.Linear(H, 3 * H, bias=False).to(dev, DT) for _ in range(nL)]
    proj = [nn.Linear(H, H, bias=False).to(dev, DT) for _ in range(nL)]
    m1 = [nn.Linear(H, 4 * H, bias=False).to(dev, DT) for _ in range(nL)]
    m2 = [nn.Linear(4 * H, H, bias=False).to(dev, DT) for _ in range(nL)]
    def make(B):
        Kc = [torch.randn(B, nH, L, hd, device=dev, dtype=DT) for _ in range(nL)]
        Vc = [torch.randn(B, nH, L, hd, device=dev, dtype=DT) for _ in range(nL)]
        cin = _pin(torch.randn(B, 1, H, dtype=DT))
        def fwd(x):
            with torch.no_grad():
                for i in range(nL):
                    q, k, v = qkv[i](x).chunk(3, dim=-1)
                    q = q.view(B, 1, nH, hd).transpose(1, 2); k = k.view(B, 1, nH, hd).transpose(1, 2); v = v.view(B, 1, nH, hd).transpose(1, 2)
                    K = torch.cat([Kc[i], k], 2); V = torch.cat([Vc[i], v], 2)
                    a = F.scaled_dot_product_attention(q, K, V).transpose(1, 2).reshape(B, 1, H)
                    x = x + proj[i](a); x = x + m2[i](F.gelu(m1[i](x)))
                return x
        return fwd, cin
    return make

_TV = ["resnet50", "resnet152", "resnext50_32x4d", "wide_resnet50_2", "vgg16", "densenet121",
       "efficientnet_b0", "efficientnet_v2_s", "convnext_tiny", "convnext_small",
       "mobilenet_v3_large", "mnasnet1_0", "regnet_y_1_6gf", "regnet_y_8gf", "shufflenet_v2_x1_0",
       "vit_b_16", "vit_b_32", "swin_t"]
MAKERS = {"llm_decode": llm_decode(), "sd_unet": sd_unet()}
for _n in _TV:
    MAKERS[_n] = vision(_n)

BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
REPS = 2
if SMOKE_FLAG:
    MAKERS = {k: MAKERS[k] for k in ["resnet50", "convnext_tiny"]}; BATCHES = [1, 4, 16, 32]; REPS = 2

def set_clock(mhz):
    subprocess.run(["sudo", "nvidia-smi", "-lgc", f"{mhz},{mhz}"], capture_output=True)
def reset_clock():
    subprocess.run(["sudo", "nvidia-smi", "-rgc"], capture_output=True)

def measure(fwd, x, secs=2.0):
    torch.cuda.synchronize(); s = Sampler(); s.start(); t0 = time.time(); n = 0
    while time.time() - t0 < secs:
        fwd(x); n += 1
    torch.cuda.synchronize(); el = time.time() - t0
    v, u = s.stop(); v = [z for z in v if z == z]
    p = sum(v) / len(v) if v else float("nan"); ut = sum(u) / len(u) if u else 0
    return n / el, p, ut

def calibrate(make):
    # loaded batch = the one with the highest steady draw at the default (uncapped) clock.
    reset_clock(); best_b = BATCHES[0]; best_p = -1.0
    for b in BATCHES:
        try:
            fwd, cin = make(b); x = cin.to(dev, non_blocking=True)
            for _ in range(3): fwd(x)
            torch.cuda.synchronize()
            thr, p, ut = measure(fwd, x, secs=1.5)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache(); break
            raise
        finally:
            try: del fwd, cin, x; torch.cuda.empty_cache()
            except Exception: pass
        print(f"# cal b={b} draw={p:.0f}W util={ut:.0f}%", flush=True)
        if p > best_p: best_p, best_b = p, b
        if p >= 0.90 * defW: break
    return best_b, best_p

def sweep_clocks(fwd, x, wname, bs):
    for mhz in CLKS:
        set_clock(mhz); time.sleep(1.5)
        for r in range(REPS):
            try:
                thr, pw, ut = measure(fwd, x)
                tc = 1000.0 / thr if thr > 0 else float("nan")
                print(f"{name},{wname},{bs},{mhz},{r},{pw:.1f},{ut:.0f},{tc:.3f},{thr:.5f}", flush=True)
            except Exception as e:
                print(f"# {wname},{mhz},{r} err {str(e)[:50]}", flush=True)

print("gpu,workload,batch,clock_mhz,rep,power_w,util,t_compute_ms,throughput", flush=True)
if not CLKS:
    print("# NO SUPPORTED CLOCKS ENUMERATED - clock control unavailable", flush=True)
for wname, make in MAKERS.items():
    if wname in BATCH_MAP:
        # matched-batch mode: reuse the SAME batch the power-cap "saturate" sweep used for this
        # (card, workload), so clock- and cap-derived per-step energy are directly combinable.
        bstar = int(BATCH_MAP[wname]); cdraw = float("nan")
        print(f"# MATCHED {wname}: batch={bstar} (from saturate map)", flush=True)
    else:
        try:
            bstar, cdraw = calibrate(make)
        except Exception as e:
            print(f"# {wname} calibrate failed: {str(e)[:80]}", flush=True); continue
    print(f"# LOADED {wname}: batch={bstar} draw={cdraw:.0f}W", flush=True)
    try:
        reset_clock()
        fwd, cin = make(bstar); x = cin.to(dev, non_blocking=True)
        for _ in range(3): fwd(x)
        torch.cuda.synchronize()
        sweep_clocks(fwd, x, wname, bstar)
    except Exception as e:
        print(f"# {wname} sweep failed: {str(e)[:80]}", flush=True)
    finally:
        try: reset_clock(); del fwd, cin, x; torch.cuda.empty_cache()
        except Exception: pass
'''


# Per-card batch maps: the SAME (card, workload) batch the power-cap "saturate" sweep used, so the
# clock-derived and cap-derived per-step energy are directly combinable. Sourced from
# data/raw/energy_model_fit.csv rows with mode=="saturate". A10G already matched, so it is not here.
BATCH_MAPS = {
    "g4dn": {  # T4
        "convnext_small": 2, "convnext_tiny": 2, "densenet121": 8, "efficientnet_b0": 8,
        "efficientnet_v2_s": 8, "llm_decode": 1, "mnasnet1_0": 8, "mobilenet_v3_large": 16,
        "regnet_y_1_6gf": 8, "regnet_y_8gf": 2, "resnet152": 4, "resnet50": 4,
        "resnext50_32x4d": 2, "sd_unet": 1, "swin_t": 4, "vgg16": 1, "vit_b_16": 1,
        "vit_b_32": 2, "wide_resnet50_2": 1,
    },
    "g6": {  # L4
        "convnext_small": 4, "convnext_tiny": 8, "densenet121": 16, "efficientnet_v2_s": 64,
        "llm_decode": 2, "regnet_y_8gf": 4, "resnet152": 8, "resnet50": 8, "resnext50_32x4d": 8,
        "sd_unet": 2, "shufflenet_v2_x1_0": 64, "swin_t": 8, "vgg16": 1, "vit_b_16": 2,
        "vit_b_32": 8, "wide_resnet50_2": 2,
    },
}


def batch_map_for(itype):
    """Return the saturate batch map for the card behind this instance type (prefix match)."""
    for prefix, m in BATCH_MAPS.items():
        if itype.startswith(prefix):
            return m
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-types", default="g4dn.xlarge,g6.xlarge")   # T4, L4
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--focus-low", action="store_true", help="concentrate clocks in the low range (for cards whose power-cap sweep already covers the high end, e.g. A10G)")
    ap.add_argument("--no-matched-batch", action="store_true", help="disable the saturate-batch injection and fall back to per-workload calibrate")
    ap.add_argument("--out", default=None, help="output csv path (default depends on smoke)")
    args = ap.parse_args()
    region = args.region
    ec2 = boto3.client("ec2", region_name=region); ami = latest_pytorch_dlami(ec2)
    bucket, _ = ensure_bucket(region)
    types = ["g4dn.xlarge"] if args.smoke else [t.strip() for t in args.instance_types.split(",") if t.strip()]
    matched = not args.no_matched_batch
    print(f"[aws] CLOCK (DVFS) sweep ami={ami} types={types} bucket=s3://{bucket} smoke={args.smoke} focus_low={args.focus_low} matched_batch={matched}")
    os.makedirs("data/raw", exist_ok=True)
    out_path = args.out or ("data/raw/aws_sweep_clock_smoke.csv" if args.smoke else "data/raw/aws_sweep_clock.csv")
    from concurrent.futures import ThreadPoolExecutor
    header = "gpu,workload,batch,clock_mhz,rep,power_w,util,t_compute_ms,throughput".split(",")

    def build_user_data(itype):
        src = CLOCK_PY
        if args.smoke:
            src = src.replace("SMOKE_FLAG = False", "SMOKE_FLAG = True")
        if args.focus_low:
            src = src.replace("FOCUS_LOW = False", "FOCUS_LOW = True")
        if matched:
            bm = batch_map_for(itype)
            if bm:
                src = src.replace("BATCH_MAP = {}", "BATCH_MAP = " + repr(bm), 1)
                print(f"[aws] {itype}: injected saturate batch map ({len(bm)} workloads)")
            else:
                print(f"[aws] {itype}: no saturate batch map for this card -- calibrating instead")
        return USER_DATA_TMPL.replace("{B64}", base64.b64encode(src.encode()).decode())

    def do_one(itype):
        c = boto3.client("ec2", region_name=region)
        user_data = build_user_data(itype)
        rows, gpu = run_one(c, itype, ami, user_data, bucket=bucket, s3=boto3.client("s3", region_name=region))
        print(f"[aws] {itype}: captured {len(rows)} rows on {gpu}")
        return rows

    all_rows = []
    with ThreadPoolExecutor(max_workers=len(types)) as ex:
        for rows in ex.map(do_one, types):
            all_rows.extend(rows or [])
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header); w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in header})
    print(f"\n[aws] CLOCK TOTAL {len(all_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
