# -*- coding: utf-8 -*-
"""
Local smoke test for the 4 inference workloads with the 3-way time split (model-load / data-H2D / GPU-compute
/ D2H), before spending any AWS money. Validates: each workload builds, runs a step, and the CUDA-event timing
separates the phases. Sizes are REDUCED here (SMOKE=1) so it fits a 6 GB local GPU; the AWS run uses full
sizes. The point is only to catch shape/dtype/API bugs and confirm the phase split works.

Phases per workload:
  t_load    : construct model + weights -> GPU (once)
  t_h2d     : pinned CPU input -> GPU     (cap-invariant, measure once on AWS)
  t_compute : forward pass, input resident (THE cap-sensitive term -> swept)
  t_d2h     : output -> CPU
"""
import os, time
import torch
import torch.nn as nn
import torch.nn.functional as F

SMOKE = os.environ.get("SMOKE", "1") == "1"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float16


def ev():
    return torch.cuda.Event(enable_timing=True)


# ---------- workload builders: return (forward_fn, cpu_input, to_gpu_fn) ----------
def llm_decode():
    """Batch-1 autoregressive decode step with a pre-filled KV cache -> memory-bound (real LLM-serving regime)."""
    H = 1024 if SMOKE else 4096
    nH = 8 if SMOKE else 32
    nL = 2 if SMOKE else 6
    L = 512 if SMOKE else 2048
    hd = H // nH
    qkv = [nn.Linear(H, 3 * H, bias=False).to(dev, DT) for _ in range(nL)]
    proj = [nn.Linear(H, H, bias=False).to(dev, DT) for _ in range(nL)]
    m1 = [nn.Linear(H, 4 * H, bias=False).to(dev, DT) for _ in range(nL)]
    m2 = [nn.Linear(4 * H, H, bias=False).to(dev, DT) for _ in range(nL)]
    Kc = [torch.randn(1, nH, L, hd, device=dev, dtype=DT) for _ in range(nL)]
    Vc = [torch.randn(1, nH, L, hd, device=dev, dtype=DT) for _ in range(nL)]
    cpu_in = torch.randn(1, 1, H, dtype=DT).pin_memory() if dev.type == "cuda" else torch.randn(1, 1, H, dtype=DT)

    def fwd(x):
        with torch.no_grad():
            for i in range(nL):
                q, k, v = qkv[i](x).chunk(3, dim=-1)
                q = q.view(1, 1, nH, hd).transpose(1, 2)
                k = k.view(1, 1, nH, hd).transpose(1, 2)
                v = v.view(1, 1, nH, hd).transpose(1, 2)
                K = torch.cat([Kc[i], k], dim=2)
                V = torch.cat([Vc[i], v], dim=2)
                a = F.scaled_dot_product_attention(q, K, V)
                a = a.transpose(1, 2).reshape(1, 1, H)
                x = x + proj[i](a)
                x = x + m2[i](F.gelu(m1[i](x)))
            return x
    return fwd, cpu_in


class _Attn2d(nn.Module):
    def __init__(s, c):
        super().__init__(); s.q = nn.Conv2d(c, c, 1); s.k = nn.Conv2d(c, c, 1); s.v = nn.Conv2d(c, c, 1); s.o = nn.Conv2d(c, c, 1)

    def forward(s, x):
        b, c, h, w = x.shape
        q = s.q(x).flatten(2).transpose(1, 2); k = s.k(x).flatten(2).transpose(1, 2); v = s.v(x).flatten(2).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, c, h, w)
        return x + s.o(a)


def sd_unet():
    """SD-1.5-shaped denoising UNet (conv down/up + bottleneck self-attention) -> compute-bound."""
    base = 64 if SMOKE else 320
    bs = 1 if SMOKE else 4
    res = 32 if SMOKE else 64

    class UNet(nn.Module):
        def __init__(s):
            super().__init__()
            s.inc = nn.Conv2d(4, base, 3, padding=1)
            s.d1 = nn.Sequential(nn.Conv2d(base, base, 3, padding=1), nn.SiLU(), nn.Conv2d(base, 2 * base, 3, stride=2, padding=1), nn.SiLU())
            s.d2 = nn.Sequential(nn.Conv2d(2 * base, 2 * base, 3, padding=1), nn.SiLU(), nn.Conv2d(2 * base, 4 * base, 3, stride=2, padding=1), nn.SiLU())
            s.mid = nn.Sequential(nn.Conv2d(4 * base, 4 * base, 3, padding=1), nn.SiLU(), _Attn2d(4 * base), nn.Conv2d(4 * base, 4 * base, 3, padding=1), nn.SiLU())
            s.u2 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(4 * base, 2 * base, 3, padding=1), nn.SiLU())
            s.u1 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(2 * base, base, 3, padding=1), nn.SiLU())
            s.outc = nn.Conv2d(base, 4, 3, padding=1)

        def forward(s, x):
            h = s.inc(x); h = s.d1(h); h = s.d2(h); h = s.mid(h); h = s.u2(h); h = s.u1(h); return s.outc(h)

    m = UNet().to(dev, DT).eval()
    cpu_in = torch.randn(bs, 4, res, res, dtype=DT)
    if dev.type == "cuda":
        cpu_in = cpu_in.pin_memory()

    def fwd(x):
        with torch.no_grad():
            return m(x)
    return fwd, cpu_in


def vision(builder, bs):
    import torchvision.models as tvm
    m = getattr(tvm, builder)(weights=None).to(dev, DT).eval()
    cpu_in = torch.randn(bs, 3, 224, 224, dtype=DT)
    if dev.type == "cuda":
        cpu_in = cpu_in.pin_memory()

    def fwd(x):
        with torch.no_grad():
            return m(x)
    return fwd, cpu_in


BUILDERS = {
    "llm_decode": llm_decode,
    "sd_unet": sd_unet,
    "vit_b16": lambda: vision("vit_b_16", 8 if SMOKE else 32),
    "resnet152": lambda: vision("resnet152", 8 if SMOKE else 32),
}


def timed_phase(fn, *a):
    torch.cuda.synchronize(); e0, e1 = ev(), ev(); e0.record(); r = fn(*a); e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1), r


def bench(name, builder, secs=1.5):
    t0 = time.time()
    fwd, cpu_in = builder()
    torch.cuda.synchronize(); t_load = (time.time() - t0) * 1000
    x = cpu_in.to(dev, non_blocking=True)                          # warm
    for _ in range(3):
        fwd(x)
    torch.cuda.synchronize()
    # phase timings (median of a few)
    t_h2d, _ = timed_phase(lambda: cpu_in.to(dev, non_blocking=True))
    t_cmp, out = timed_phase(fwd, x)
    t_d2h, _ = timed_phase(lambda: out.detach().to("cpu"))
    # throughput of the compute step
    n = 0; s = time.time()
    while time.time() - s < secs:
        fwd(x); n += 1
    torch.cuda.synchronize(); thr = n / (time.time() - s)
    mem = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else 0
    print(f"{name:12} load={t_load:7.1f}ms  h2d={t_h2d:6.3f}ms  compute={t_cmp:7.3f}ms  d2h={t_d2h:6.3f}ms  "
          f"thr={thr:8.1f}/s  peakmem={mem:.2f}GB")
    del fwd; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


def main():
    print(f"device={dev} SMOKE={SMOKE}")
    for name, b in BUILDERS.items():
        try:
            bench(name, b)
        except Exception as e:
            print(f"{name:12} FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
