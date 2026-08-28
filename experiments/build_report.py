# -*- coding: utf-8 -*-
"""Generate the pilot technical report with the measured data embedded."""
import io, json, math
from collections import defaultdict
import numpy as np

d = json.load(io.open("experiments/pilot_results.json", encoding="utf-8"))
rows = [r for b in d for r in b["rows"] if r.get("status") == "ok"]
MACH = ["T4", "L4", "A10G", "L40S", "A100-40GB"]
LOADS = [("resnet50", "ResNet-50"), ("vit_b16", "ViT-B/16"),
         ("convnext_t", "ConvNeXt-T"), ("transformer", "Transformer-6L")]
caps = {b["machine"]: b.get("power_cap_w") for b in d}
devs = {b["machine"]: b.get("device") for b in d}

E, T, P, MEM = defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict)
for r in rows:
    k = (r["load"], r["precision"], r["batch"])
    E[k][r["machine"]] = r["energy_per_sample_mj"]
    T[k][r["machine"]] = r["throughput_sps"]
    P[k][r["machine"]] = r["peak_power_w"]
    MEM[k][r["machine"]] = r["peak_mem_gb"]

# --- energy matrix tables, one per configuration ---
def matrix_table(prec, bs):
    h = f'<table><caption>Energy per sample (mJ), {prec}, batch {bs}</caption><tr><th>Load</th>'
    h += "".join(f"<th>{m}</th>" for m in MACH) + "<th>best</th><th>spread</th></tr>"
    for key, disp in LOADS:
        k = (key, prec, bs)
        v = E[k]
        best = min(v, key=v.get)
        h += f"<tr><td>{disp}</td>"
        for m in MACH:
            cls = ' class="best"' if m == best else ""
            h += f"<td{cls}>{v[m]:.1f}</td>"
        h += f"<td>{best}</td><td>{max(v.values())/min(v.values()):.2f}x</td></tr>"
    return h + "</table>"

# --- performance-first penalty ---
pen_rows = ""
pens = []
for key, disp in LOADS:
    for prec in ("fp16", "fp32"):
        for bs in (8, 32, 128):
            k = (key, prec, bs)
            fast = max(T[k], key=T[k].get); green = min(E[k], key=E[k].get)
            pen = 100 * (E[k][fast] - E[k][green]) / E[k][green]
            pens.append(pen)
            pen_rows += (f"<tr><td>{disp}</td><td>{prec}</td><td>{bs}</td><td>{fast}</td>"
                         f"<td>{green}</td><td>{pen:.1f}%</td></tr>")
pens_sorted = sorted(pens)
pen_med = pens_sorted[len(pens_sorted)//2]
pen_max = pens_sorted[-1]

# --- reversals ---
rev = []
for key, disp in LOADS:
    cfgs = [k for k in E if k[0] == key]
    for i, a in enumerate(MACH):
        for b in MACH[i+1:]:
            signs = {(E[c][a] < E[c][b]) for c in cfgs}
            if len(signs) > 1:
                byprec = {p: {E[c][a] < E[c][b] for c in cfgs if c[1] == p} for p in ("fp16", "fp32")}
                drv = ("precision" if all(len(v) == 1 for v in byprec.values())
                       and byprec["fp16"] != byprec["fp32"] else "batch size")
                rev.append((disp, a, b, drv))
rev_rows = "".join(f"<tr><td>{l}</td><td>{a}</td><td>{b}</td><td>{dr}</td></tr>" for l, a, b, dr in rev)
n_prec = sum(1 for r in rev if r[3] == "precision")

# --- SVD / additive ---
keys = sorted(E)
L = np.array([[math.log(E[k][m]) for m in MACH] for k in keys])
A = L - L.mean()
S = np.linalg.svd(A, compute_uv=False)
tot = (S**2).sum()
ev = [100*(S[:r+1]**2).sum()/tot for r in range(len(S))]
rm, cm, gm = L.mean(axis=1, keepdims=True), L.mean(axis=0, keepdims=True), L.mean()
resid = L - (rm + cm - gm)
ss_tot = ((L-gm)**2).sum(); ss_res = (resid**2).sum()
add_pct = 100*(1-ss_res/ss_tot)
max_res = float(np.abs(resid).max())

# --- power ---
pw_rows = ""
for m in MACH:
    obs = [P[k][m] for k in P]
    pw_rows += (f"<tr><td>{m}</td><td>{devs[m]}</td><td>{caps[m]:.0f}</td>"
                f"<td>{min(obs):.1f}</td><td>{max(obs):.1f}</td><td>{max(obs)/min(obs):.2f}x</td></tr>")

# --- full appendix table ---
full = ""
for key, disp in LOADS:
    for prec in ("fp16", "fp32"):
        for bs in (8, 32, 128):
            k = (key, prec, bs)
            for m in MACH:
                r = next(x for x in rows if x["load"] == key and x["precision"] == prec
                         and x["batch"] == bs and x["machine"] == m)
                full += (f"<tr><td>{disp}</td><td>{prec}</td><td>{bs}</td><td>{m}</td>"
                         f"<td>{r['energy_per_sample_mj']:.2f}</td><td>{r['throughput_sps']:.1f}</td>"
                         f"<td>{r['peak_power_w']:.0f}</td><td>{r['mean_power_w']:.0f}</td>"
                         f"<td>{r['peak_mem_gb']:.2f}</td><td>{r['iters']}</td></tr>")

sp = sorted(max(v.values())/min(v.values()) for v in E.values())

vals = dict(
    n_cells=len(rows), n_cfg=len(E),
    sp_min=f"{sp[0]:.2f}", sp_med=f"{sp[len(sp)//2]:.2f}", sp_max=f"{sp[-1]:.2f}",
    pen_med=f"{pen_med:.1f}", pen_max=f"{pen_max:.1f}",
    n_rev=len(rev), n_pairs=len(LOADS)*len(MACH)*(len(MACH)-1)//2,
    pct_rev=f"{100*len(rev)/(len(LOADS)*len(MACH)*(len(MACH)-1)//2):.0f}",
    n_prec=n_prec, n_bs=len(rev)-n_prec,
    r1=f"{ev[0]:.1f}", r2=f"{ev[1]:.1f}", r3=f"{ev[2]:.1f}",
    add_pct=f"{add_pct:.1f}", res_pct=f"{100-add_pct:.1f}",
    max_res_x=f"{math.exp(max_res):.2f}",
    m_fp16_32=matrix_table("fp16", 32), m_fp32_32=matrix_table("fp32", 32),
    m_fp16_8=matrix_table("fp16", 8), m_fp32_128=matrix_table("fp32", 128),
    pen_rows=pen_rows, rev_rows=rev_rows, pw_rows=pw_rows, full=full,
)
json.dump(vals, io.open("experiments/report_values.json", "w", encoding="utf-8"), indent=1)
print("computed:", {k: v for k, v in vals.items() if not isinstance(v, str) or len(v) < 40})
