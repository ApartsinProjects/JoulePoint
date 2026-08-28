"""
Experiment 1 (execution-idle): power when a GPU is ALLOCATED but barely working.
  Literature reports execution-idle at 10.7% of cluster energy, 48% for serving.
  Nobody has characterised which workload-hardware pairs strand the most, and our
  own grid never measured it because every cell runs flat out.
  States measured: deep idle -> CUDA context -> model resident -> duty cycle sweep.

Experiment 3 (N4, break the dominance confound): loads large enough that the L4's
  24 GB binds, so memory infeasibility appears and no single machine wins every cell.
  Until this exists, placement REGRET is untestable on our grid.
"""
import json
import modal

image = (modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel", add_python=None)
         .pip_install("nvidia-ml-py==12.560.30", "numpy"))
app = modal.App("greenmatch-execidle-n4", image=image)

DUTY = [0.0, 0.05, 0.10, 0.25, 0.50, 1.00]
BIG = [
    # (name, builder key, batch sizes) chosen to straddle 16/24/40/48 GB
    ("vit_l16", [16, 64, 128]),
    ("resnet152", [64, 256, 512]),
    ("xformer_big", [8, 32, 64]),
]


def measure(machine):
    import time, gc, torch, pynvml
    import torchvision.models as tvm
    import statistics as st

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    dev = pynvml.nvmlDeviceGetName(h)
    dev = dev.decode() if isinstance(dev, bytes) else dev
    cap = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
    total_mem = pynvml.nvmlDeviceGetMemoryInfo(h).total / 1e9

    def watts(sec=8.0):
        s = []
        t0 = time.time()
        while time.time() - t0 < sec:
            s.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
            time.sleep(0.2)
        return float(st.median(s)), float(max(s))

    def energy_mj():
        try:
            return pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        except Exception:
            return None

    out = {"machine": machine, "device": dev, "power_cap_w": cap, "mem_total_gb": total_mem}

    # ---------- experiment 1: execution-idle ladder ----------
    states = {}
    states["deep_idle"] = watts(10)                      # no CUDA context yet
    torch.zeros(1, device="cuda"); torch.cuda.synchronize()
    states["cuda_context"] = watts(10)                   # context created, nothing resident
    model = tvm.resnet50(weights=None).cuda().eval()
    x = torch.randn(32, 3, 224, 224, device="cuda")
    torch.cuda.synchronize()
    states["model_resident"] = watts(10)                 # weights + activations allocated, no compute

    duty_rows = []
    for d in DUTY:
        e0 = energy_mj(); t0 = time.perf_counter(); samples = []; peak = 0.0
        period = 0.5
        end = t0 + 12.0
        n_iter = 0
        while time.perf_counter() < end:
            cyc = time.perf_counter()
            if d > 0:
                busy_until = cyc + period * d
                with torch.no_grad():
                    while time.perf_counter() < busy_until:
                        model(x); n_iter += 1
                torch.cuda.synchronize()
            rest = period * (1 - d) - max(0.0, time.perf_counter() - cyc - period * d)
            if rest > 0:
                t1 = time.perf_counter()
                while time.perf_counter() - t1 < rest:
                    w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                    samples.append(w); peak = max(peak, w)
                    time.sleep(0.05)
        el = time.perf_counter() - t0
        e1 = energy_mj()
        j = (e1 - e0) / 1000.0 if (e0 and e1) else None
        duty_rows.append({"duty": d, "elapsed_s": round(el, 2), "iters": n_iter,
                          "energy_j": round(j, 2) if j else None,
                          "avg_power_w": round(j / el, 1) if j else None,
                          "idle_sample_median_w": round(st.median(samples), 1) if samples else None})
    out["exec_idle"] = {"states": {k: {"median_w": v[0], "max_w": v[1]} for k, v in states.items()},
                        "duty_cycle": duty_rows}
    del model, x
    gc.collect(); torch.cuda.empty_cache()

    # ---------- experiment 3: large loads, memory binds ----------
    def build(name):
        if name == "vit_l16":
            return tvm.vit_l_16(weights=None).cuda().eval(), (3, 224, 224)
        if name == "resnet152":
            return tvm.resnet152(weights=None).cuda().eval(), (3, 224, 224)
        enc = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=2048, nhead=16, dim_feedforward=8192,
                                             batch_first=True),
            num_layers=12).cuda().eval()
        return enc, (512, 2048)

    big_rows = []
    for name, batches in BIG:
        try:
            model, shape = build(name)
        except Exception as e:
            big_rows.append({"load": name, "status": "build_failed", "error": str(e)[:200]})
            continue
        for prec in ("fp16", "fp32"):
            for bs in batches:
                rec = {"machine": machine, "load": name, "precision": prec, "batch": bs}
                try:
                    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                    xb = torch.randn(bs, *shape, device="cuda")
                    amp = (prec == "fp16")
                    with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=amp):
                        for _ in range(2):
                            model(xb)
                    torch.cuda.synchronize()
                    it = 0; el = 0.0; pk = 0.0; ps = []
                    e0 = energy_mj(); t0 = time.perf_counter()
                    while el < 4.0 and it < 500:
                        with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=amp):
                            model(xb)
                        it += 1
                        if it % 2 == 0:
                            torch.cuda.synchronize(); el = time.perf_counter() - t0
                            w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                            ps.append(w); pk = max(pk, w)
                    torch.cuda.synchronize(); el = time.perf_counter() - t0
                    e1 = energy_mj()
                    j = (e1 - e0) / 1000.0 if (e0 and e1) else None
                    ns = it * bs
                    rec.update({"status": "ok", "iters": it, "runtime_s": round(el, 3),
                                "energy_j": round(j, 3) if j else None,
                                "energy_per_sample_mj": round(j / ns * 1000, 4) if j else None,
                                "throughput_sps": round(ns / el, 2),
                                "peak_power_w": round(pk, 1),
                                "mean_power_w": round(sum(ps) / len(ps), 1) if ps else None,
                                "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)})
                    del xb
                except torch.cuda.OutOfMemoryError:
                    rec.update({"status": "oom"})          # a real value, not a failure
                    torch.cuda.empty_cache()
                except Exception as e:
                    rec.update({"status": "error", "error": str(e)[:200]})
                big_rows.append(rec)
        del model
        gc.collect(); torch.cuda.empty_cache()
    out["n4_large"] = big_rows
    pynvml.nvmlShutdown()
    return out


@app.function(gpu="T4", timeout=3600)
def t4(): return measure("T4")
@app.function(gpu="L4", timeout=3600)
def l4(): return measure("L4")
@app.function(gpu="A10G", timeout=3600)
def a10g(): return measure("A10G")
@app.function(gpu="L40S", timeout=3600)
def l40s(): return measure("L40S")
@app.function(gpu="A100-40GB", timeout=3600)
def a100(): return measure("A100-40GB")


@app.local_entrypoint()
def main():
    calls = {"T4": t4.spawn(), "L4": l4.spawn(), "A10G": a10g.spawn(),
             "L40S": l40s.spawn(), "A100-40GB": a100.spawn()}
    out = []
    for name, c in calls.items():
        try:
            r = c.get(); out.append(r)
            ok = sum(1 for x in r["n4_large"] if x.get("status") == "ok")
            oom = sum(1 for x in r["n4_large"] if x.get("status") == "oom")
            print(f"[{name}] ok={ok} oom={oom}  deep_idle="
                  f"{r['exec_idle']['states']['deep_idle']['median_w']}W")
        except Exception as e:
            print(f"[{name}] FAILED {e}")
    print("===EI_N4_JSON_START===")
    print(json.dumps(out))
    print("===EI_N4_JSON_END===")
