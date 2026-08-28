"""
E0. Replicate noise and idle power.

Two quantities every earlier conclusion depends on and neither of which we have measured:
  (a) within-cell variance across INDEPENDENT container invocations, which sets the
      noise floor against which the 2.5% interaction residual must be judged;
  (b) true idle power per machine, needed for the facility-level accounting in E1.

Each repetition is a separate .spawn(), so repetitions land in separate containers
on separate physical devices rather than looping inside one process.
"""
import json
import modal

APP = "greenmatch-e0-noise"
image = (modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel", add_python=None)
         .pip_install("nvidia-ml-py==12.560.30", "numpy"))
app = modal.App(APP, image=image)

REPS = 5
CELLS = [("resnet50", "fp16", 32), ("resnet50", "fp32", 32),
         ("vit_b16", "fp16", 32), ("vit_b16", "fp32", 32),
         ("transformer", "fp16", 32), ("transformer", "fp32", 32)]


def measure(machine: str, rep: int) -> dict:
    import time, torch, pynvml
    import torchvision.models as tvm

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(h)
    if isinstance(name, bytes):
        name = name.decode()
    try:
        cap = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
    except Exception:
        cap = None

    def energy_mj():
        try:
            return pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        except Exception:
            return None

    # ---- idle power, before any work (cold) ----
    idle_cold = []
    t0 = time.time()
    while time.time() - t0 < 12:
        idle_cold.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
        time.sleep(0.25)

    def build(load):
        if load == "resnet50":
            return tvm.resnet50(weights=None).cuda().eval(), (3, 224, 224)
        if load == "vit_b16":
            return tvm.vit_b_16(weights=None).cuda().eval(), (3, 224, 224)
        enc = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=1024, nhead=16,
                                             dim_feedforward=4096, batch_first=True),
            num_layers=6).cuda().eval()
        return enc, (256, 1024)

    rows = []
    for load, prec, bs in CELLS:
        model, shape = build(load)
        x = torch.randn(bs, *shape, device="cuda")
        amp = (prec == "fp16")
        with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=amp):
            for _ in range(3):
                model(x)
        torch.cuda.synchronize()

        iters, elapsed, peak_w, samples_p = 0, 0.0, 0.0, []
        e0 = energy_mj(); t0 = time.perf_counter()
        while elapsed < 4.0 and iters < 3000:
            with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=amp):
                model(x)
            iters += 1
            if iters % 5 == 0:
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                samples_p.append(w); peak_w = max(peak_w, w)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        e1 = energy_mj()
        j = (e1 - e0) / 1000.0 if (e0 is not None and e1 is not None) else None
        n = iters * bs
        rows.append({"machine": machine, "rep": rep, "load": load, "precision": prec,
                     "batch": bs, "iters": iters, "runtime_s": round(elapsed, 4),
                     "energy_j": round(j, 3) if j else None,
                     "energy_per_sample_mj": round(j / n * 1000, 4) if j else None,
                     "peak_power_w": round(peak_w, 1),
                     "mean_power_w": round(sum(samples_p) / len(samples_p), 1) if samples_p else None})
        del x, model
        torch.cuda.empty_cache()

    # ---- idle power again, after work (warm) ----
    idle_warm = []
    t0 = time.time()
    while time.time() - t0 < 12:
        idle_warm.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
        time.sleep(0.25)

    import statistics as st
    pynvml.nvmlShutdown()
    return {"machine": machine, "rep": rep, "device": name, "power_cap_w": cap,
            "idle_cold_w": round(st.median(idle_cold), 1),
            "idle_warm_w": round(st.median(idle_warm), 1),
            "rows": rows}


@app.function(gpu="T4", timeout=1800)
def t4(rep): return measure("T4", rep)


@app.function(gpu="L4", timeout=1800)
def l4(rep): return measure("L4", rep)


@app.function(gpu="A10G", timeout=1800)
def a10g(rep): return measure("A10G", rep)


@app.function(gpu="L40S", timeout=1800)
def l40s(rep): return measure("L40S", rep)


@app.function(gpu="A100-40GB", timeout=1800)
def a100(rep): return measure("A100-40GB", rep)


@app.local_entrypoint()
def main():
    fns = {"T4": t4, "L4": l4, "A10G": a10g, "L40S": l40s, "A100-40GB": a100}
    calls = []
    for name, fn in fns.items():
        for rep in range(REPS):
            calls.append((name, rep, fn.spawn(rep)))
    out = []
    for name, rep, c in calls:
        try:
            out.append(c.get())
            print(f"[{name} rep{rep}] ok")
        except Exception as e:
            print(f"[{name} rep{rep}] FAILED {e}")
    print("===E0_JSON_START===")
    print(json.dumps(out))
    print("===E0_JSON_END===")
