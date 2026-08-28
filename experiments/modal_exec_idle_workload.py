"""
Execution-idle at WORKLOAD level.

We measured the duty-cycle power curve per machine, but not which workload-hardware
pairs strand the most energy while allocated. Reported cluster figures put
execution-idle at 10.7 per cent of energy overall and 48 per cent for serving, so
the question is which pairs contribute it.

For each (workload, machine) we hold the device allocated and drive it at several
duty cycles, then report the energy actually spent per unit of useful work against
the energy that would be spent at full occupancy. The gap is the stranded energy,
and it is a property of the pair, not of the device alone.
"""
import json
import modal

image = (modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel", add_python=None)
         .pip_install("nvidia-ml-py==12.560.30", "numpy"))
app = modal.App("greenmatch-execidle-workload", image=image)

LOADS = ["resnet50", "vit_b16", "convnext_t", "transformer"]
DUTY = [0.10, 0.25, 0.50, 1.00]
BATCH = 32


def measure(machine):
    import time, gc, torch, pynvml
    import torchvision.models as tvm
    import statistics as st

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    dev = pynvml.nvmlDeviceGetName(h)
    dev = dev.decode() if isinstance(dev, bytes) else dev
    cap = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0

    def energy_mj():
        try:
            return pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        except Exception:
            return None

    def build(name):
        if name == "resnet50":
            return tvm.resnet50(weights=None).cuda().eval(), (3, 224, 224)
        if name == "vit_b16":
            return tvm.vit_b_16(weights=None).cuda().eval(), (3, 224, 224)
        if name == "convnext_t":
            return tvm.convnext_tiny(weights=None).cuda().eval(), (3, 224, 224)
        enc = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=1024, nhead=16, dim_feedforward=4096,
                                             batch_first=True), num_layers=6).cuda().eval()
        return enc, (256, 1024)

    rows = []
    for name in LOADS:
        model, shape = build(name)
        x = torch.randn(BATCH, *shape, device="cuda")
        with torch.no_grad():
            for _ in range(3):
                model(x)
        torch.cuda.synchronize()
        # resident-but-idle power for this specific model on this machine
        s = []
        t0 = time.time()
        while time.time() - t0 < 8:
            s.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
            time.sleep(0.2)
        resident_w = float(st.median(s))

        for d in DUTY:
            period, window = 0.5, 14.0
            e0 = energy_mj(); t0 = time.perf_counter(); it = 0
            end = t0 + window
            while time.perf_counter() < end:
                cyc = time.perf_counter()
                busy_until = cyc + period * d
                with torch.no_grad():
                    while time.perf_counter() < busy_until:
                        model(x); it += 1
                torch.cuda.synchronize()
                rest = period - (time.perf_counter() - cyc)
                if rest > 0:
                    time.sleep(rest)
            el = time.perf_counter() - t0
            e1 = energy_mj()
            j = (e1 - e0) / 1000.0 if (e0 and e1) else None
            ns = it * BATCH
            rows.append({"machine": machine, "load": name, "duty": d,
                         "elapsed_s": round(el, 2), "iters": it, "samples": ns,
                         "energy_j": round(j, 2) if j else None,
                         "avg_power_w": round(j / el, 1) if j else None,
                         "energy_per_sample_mj": round(j / ns * 1000, 3) if (j and ns) else None,
                         "resident_idle_w": resident_w, "power_cap_w": cap})
        del model, x
        gc.collect(); torch.cuda.empty_cache()
    pynvml.nvmlShutdown()
    return {"machine": machine, "device": dev, "power_cap_w": cap, "rows": rows}


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
            print(f"[{name}] {len(r['rows'])} cells")
        except Exception as e:
            print(f"[{name}] FAILED {e}")
    print("===EIW_JSON_START===")
    print(json.dumps(out))
    print("===EIW_JSON_END===")
