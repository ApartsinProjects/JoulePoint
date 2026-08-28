"""
GreenMatch-AI pilot: workload x machine x configuration energy measurement on Modal.

Produces the preliminary-results table for section 2.5.5.8 of the Call 34/2026 proposal:
a small slice of the sparse observation matrix, measured on real heterogeneous hardware.

Energy is read from NVML's total-energy counter (millijoule resolution, Volta and later)
with a power-integration fallback. Every cell records energy, runtime and peak power, which
are exactly the three outputs the proposed model predicts.
"""

import json
import modal

APP = "greenmatch-energy-pilot"

image = (
    modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel", add_python=None)
    .pip_install("nvidia-ml-py==12.560.30", "numpy")
)

app = modal.App(APP, image=image)

# machines under test: five accelerator families spanning three architecture generations
GPUS = ["T4", "L4", "A10G", "L40S", "A100-40GB"]

# loads x configurations
BATCHES = [8, 32, 128]
PRECISIONS = ["fp32", "fp16"]


def run_suite(machine: str) -> dict:
    """Execute the full load x configuration sweep on whichever GPU this container holds."""
    import time
    import torch
    import torchvision.models as tvm
    import pynvml

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    dev_name = pynvml.nvmlDeviceGetName(h)
    if isinstance(dev_name, bytes):
        dev_name = dev_name.decode()

    try:
        power_cap = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
    except Exception:
        power_cap = None
    mem_total = pynvml.nvmlDeviceGetMemoryInfo(h).total / 1e9

    def energy_mj():
        try:
            return pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        except Exception:
            return None

    def build(load: str):
        if load == "resnet50":
            return tvm.resnet50(weights=None).cuda().eval(), (3, 224, 224)
        if load == "vit_b16":
            return tvm.vit_b_16(weights=None).cuda().eval(), (3, 224, 224)
        if load == "convnext_t":
            return tvm.convnext_tiny(weights=None).cuda().eval(), (3, 224, 224)
        if load == "transformer":
            enc = torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(
                    d_model=1024, nhead=16, dim_feedforward=4096, batch_first=True
                ),
                num_layers=6,
            ).cuda().eval()
            return enc, (256, 1024)  # sequence length x model dim
        raise ValueError(load)

    rows = []
    for load in ["resnet50", "vit_b16", "convnext_t", "transformer"]:
        try:
            model, shape = build(load)
        except Exception as e:  # a load that will not even construct here
            rows.append({"load": load, "status": "build_failed", "error": str(e)[:200]})
            continue

        for prec in PRECISIONS:
            for bs in BATCHES:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                rec = {
                    "machine": machine, "device": dev_name, "load": load,
                    "precision": prec, "batch": bs,
                }
                try:
                    x = torch.randn(bs, *shape, device="cuda")
                    amp = (prec == "fp16")

                    # warmup
                    with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=amp):
                        for _ in range(3):
                            model(x)
                    torch.cuda.synchronize()

                    # measured window: repeat until at least ~4 s of work
                    iters, elapsed = 0, 0.0
                    peak_w = 0.0
                    e0 = energy_mj()
                    t0 = time.perf_counter()
                    power_samples = []
                    while elapsed < 4.0 and iters < 2000:
                        with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=amp):
                            model(x)
                        iters += 1
                        if iters % 5 == 0:
                            torch.cuda.synchronize()
                            elapsed = time.perf_counter() - t0
                            try:
                                w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                                power_samples.append(w)
                                peak_w = max(peak_w, w)
                            except Exception:
                                pass
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - t0
                    e1 = energy_mj()

                    if e0 is not None and e1 is not None and e1 >= e0:
                        joules = (e1 - e0) / 1000.0
                        e_src = "nvml_counter"
                    elif power_samples:
                        joules = (sum(power_samples) / len(power_samples)) * elapsed
                        e_src = "power_integration"
                    else:
                        joules, e_src = None, "unavailable"

                    samples = iters * bs
                    rec.update({
                        "status": "ok",
                        "iters": iters,
                        "runtime_s": round(elapsed, 4),
                        "energy_j": round(joules, 3) if joules else None,
                        "peak_power_w": round(peak_w, 1),
                        "mean_power_w": round(sum(power_samples) / len(power_samples), 1) if power_samples else None,
                        "samples": samples,
                        "throughput_sps": round(samples / elapsed, 2),
                        "energy_per_sample_mj": round(joules / samples * 1000, 4) if joules else None,
                        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
                        "energy_source": e_src,
                    })
                    del x
                except torch.cuda.OutOfMemoryError:
                    # memory infeasibility is a real cell value, not a failure
                    rec.update({"status": "oom"})
                    torch.cuda.empty_cache()
                except Exception as e:
                    rec.update({"status": "error", "error": str(e)[:200]})
                rows.append(rec)

        del model
        torch.cuda.empty_cache()

    pynvml.nvmlShutdown()
    return {"machine": machine, "device": dev_name, "power_cap_w": power_cap,
            "mem_total_gb": round(mem_total, 1), "rows": rows}


# One decorated function per machine. Written out explicitly rather than generated in a
# loop, because Modal resolves remote functions by module-level name.
@app.function(gpu="T4", timeout=1800)
def bench_t4():
    return run_suite("T4")


@app.function(gpu="L4", timeout=1800)
def bench_l4():
    return run_suite("L4")


@app.function(gpu="A10G", timeout=1800)
def bench_a10g():
    return run_suite("A10G")


@app.function(gpu="L40S", timeout=1800)
def bench_l40s():
    return run_suite("L40S")


@app.function(gpu="A100-40GB", timeout=1800)
def bench_a100():
    return run_suite("A100-40GB")


@app.local_entrypoint()
def main():
    calls = {
        "T4": bench_t4.spawn(),
        "L4": bench_l4.spawn(),
        "A10G": bench_a10g.spawn(),
        "L40S": bench_l40s.spawn(),
        "A100-40GB": bench_a100.spawn(),
    }
    out = []
    for name, call in calls.items():
        try:
            res = call.get()
            out.append(res)
            ok = sum(1 for r in res["rows"] if r.get("status") == "ok")
            print(f"[{name}] {res['device']}  ok={ok}/{len(res['rows'])}  cap={res['power_cap_w']}W")
        except Exception as e:
            print(f"[{name}] FAILED: {e}")
    print("===RESULTS_JSON_START===")
    print(json.dumps(out))
    print("===RESULTS_JSON_END===")
