"""
C1: the cross-corpus bridge.

Section 9 currently has to declare that cross-corpus transfer cannot be tested, because our
grid and the MLPerf datacenter set share only two accelerators and zero workloads. A model
fitted on one cannot be evaluated on the other with nothing in common. This run closes that
gap from both sides at once:

  accelerator bridge   our four existing workloads on H100 and H200, both of which appear
                       in the MLPerf datacenter results and are rentable on Modal. Shared
                       accelerators go from 2 to 4.

  workload bridge      BERT-Large, an MLPerf datacenter inference benchmark, added to all
                       machines. Shared workloads go from 0 to 2 (ResNet-50 is already in
                       both). BERT is constructed from config rather than downloaded
                       pretrained, exactly as the other loads are, so the run needs no
                       network and no credentials.

Measurement is identical to the original pilot: the NVML total-energy counter with a
power-integration fallback, a warmup, and a measured window of at least four seconds.
Keeping the protocol byte-identical is the point, since the whole purpose is comparability.

Every cell records energy, runtime, peak power, peak memory and throughput. OOM is recorded
as a cell value rather than an error, because memory infeasibility is a real observation
that Section 7.2 depends on.
"""

import json
import modal

APP = "greenmatch-c1-bridge"

image = (
    modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel", add_python=None)
    .pip_install("nvidia-ml-py==12.560.30", "numpy", "transformers==4.44.2")
)

app = modal.App(APP, image=image)

BATCHES = [8, 32, 128]
PRECISIONS = ["fp32", "fp16"]
LOADS = ["resnet50", "vit_b16", "convnext_t", "transformer", "bert_large"]


def run_suite(machine: str) -> dict:
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
            return tvm.resnet50(weights=None).cuda().eval(), ("img", (3, 224, 224))
        if load == "vit_b16":
            return tvm.vit_b_16(weights=None).cuda().eval(), ("img", (3, 224, 224))
        if load == "convnext_t":
            return tvm.convnext_tiny(weights=None).cuda().eval(), ("img", (3, 224, 224))
        if load == "transformer":
            enc = torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(
                    d_model=1024, nhead=16, dim_feedforward=4096, batch_first=True
                ),
                num_layers=6,
            ).cuda().eval()
            return enc, ("seq", (256, 1024))
        if load == "bert_large":
            # MLPerf datacenter inference uses BERT-Large at sequence length 384.
            # Built from config, not pretrained weights: identical FLOPs and memory,
            # no network dependency, and weights do not affect energy (Section 8).
            from transformers import BertConfig, BertModel
            cfg = BertConfig(hidden_size=1024, num_hidden_layers=24,
                             num_attention_heads=16, intermediate_size=4096,
                             vocab_size=30522, max_position_embeddings=512)
            return BertModel(cfg).cuda().eval(), ("ids", (384,))
        raise ValueError(load)

    rows = []
    for load in LOADS:
        try:
            model, (kind, shape) = build(load)
        except Exception as e:
            rows.append({"load": load, "status": "build_failed", "error": str(e)[:300]})
            continue

        for prec in PRECISIONS:
            for bs in BATCHES:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                rec = {"machine": machine, "device": dev_name, "load": load,
                       "precision": prec, "batch": bs}
                try:
                    if kind == "ids":
                        x = torch.randint(0, 30000, (bs, shape[0]), device="cuda")
                        call = lambda: model(input_ids=x)
                    else:
                        x = torch.randn(bs, *shape, device="cuda")
                        call = lambda: model(x)
                    amp = (prec == "fp16")

                    with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=amp):
                        for _ in range(3):
                            call()
                    torch.cuda.synchronize()

                    iters, elapsed, peak_w = 0, 0.0, 0.0
                    e0 = energy_mj()
                    t0 = time.perf_counter()
                    power_samples = []
                    while elapsed < 4.0 and iters < 2000:
                        with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=amp):
                            call()
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
                        joules, e_src = (e1 - e0) / 1000.0, "nvml_counter"
                    elif power_samples:
                        joules = (sum(power_samples) / len(power_samples)) * elapsed
                        e_src = "power_integration"
                    else:
                        joules, e_src = None, "unavailable"

                    samples = iters * bs
                    rec.update({
                        "status": "ok", "iters": iters, "runtime_s": round(elapsed, 4),
                        "energy_j": round(joules, 3) if joules else None,
                        "peak_power_w": round(peak_w, 1),
                        "mean_power_w": round(sum(power_samples) / len(power_samples), 1)
                                        if power_samples else None,
                        "samples": samples,
                        "throughput_sps": round(samples / elapsed, 2),
                        "energy_per_sample_mj": round(joules / samples * 1000, 4) if joules else None,
                        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
                        "energy_source": e_src,
                    })
                    del x
                except torch.cuda.OutOfMemoryError:
                    rec.update({"status": "oom"})
                    torch.cuda.empty_cache()
                except Exception as e:
                    rec.update({"status": "error", "error": str(e)[:300]})
                rows.append(rec)

        del model
        torch.cuda.empty_cache()

    pynvml.nvmlShutdown()
    return {"machine": machine, "device": dev_name, "power_cap_w": power_cap,
            "mem_total_gb": round(mem_total, 1), "rows": rows}


# One decorated function per machine: Modal resolves remote functions by module-level name,
# so these cannot be generated in a loop.
@app.function(gpu="T4", timeout=3600)
def bench_t4():
    return run_suite("T4")

@app.function(gpu="L4", timeout=3600)
def bench_l4():
    return run_suite("L4")

@app.function(gpu="A10G", timeout=3600)
def bench_a10g():
    return run_suite("A10G")

@app.function(gpu="L40S", timeout=3600)
def bench_l40s():
    return run_suite("L40S")

@app.function(gpu="A100-40GB", timeout=3600)
def bench_a100():
    return run_suite("A100-40GB")

@app.function(gpu="H100", timeout=3600)
def bench_h100():
    return run_suite("H100")

@app.function(gpu="H200", timeout=3600)
def bench_h200():
    return run_suite("H200")


@app.local_entrypoint()
def main():
    handles = {
        "T4": bench_t4.spawn(),
        "L4": bench_l4.spawn(),
        "A10G": bench_a10g.spawn(),
        "L40S": bench_l40s.spawn(),
        "A100-40GB": bench_a100.spawn(),
        "H100": bench_h100.spawn(),
        "H200": bench_h200.spawn(),
    }
    out = {}
    for name, hd in handles.items():
        try:
            out[name] = hd.get()
            ok = sum(1 for r in out[name]["rows"] if r.get("status") == "ok")
            oom = sum(1 for r in out[name]["rows"] if r.get("status") == "oom")
            print(f"[{name}] device={out[name]['device']} mem={out[name]['mem_total_gb']}GB "
                  f"ok={ok} oom={oom} of {len(out[name]['rows'])}")
        except Exception as e:
            out[name] = {"error": str(e)[:400]}
            print(f"[{name}] FAILED: {str(e)[:200]}")
    print(json.dumps(out))
