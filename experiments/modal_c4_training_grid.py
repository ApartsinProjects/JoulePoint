"""
C4: the full training grid, plus a power-cap privilege probe.

Why this matters more than its cost suggests. Section 9 has to concede F3: the L4 is
energy-optimal in all 120 INFERENCE cells, so on that grid performance-first and
energy-first selection could never coincide, and the 41.6 per cent median penalty is partly
a statement about the candidate set. The N4 redesign broke that pattern but only on a narrow
memory-staircase subset chosen to induce OOM. This runs the full workload x configuration
grid in TRAINING mode on every machine, which is the regime where optimiser state makes
memory bind and where distinct winners actually appear.

Training differs from inference in exactly the way that matters here: gradients and Adam
moments add roughly 16 bytes per parameter, so the memory ceiling moves from "activations"
to "parameters", and the accelerator ordering is no longer dominated by one efficient part.

Bundled second experiment, costing pennies: a PRIVILEGE PROBE. E1 in the register is blocked
because nvmlDeviceSetPowerManagementLimit returned Insufficient Permissions on L4 and A100
under Modal serverless, and MIG reported Not Supported. That was measured on older machine
types and on a different workspace. Re-testing on H100 and H200 is nearly free and, if it
succeeds, unblocks a whole axis (allocation amount: power caps, clocks, partitioning) that
is currently cited from other people's work rather than measured.

OOM is recorded as a cell value, not an error: memory infeasibility is the observation that
Section 7.2's feasibility model is fitted against.
"""

import json
import modal

APP = "greenmatch-c4-training"

image = (
    modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel", add_python=None)
    .pip_install("nvidia-ml-py==12.560.30", "numpy", "transformers==4.44.2")
)

app = modal.App(APP, image=image)

BATCHES = [8, 32]
PRECISIONS = ["fp32", "fp16"]
LOADS = ["resnet50", "vit_b16", "convnext_t", "transformer", "bert_large"]


def probe_privileges(h, pynvml):
    """Can this container control power limit, clocks or partitioning? E1 in the register."""
    out = {}
    try:
        cur = pynvml.nvmlDeviceGetEnforcedPowerLimit(h)
        lo, hi = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)
        out["power_limit_constraints_w"] = [lo / 1000.0, hi / 1000.0]
        out["power_limit_current_w"] = cur / 1000.0
        try:
            target = int((lo + cur) / 2)
            pynvml.nvmlDeviceSetPowerManagementLimit(h, target)
            out["set_power_limit"] = "OK"
            pynvml.nvmlDeviceSetPowerManagementLimit(h, cur)   # restore
        except Exception as e:
            out["set_power_limit"] = "DENIED: {}".format(str(e)[:120])
    except Exception as e:
        out["power_limit_query"] = "FAILED: {}".format(str(e)[:120])
    try:
        out["mig_mode"] = pynvml.nvmlDeviceGetMigMode(h)
    except Exception as e:
        out["mig_mode"] = "unsupported: {}".format(str(e)[:100])
    try:
        pynvml.nvmlDeviceSetApplicationsClocks(h, 877, 1000)
        out["set_clocks"] = "OK"
        try:
            pynvml.nvmlDeviceResetApplicationsClocks(h)
        except Exception:
            pass
    except Exception as e:
        out["set_clocks"] = "DENIED: {}".format(str(e)[:120])
    return out


def run_training(machine: str) -> dict:
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
    priv = probe_privileges(h, pynvml)

    def energy_mj():
        try:
            return pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        except Exception:
            return None

    def build(load):
        if load == "resnet50":
            return tvm.resnet50(weights=None).cuda().train(), ("img", (3, 224, 224))
        if load == "vit_b16":
            return tvm.vit_b_16(weights=None).cuda().train(), ("img", (3, 224, 224))
        if load == "convnext_t":
            return tvm.convnext_tiny(weights=None).cuda().train(), ("img", (3, 224, 224))
        if load == "transformer":
            enc = torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(
                    d_model=1024, nhead=16, dim_feedforward=4096, batch_first=True
                ),
                num_layers=6,
            ).cuda().train()
            return enc, ("seq", (256, 1024))
        if load == "bert_large":
            from transformers import BertConfig, BertModel
            cfg = BertConfig(hidden_size=1024, num_hidden_layers=24,
                             num_attention_heads=16, intermediate_size=4096,
                             vocab_size=30522, max_position_embeddings=512)
            return BertModel(cfg).cuda().train(), ("ids", (384,))
        raise ValueError(load)

    rows = []
    for load in LOADS:
        for prec in PRECISIONS:
            for bs in BATCHES:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                rec = {"machine": machine, "device": dev_name, "load": load,
                       "precision": prec, "batch": bs, "mode": "train"}
                model = opt = None
                try:
                    model, (kind, shape) = build(load)
                    nparam = sum(p.numel() for p in model.parameters())
                    rec["params_m"] = round(nparam / 1e6, 2)
                    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
                    scaler = torch.amp.GradScaler("cuda", enabled=(prec == "fp16"))
                    amp = (prec == "fp16")

                    if kind == "ids":
                        x = torch.randint(0, 30000, (bs, shape[0]), device="cuda")
                        fwd = lambda: model(input_ids=x).last_hidden_state.float().pow(2).mean()
                    else:
                        x = torch.randn(bs, *shape, device="cuda")
                        def fwd():
                            o = model(x)
                            o = o.last_hidden_state if hasattr(o, "last_hidden_state") else o
                            return o.float().pow(2).mean()

                    def step():
                        opt.zero_grad(set_to_none=True)
                        with torch.autocast("cuda", torch.float16, enabled=amp):
                            loss = fwd()
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()

                    for _ in range(3):
                        step()
                    torch.cuda.synchronize()

                    iters, elapsed, peak_w = 0, 0.0, 0.0
                    e0 = energy_mj()
                    t0 = time.perf_counter()
                    power_samples = []
                    while elapsed < 4.0 and iters < 500:
                        step()
                        iters += 1
                        if iters % 3 == 0:
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
                except Exception as e:
                    rec.update({"status": "error", "error": str(e)[:300]})
                finally:
                    del model, opt
                    torch.cuda.empty_cache()
                rows.append(rec)

    pynvml.nvmlShutdown()
    return {"machine": machine, "device": dev_name, "power_cap_w": power_cap,
            "mem_total_gb": round(mem_total, 1), "privileges": priv, "rows": rows}


@app.function(gpu="T4", timeout=5400)
def tr_t4():
    return run_training("T4")

@app.function(gpu="L4", timeout=5400)
def tr_l4():
    return run_training("L4")

@app.function(gpu="A10G", timeout=5400)
def tr_a10g():
    return run_training("A10G")

@app.function(gpu="L40S", timeout=5400)
def tr_l40s():
    return run_training("L40S")

@app.function(gpu="A100-40GB", timeout=5400)
def tr_a100():
    return run_training("A100-40GB")

@app.function(gpu="H100", timeout=5400)
def tr_h100():
    return run_training("H100")

@app.function(gpu="H200", timeout=5400)
def tr_h200():
    return run_training("H200")


@app.local_entrypoint()
def main():
    handles = {
        "T4": tr_t4.spawn(), "L4": tr_l4.spawn(), "A10G": tr_a10g.spawn(),
        "L40S": tr_l40s.spawn(), "A100-40GB": tr_a100.spawn(),
        "H100": tr_h100.spawn(), "H200": tr_h200.spawn(),
    }
    out = {}
    for name, hd in handles.items():
        try:
            out[name] = hd.get()
            ok = sum(1 for r in out[name]["rows"] if r.get("status") == "ok")
            oom = sum(1 for r in out[name]["rows"] if r.get("status") == "oom")
            print(f"[{name}] {out[name]['device']} mem={out[name]['mem_total_gb']}GB "
                  f"ok={ok} oom={oom}/{len(out[name]['rows'])} "
                  f"powerlimit={out[name]['privileges'].get('set_power_limit')}")
        except Exception as e:
            out[name] = {"error": str(e)[:400]}
            print(f"[{name}] FAILED: {str(e)[:200]}")
    print(json.dumps(out))
