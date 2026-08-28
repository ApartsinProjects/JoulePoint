"""
N4 redesign. The first attempt produced zero memory-infeasible cells because
inference under no_grad has negligible activation memory: only weights matter, and
ViT-L is 307M parameters (~1.2 GB), nowhere near a T4's 16 GB.

Fix: TRAINING mode. Parameters, gradients and Adam moments together cost roughly
16 bytes per parameter, so a synthetic transformer sized by parameter count gives a
predictable feasibility staircase across 16 / 24 / 40 / 48 GB devices. This also
removes the "inference only" limitation of the original grid.
"""
import json
import modal

image = (modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel", add_python=None)
         .pip_install("nvidia-ml-py==12.560.30", "numpy"))
app = modal.App("greenmatch-n4-redesign", image=image)

# (label, d_model, layers, approx params) sized to straddle 16/24/40/48 GB in training
SIZES = [("xf_0.25B", 1024, 16), ("xf_0.5B", 1536, 16), ("xf_1B", 2048, 20), ("xf_2B", 2560, 26)]
SEQ = 256
BATCH = {"train": [4, 16], "infer": [16, 64]}


def measure(machine):
    import time, gc, torch, pynvml

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    dev = pynvml.nvmlDeviceGetName(h)
    dev = dev.decode() if isinstance(dev, bytes) else dev
    cap = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
    total_gb = pynvml.nvmlDeviceGetMemoryInfo(h).total / 1e9

    def energy_mj():
        try:
            return pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        except Exception:
            return None

    def build(d, layers):
        return torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=d, nhead=16, dim_feedforward=4 * d,
                                             batch_first=True),
            num_layers=layers)

    rows = []
    for label, d, layers in SIZES:
        for mode in ("infer", "train"):
            for bs in BATCH[mode]:
                for prec in ("fp16", "fp32"):
                    rec = {"machine": machine, "load": label, "mode": mode,
                           "precision": prec, "batch": bs, "d_model": d, "layers": layers}
                    model = opt = x = None
                    try:
                        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                        model = build(d, layers).cuda()
                        nparam = sum(p.numel() for p in model.parameters())
                        rec["params_m"] = round(nparam / 1e6, 1)
                        amp = (prec == "fp16")
                        if mode == "train":
                            model.train()
                            opt = torch.optim.Adam(model.parameters(), lr=1e-4)
                        else:
                            model.eval()
                        x = torch.randn(bs, SEQ, d, device="cuda")

                        def step():
                            if mode == "train":
                                opt.zero_grad(set_to_none=True)
                                with torch.autocast("cuda", torch.float16, enabled=amp):
                                    loss = model(x).float().pow(2).mean()
                                loss.backward()
                                opt.step()
                            else:
                                with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=amp):
                                    model(x)

                        for _ in range(2):
                            step()
                        torch.cuda.synchronize()

                        it = 0; el = 0.0; pk = 0.0; ps = []
                        e0 = energy_mj(); t0 = time.perf_counter()
                        while el < 4.0 and it < 200:
                            step(); it += 1
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
                                    "throughput_sps": round(ns / el, 3),
                                    "peak_power_w": round(pk, 1),
                                    "mean_power_w": round(sum(ps) / len(ps), 1) if ps else None,
                                    "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)})
                    except torch.cuda.OutOfMemoryError:
                        rec.update({"status": "oom"})     # a value, not a failure
                    except Exception as e:
                        rec.update({"status": "error", "error": str(e)[:180]})
                    finally:
                        del model, opt, x
                        gc.collect(); torch.cuda.empty_cache()
                    rows.append(rec)
    pynvml.nvmlShutdown()
    return {"machine": machine, "device": dev, "power_cap_w": cap,
            "mem_total_gb": round(total_gb, 1), "rows": rows}


@app.function(gpu="T4", timeout=5400)
def t4(): return measure("T4")
@app.function(gpu="L4", timeout=5400)
def l4(): return measure("L4")
@app.function(gpu="A10G", timeout=5400)
def a10g(): return measure("A10G")
@app.function(gpu="L40S", timeout=5400)
def l40s(): return measure("L40S")
@app.function(gpu="A100-40GB", timeout=5400)
def a100(): return measure("A100-40GB")


@app.local_entrypoint()
def main():
    calls = {"T4": t4.spawn(), "L4": l4.spawn(), "A10G": a10g.spawn(),
             "L40S": l40s.spawn(), "A100-40GB": a100.spawn()}
    out = []
    for name, c in calls.items():
        try:
            r = c.get(); out.append(r)
            ok = sum(1 for x in r["rows"] if x.get("status") == "ok")
            oom = sum(1 for x in r["rows"] if x.get("status") == "oom")
            print(f"[{name}] mem={r['mem_total_gb']}GB ok={ok} OOM={oom}")
        except Exception as e:
            print(f"[{name}] FAILED {e}")
    print("===N4R_JSON_START===")
    print(json.dumps(out))
    print("===N4R_JSON_END===")
