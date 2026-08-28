"""
T4 replicate run: is the +10.5 per cent cross-run drift card-to-card variance or systematic?

The C1 bridge re-executed the original grid months later on a different cloud account. Four
of five machines reproduced within 2.4 per cent, at or near the 1.64 per cent within-run
replicate figure. The T4 did not: median +10.5 per cent. Two explanations fit, and they have
different consequences for the paper:

  (a) CARD-TO-CARD VARIANCE. The T4 is the oldest part in the set and the most variably
      binned, and at a 70 W cap a fixed overhead is proportionally largest. If so, repeated
      independent containers should show a WIDE spread that brackets both the old and new
      values, and the paper simply needs a larger tolerance on T4 absolute energies.

  (b) SOMETHING SYSTEMATIC. A driver, clock or firmware difference between the two accounts
      or two points in time. If so, repeated containers should cluster TIGHTLY around the new
      value and the old T4 measurements would need re-stating rather than widening.

The discriminator is the spread, not the mean, so this run measures the spread directly:
eight independent containers, each re-measuring a fixed subset of T4 cells. Independent
containers, not repeats inside one container, because card assignment is what is in question.

Predictions stated in advance:
  P1  if (a), the across-container standard deviation on the T4 is much larger than the
      1.64 per cent within-run figure, and the old value lies inside the observed range
  P2  if (b), the across-container spread stays near 1.64 per cent and the old value lies
      outside it
  S1  every cell from the NVML hardware counter, no fallback
  S2  every container reports device 'Tesla T4'; any other device invalidates the comparison
  S3  the enforced power limit is identical across containers, since a differing cap would
      be a third explanation and would need separating from the other two
"""

import json
import modal

APP = "greenmatch-t4-replicates"
image = (
    modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel", add_python=None)
    .pip_install("nvidia-ml-py==12.560.30", "numpy")
)
app = modal.App(APP, image=image)

# the cells with the largest observed drift, plus a stable control
CELLS = [("convnext_t", "fp32", 8), ("convnext_t", "fp32", 128),
         ("vit_b16", "fp16", 128), ("transformer", "fp32", 8),
         ("resnet50", "fp16", 32)]


def measure_once(tag: str) -> dict:
    import time
    import torch
    import torchvision.models as tvm
    import pynvml

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    dev = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(h)
    if isinstance(name, bytes):
        name = name.decode()
    try:
        cap = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
    except Exception:
        cap = None
    try:
        uuid = pynvml.nvmlDeviceGetUUID(h)
        if isinstance(uuid, bytes):
            uuid = uuid.decode()
    except Exception:
        uuid = None
    try:
        serial = pynvml.nvmlDeviceGetSerial(h)
        if isinstance(serial, bytes):
            serial = serial.decode()
    except Exception:
        serial = None

    def build(load):
        if load == "resnet50":
            return tvm.resnet50(weights=None).cuda().eval(), (3, 224, 224)
        if load == "vit_b16":
            return tvm.vit_b_16(weights=None).cuda().eval(), (3, 224, 224)
        if load == "convnext_t":
            return tvm.convnext_tiny(weights=None).cuda().eval(), (3, 224, 224)
        if load == "transformer":
            return torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(d_model=1024, nhead=16,
                                                 dim_feedforward=4096, batch_first=True),
                num_layers=6).cuda().eval(), (256, 1024)
        raise ValueError(load)

    rows = []
    for load, prec, bs in CELLS:
        model, shape = build(load)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        rec = {"container": tag, "device": name, "uuid": uuid, "serial": serial,
               "power_cap_w": cap, "load": load, "precision": prec, "batch": bs}
        try:
            x = torch.randn(bs, *shape, device="cuda")
            amp = (prec == "fp16")
            with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=amp):
                for _ in range(3):
                    model(x)
            torch.cuda.synchronize()
            iters, elapsed = 0, 0.0
            e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
            t0 = time.perf_counter()
            ps = []
            while elapsed < 4.0 and iters < 2000:
                with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=amp):
                    model(x)
                iters += 1
                if iters % 5 == 0:
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - t0
                    try:
                        ps.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
                    except Exception:
                        pass
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
            j = (e1 - e0) / 1000.0
            n = iters * bs
            rec.update({"status": "ok", "runtime_s": round(elapsed, 4), "iters": iters,
                        "energy_j": round(j, 3), "samples": n,
                        "energy_per_sample_mj": round(j / n * 1000, 4),
                        "throughput_sps": round(n / elapsed, 2),
                        "mean_power_w": round(sum(ps) / len(ps), 1) if ps else None,
                        "energy_source": "nvml_counter"})
            del x
        except Exception as e:
            rec.update({"status": "error", "error": str(e)[:200]})
        rows.append(rec)
        del model
        torch.cuda.empty_cache()
    pynvml.nvmlShutdown()
    return {"container": tag, "device": name, "uuid": uuid, "serial": serial,
            "power_cap_w": cap, "rows": rows}


@app.function(gpu="T4", timeout=1800)
def rep(tag: str):
    return measure_once(tag)


@app.local_entrypoint()
def main():
    # eight INDEPENDENT containers: card assignment is the variable under test
    handles = [rep.spawn("c{}".format(i)) for i in range(8)]
    out = []
    for i, hd in enumerate(handles):
        try:
            r = hd.get()
            out.append(r)
            print("[c{}] {} uuid={} cap={}W ok={}".format(
                i, r["device"], (r["uuid"] or "?")[-12:], r["power_cap_w"],
                sum(1 for x in r["rows"] if x.get("status") == "ok")))
        except Exception as e:
            print("[c{}] FAILED {}".format(i, str(e)[:160]))
    print(json.dumps(out))
