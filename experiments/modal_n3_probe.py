"""N3. Probe whether power capping / MIG / clock control are available on serverless GPUs.
Decides whether E8 (allocation amount) is a feasible experiment at all."""
import json, modal
image = (modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel", add_python=None)
         .pip_install("nvidia-ml-py==12.560.30"))
app = modal.App("greenmatch-n3-probe", image=image)

def probe(machine):
    import subprocess, pynvml
    out = {"machine": machine}
    def sh(cmd):
        try:
            p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
            return {"rc": p.returncode, "out": p.stdout.strip()[:400], "err": p.stderr.strip()[:400]}
        except Exception as e:
            return {"rc": -1, "err": str(e)[:200]}
    pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(h)
    out["device"] = name.decode() if isinstance(name, bytes) else name
    try:
        out["cap_now_w"] = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
        mn, mx = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)
        out["cap_range_w"] = [mn / 1000.0, mx / 1000.0]
    except Exception as e:
        out["cap_range_error"] = str(e)[:200]
    out["persistence"] = sh("nvidia-smi -q -d PERSISTENCE_MODE | head -5")
    # try to actually set a cap
    try:
        mn, mx = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)
        target = int((mn + mx) / 2)
        out["set_cap_nvml"] = sh(f"nvidia-smi -pl {target//1000}")
        pynvml.nvmlDeviceSetPowerManagementLimit(h, target)
        out["set_cap_nvml_api"] = "OK"
        out["cap_after_w"] = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
    except Exception as e:
        out["set_cap_nvml_api"] = f"DENIED: {str(e)[:160]}"
    out["mig_mode"] = sh("nvidia-smi -q | grep -A2 'MIG Mode' | head -6")
    out["mig_enable"] = sh("nvidia-smi -mig 1")
    out["clocks"] = sh("nvidia-smi -q -d SUPPORTED_CLOCKS | head -8")
    out["set_clock"] = sh("nvidia-smi -lgc 500,900")
    # in-process levers that never need privileges
    import torch
    out["in_process_levers"] = {
        "batch_size": True, "precision_autocast": True,
        "cuda_streams": True, "torch_compile": hasattr(torch, "compile"),
        "sm_partition_via_MPS": "needs daemon, untested here",
    }
    pynvml.nvmlShutdown()
    return out

@app.function(gpu="L4", timeout=900)
def l4(): return probe("L4")
@app.function(gpu="A100-40GB", timeout=900)
def a100(): return probe("A100-40GB")

@app.local_entrypoint()
def main():
    res = [f.remote() for f in (l4, a100)]
    print("===N3_JSON_START===")
    print(json.dumps(res))
    print("===N3_JSON_END===")
