# -*- coding: utf-8 -*-
"""
Permission probe: can a Modal T4 container SET the GPU power limit (and/or lock clocks)?
This decides whether the power-cap elasticity sweep uses power-capping or a locked-clocks
fallback. Costs a few cents (one short T4 run). Prints a JSON verdict to the logs.
"""
import json
import modal

image = (
    modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel", add_python=None)
    .pip_install("nvidia-ml-py==12.560.30", "numpy")
)
app = modal.App("powershaping-powercap-probe", image=image)


@app.function(gpu="T4", timeout=300)
def probe() -> dict:
    import time, pynvml, torch

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(h)
    name = name.decode() if isinstance(name, bytes) else name
    out = {"device": name}

    def W(mw): return round(mw / 1000.0, 1)

    # current limits / constraints
    try:
        out["default_limit_w"] = W(pynvml.nvmlDeviceGetPowerManagementLimit(h))
        lo, hi = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)
        out["limit_constraints_w"] = [W(lo), W(hi)]
        out["enforced_limit_w"] = W(pynvml.nvmlDeviceGetEnforcedPowerLimit(h))
    except Exception as e:
        out["limit_read_error"] = str(e)[:200]

    # a steady GPU load so power is meaningful
    def busy(seconds=3.0):
        a = torch.randn(4096, 4096, device="cuda")
        b = torch.randn(4096, 4096, device="cuda")
        t0 = time.time(); n = 0
        while time.time() - t0 < seconds:
            a = (a @ b).relu(); n += 1
        torch.cuda.synchronize()
        return n

    def mean_power(seconds=2.0):
        s = []; t0 = time.time()
        while time.time() - t0 < seconds:
            s.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0); time.sleep(0.05)
        return round(sum(s) / len(s), 1)

    busy(2.0)
    out["power_at_default_w"] = mean_power()

    # --- TEST 1: set power management limit to ~60% of default ---
    try:
        lo, hi = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)
        target = int(max(lo, 0.6 * pynvml.nvmlDeviceGetPowerManagementLimit(h)))
        pynvml.nvmlDeviceSetPowerManagementLimit(h, target)
        readback = pynvml.nvmlDeviceGetEnforcedPowerLimit(h)
        busy(2.0)
        out["set_power_limit"] = {"ok": True, "target_w": W(target), "readback_w": W(readback),
                                  "power_under_cap_w": mean_power()}
        try: pynvml.nvmlDeviceSetPowerManagementLimit(h, pynvml.nvmlDeviceGetPowerManagementDefaultLimit(h))
        except Exception: pass
    except Exception as e:
        out["set_power_limit"] = {"ok": False, "error": str(e)[:200]}

    # --- TEST 2: locked clocks fallback (DVFS) ---
    try:
        try:
            maxc = pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM)
        except Exception:
            maxc = 1000
        lock = int(0.5 * maxc)
        pynvml.nvmlDeviceSetGpuLockedClocks(h, 0, lock)
        busy(2.0)
        out["set_locked_clocks"] = {"ok": True, "locked_sm_mhz": lock,
                                    "power_under_lock_w": mean_power(),
                                    "sm_clock_readback_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)}
        try: pynvml.nvmlDeviceResetGpuLockedClocks(h)
        except Exception: pass
    except Exception as e:
        out["set_locked_clocks"] = {"ok": False, "error": str(e)[:200]}

    return out


@app.local_entrypoint()
def main():
    result = probe.remote()
    print("PROBE_RESULT_JSON " + json.dumps(result, indent=2))
