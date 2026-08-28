# -*- coding: utf-8 -*-
"""
Power-cap permission probe on RunPod (root pod). Tests whether we can set the GPU power
limit (nvidia-smi -pl / NVML) and/or lock clocks, and whether power actually drops under a
tiny load. Decides whether the power-cap elasticity sweep can run on RunPod.
Writes results/powercap_probe.json.
"""
import sys, os, json, time, subprocess

print("[train] power-cap permission probe starting"); sys.stdout.flush()
import torch
assert torch.cuda.is_available(), "CUDA not available."
print(f"[train] GPU: {torch.cuda.get_device_name(0)}"); sys.stdout.flush()

os.makedirs("results", exist_ok=True)
out = {"gpu": torch.cuda.get_device_name(0)}


def sh(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


import pynvml
pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)

def W(mw): return round(mw / 1000.0, 1)

try:
    out["default_limit_w"] = W(pynvml.nvmlDeviceGetPowerManagementLimit(h))
    lo, hi = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)
    out["limit_constraints_w"] = [W(lo), W(hi)]
except Exception as e:
    out["limit_read_error"] = str(e)[:200]

def busy(sec):
    a = torch.randn(4096, 4096, device="cuda"); b = torch.randn(4096, 4096, device="cuda")
    t0 = time.time()
    while time.time() - t0 < sec:
        a = (a @ b).relu()
    torch.cuda.synchronize()

def mean_power(sec=2.0):
    s = []; t0 = time.time()
    while time.time() - t0 < sec:
        s.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0); time.sleep(0.05)
    return round(sum(s) / len(s), 1)

print("[train] measuring default power..."); sys.stdout.flush()
busy(2); out["power_at_default_w"] = mean_power()

# enable persistence mode (often required before -pl)
out["persistence_cmd"] = sh("nvidia-smi -pm 1")[1][:150]

# TEST 1: nvidia-smi -pl to ~65% of default
try:
    default = pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0
    lo = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)[0] / 1000.0
    target = int(max(lo, 0.65 * default))
    rc, msg = sh(f"nvidia-smi -pl {target}")
    busy(2)
    out["nvidia_smi_pl"] = {"rc": rc, "msg": msg[:200], "target_w": target,
                            "enforced_after_w": W(pynvml.nvmlDeviceGetEnforcedPowerLimit(h)),
                            "power_under_cap_w": mean_power()}
except Exception as e:
    out["nvidia_smi_pl"] = {"error": str(e)[:200]}

# TEST 2: NVML set power limit
try:
    default_mw = pynvml.nvmlDeviceGetPowerManagementLimit(h)
    lo_mw = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)[0]
    tgt = int(max(lo_mw, 0.6 * default_mw))
    pynvml.nvmlDeviceSetPowerManagementLimit(h, tgt)
    busy(2)
    out["nvml_set_power_limit"] = {"ok": True, "target_w": W(tgt),
                                   "power_under_cap_w": mean_power()}
    try: pynvml.nvmlDeviceSetPowerManagementLimit(h, pynvml.nvmlDeviceGetPowerManagementDefaultLimit(h))
    except Exception: pass
except Exception as e:
    out["nvml_set_power_limit"] = {"ok": False, "error": str(e)[:200]}

# TEST 3: locked clocks
try:
    maxc = pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM)
    lock = int(0.5 * maxc)
    pynvml.nvmlDeviceSetGpuLockedClocks(h, 0, lock)
    busy(2)
    out["nvml_locked_clocks"] = {"ok": True, "locked_sm_mhz": lock,
                                 "sm_readback_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
                                 "power_under_lock_w": mean_power()}
    try: pynvml.nvmlDeviceResetGpuLockedClocks(h)
    except Exception: pass
except Exception as e:
    out["nvml_locked_clocks"] = {"ok": False, "error": str(e)[:200]}

with open("results/powercap_probe.json", "w") as f:
    json.dump(out, f, indent=2)
print("[train] PROBE_RESULT " + json.dumps(out)); sys.stdout.flush()
print("[train] === DONE ==="); sys.stdout.flush()
