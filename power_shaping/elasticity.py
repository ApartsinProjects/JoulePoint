# -*- coding: utf-8 -*-
"""
Power-elasticity response curves for AI workloads.

An elasticity curve maps a GPU power-cap setting u in {60..100}% of TDP to a pair
(relative_power, relative_throughput). The workload-shaping question turns entirely
on whether these curves DIFFER across workloads: if every workload has the same
normalized curve, a workload-aware controller cannot beat a uniform cap.

DATA PROVENANCE
---------------
The spec names a Phoenix/Emerald `dvfs_sweep.csv`. If a real sweep file is present
at power_shaping/data/raw/dvfs_sweep.csv it is loaded verbatim (function
`load_real_curves`). When no measured sweep is available we fall back to a
PHYSICALLY-GROUNDED SURROGATE (`synthetic_curves`) whose heterogeneity comes from a
single real, published distinction:

  * compute-bound work (LLM prefill, dense training, CV): throughput tracks core
    clock, which a power cap lowers -> capping costs throughput (low elasticity).
  * memory-bound work (LLM decode, embedding, recommendation): throughput is limited
    by memory bandwidth, which a core-power cap barely touches -> large power cut,
    small throughput loss (high elasticity).

This compute-vs-memory-bound split is a measured property of real GPUs (roofline),
not an assumption we invented to make the method win. The surrogate is deterministic
(fixed rng) and every downstream result that depends on it is reported as
SURROGATE-CONTINGENT until a real sweep is substituted. The kill test additionally
runs a HOMOGENEOUS-POOL invariant (all curves identical) that MUST yield zero oracle
headroom; if it does not, the code is inventing headroom and the result is void.
"""
from __future__ import annotations
import os, json
import numpy as np

RAW = os.path.join(os.path.dirname(__file__), "data", "raw")

# Power-cap settings tested, as a fraction of the workload's default power limit.
CAP_SETTINGS = np.array([1.00, 0.90, 0.80, 0.70, 0.60])

# Workload archetypes with a compute-bound fraction beta in [0,1].
# beta=1 fully compute-bound (clock-limited), beta=0 fully memory-bound (bw-limited).
# Proportions and beta values are calibrated to the qualitative behaviour reported
# for these workload families in the GPU-serving literature; they are surrogate
# parameters, not measurements.
ARCHETYPES = {
    "llm_prefill":       dict(beta=0.90, weight_class="online"),
    "llm_decode":        dict(beta=0.20, weight_class="online"),
    "llm_serving_mixed": dict(beta=0.55, weight_class="online"),
    "llm_training":      dict(beta=0.95, weight_class="training"),
    "vision_training":   dict(beta=0.88, weight_class="training"),
    "embedding":         dict(beta=0.15, weight_class="offline"),
    "batch_inference":   dict(beta=0.45, weight_class="offline"),
    "recommendation":    dict(beta=0.25, weight_class="offline"),
    "dev_interactive":   dict(beta=0.60, weight_class="dev"),
}


def _throughput_at_cap(beta: float, u: np.ndarray) -> np.ndarray:
    """Relative throughput q(u) in (0,1], q(1)=1.

    Compute-bound component: at power cap u the core clock scales roughly as u^0.5
    (power ~ f^2..f^3 in the DVFS regime), and compute-bound throughput tracks clock,
    so its relative throughput ~ u^0.5 in the capped range, saturating near u=1.
    Memory-bound component: throughput is set by memory bandwidth, unaffected until
    the cap starves the memory subsystem (below ~0.6 TDP), modelled as a gentle floor.
    A workload is a beta-mix of the two.
    """
    u = np.asarray(u, dtype=float)
    compute_q = np.clip(u, 0.0, 1.0) ** 0.5           # clock-limited
    # memory-bound: near-flat until deep caps, then a mild knee
    mem_q = np.clip(1.0 - 0.25 * np.maximum(0.0, 0.75 - u), 0.0, 1.0)
    q = beta * compute_q + (1.0 - beta) * mem_q
    return q / q[0]  # normalise so q at u=1 (first entry) is exactly 1


def _power_at_cap(u: np.ndarray, natural_draw: float = 0.92) -> np.ndarray:
    """Relative power p(u), p(1)=1. A workload naturally draws `natural_draw` of TDP;
    a cap below that pulls draw down to the cap, above it the cap is not binding."""
    u = np.asarray(u, dtype=float)
    p = np.minimum(u, natural_draw)
    return p / p[0]


def synthetic_curves(seed: int = 0):
    """Return list of workload dicts, each with power[] and throughput[] over CAP_SETTINGS.
    Deterministic given seed. Each archetype gets small heterogeneous jitter in beta and
    natural draw so that within-class variation exists but the class signal dominates."""
    rng = np.random.default_rng(seed)
    curves = []
    for name, meta in ARCHETYPES.items():
        beta0 = meta["beta"]
        # a few instances per archetype with jittered parameters
        for k in range(3):
            beta = float(np.clip(beta0 + rng.normal(0, 0.05), 0.02, 0.99))
            draw = float(np.clip(rng.normal(0.92, 0.03), 0.80, 0.99))
            q = _throughput_at_cap(beta, CAP_SETTINGS)
            p = _power_at_cap(CAP_SETTINGS, draw)
            curves.append(dict(
                workload_type=name, instance=k, beta=beta, natural_draw=draw,
                weight_class=meta["weight_class"],
                power=p.tolist(), throughput=q.tolist(),
                source="surrogate",
            ))
    return curves


def load_real_curves(path: str | None = None):
    """Load a real DVFS sweep if present. Expected long format with columns:
    workload_type, power_setting (fraction of default), power (W or relative),
    throughput (any consistent unit). Returns None if the file is absent."""
    path = path or os.path.join(RAW, "dvfs_sweep.csv")
    if not os.path.exists(path):
        return None
    import pandas as pd
    df = pd.read_csv(path)
    curves = []
    for wl, g in df.groupby("workload_type"):
        g = g.sort_values("power_setting", ascending=False)
        p = g["power"].to_numpy(float); q = g["throughput"].to_numpy(float)
        curves.append(dict(
            workload_type=str(wl), instance=0, weight_class="online",
            power=(p / p[0]).tolist(), throughput=(q / q[0]).tolist(),
            source="real:dvfs_sweep.csv",
        ))
    return curves


def get_curves(seed: int = 0):
    """Real curves if a sweep file exists, else the physically-grounded surrogate.
    Returns (curves, provenance_string)."""
    real = load_real_curves()
    if real:
        return real, "real:dvfs_sweep.csv"
    return synthetic_curves(seed), "surrogate:compute-vs-memory-bound"


if __name__ == "__main__":
    curves, prov = get_curves()
    print(f"provenance: {prov}; {len(curves)} workload curves")
    # quick heterogeneity check: spread of throughput at the deepest cap
    q_at_60 = np.array([c["throughput"][-1] for c in curves])
    print(f"throughput at 60% cap: min={q_at_60.min():.3f} max={q_at_60.max():.3f} "
          f"spread={q_at_60.max()-q_at_60.min():.3f}")
