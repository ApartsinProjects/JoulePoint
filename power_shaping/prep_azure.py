# -*- coding: utf-8 -*-
"""Profile the Azure LLM conv trace (27.3M requests, 1 week) and extract a busy window
at full arrival fidelity for the PoC-B replay. Reads in chunks to stay in memory."""
import os, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "data", "raw")
SRC = os.path.join(RAW, "azure_llm_conv.csv")
OUT_WINDOW = os.path.join(RAW, "azure_window.parquet")
OUT_PROFILE = os.path.join(HERE, "results", "azure_profile.json")

CHUNK = 2_000_000


def main():
    t0 = None
    per_min_counts = {}      # minute-of-week -> count
    tok_ctx_sum = 0; tok_gen_sum = 0; n = 0
    ctx_samples = []; gen_samples = []
    for ch in pd.read_csv(SRC, chunksize=CHUNK,
                          usecols=["TIMESTAMP", "ContextTokens", "GeneratedTokens"]):
        ts = pd.to_datetime(ch["TIMESTAMP"], utc=True, format="ISO8601")
        if t0 is None:
            t0 = ts.iloc[0]
        minute = ((ts - t0).dt.total_seconds() // 60).astype(int)
        vc = minute.value_counts()
        for m, c in vc.items():
            per_min_counts[int(m)] = per_min_counts.get(int(m), 0) + int(c)
        tok_ctx_sum += int(ch["ContextTokens"].sum())
        tok_gen_sum += int(ch["GeneratedTokens"].sum())
        n += len(ch)
        if len(ctx_samples) < 200000:
            ctx_samples.extend(ch["ContextTokens"].sample(min(20000, len(ch))).tolist())
            gen_samples.extend(ch["GeneratedTokens"].sample(min(20000, len(ch))).tolist())

    minutes = np.array(sorted(per_min_counts))
    counts = np.array([per_min_counts[m] for m in minutes])
    # find the busiest contiguous 60-min window
    rate = counts.astype(float)            # requests per minute
    if len(rate) >= 60:
        win = np.convolve(rate, np.ones(60), "valid")
        busy_start = int(minutes[np.argmax(win)])
    else:
        busy_start = int(minutes[0])
    busy_end = busy_start + 60

    profile = {
        "n_requests": n,
        "duration_hours": float((minutes.max() - minutes.min()) / 60.0),
        "mean_rate_per_s": n / ((minutes.max() - minutes.min()) * 60.0),
        "peak_rate_per_s": float(counts.max() / 60.0),
        "ctx_tokens": {"mean": np.mean(ctx_samples), "p50": float(np.percentile(ctx_samples, 50)),
                       "p95": float(np.percentile(ctx_samples, 95))},
        "gen_tokens": {"mean": np.mean(gen_samples), "p50": float(np.percentile(gen_samples, 50)),
                       "p95": float(np.percentile(gen_samples, 95))},
        "busy_window_min": [busy_start, busy_end],
    }
    with open(OUT_PROFILE, "w") as f:
        json.dump(profile, f, indent=2)
    print("profile:", json.dumps(profile, indent=2))

    # second pass: extract the busy window at full fidelity
    rows = []
    for ch in pd.read_csv(SRC, chunksize=CHUNK,
                          usecols=["TIMESTAMP", "ContextTokens", "GeneratedTokens"]):
        ts = pd.to_datetime(ch["TIMESTAMP"], utc=True, format="ISO8601")
        rel_min = (ts - t0).dt.total_seconds() / 60.0
        mask = (rel_min >= busy_start) & (rel_min < busy_end)
        if mask.any():
            sub = ch[mask].copy()
            sub["t_s"] = (ts[mask] - t0).dt.total_seconds() - busy_start * 60.0
            rows.append(sub[["t_s", "ContextTokens", "GeneratedTokens"]])
        if rel_min.min() > busy_end:
            break
    win_df = pd.concat(rows).sort_values("t_s").reset_index(drop=True)
    win_df.to_parquet(OUT_WINDOW)
    print(f"\nbusy window: {len(win_df)} requests over 60 min "
          f"({len(win_df)/3600:.1f} req/s avg) -> {OUT_WINDOW}")


if __name__ == "__main__":
    main()
