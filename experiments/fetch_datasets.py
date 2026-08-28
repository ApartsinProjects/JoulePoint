# -*- coding: utf-8 -*-
"""
Fetch the public datasets the GreenMatch-AI work plan (WP1) depends on.

Resumable: each file is downloaded to <name>.part and renamed on completion, so
re-running skips anything already present and continues anything interrupted.
"""
import io, os, sys, time, json, urllib.request

DEST = "data/butter-e"
BASE = "https://data.openei.org/files/5991/"
FILES = [
    # (name, why it matters for this project)
    ("runs_with_standardized_energy.csv.zip", "power joined to run data with standardized energy: the observation matrix itself"),
    ("butter_e_metadata.csv.zip",             "per-run metadata: the load (workload) descriptors"),
    ("node_sinfo.csv",                        "per-node characteristics: the machine (hardware) descriptors"),
    ("node_power_dist.csv",                   "per-node power quantiles: idle/baseline power for standardisation"),
    ("butter_e_energy.zip",                   "1-minute raw power time series per run: peak-power and power-profile ground truth"),
    ("summary_by_epoch.tar",                  "per-epoch training losses re-summarised from BUTTER (large, lowest priority)"),
]
UA = {"User-Agent": "Mozilla/5.0 (research dataset fetch)"}


def human(n):
    return f"{n/1048576:.1f} MB"


def fetch(name):
    dst = os.path.join(DEST, name)
    part = dst + ".part"
    if os.path.exists(dst):
        print(f"[skip] {name} already present ({human(os.path.getsize(dst))})", flush=True)
        return True
    have = os.path.getsize(part) if os.path.exists(part) else 0
    headers = dict(UA)
    if have:
        headers["Range"] = f"bytes={have}-"
        print(f"[resume] {name} from {human(have)}", flush=True)
    req = urllib.request.Request(BASE + name, headers=headers)
    try:
        r = urllib.request.urlopen(req, timeout=120)
    except Exception as e:
        print(f"[FAIL] {name}: {e}", flush=True)
        return False
    total = int(r.headers.get("Content-Length") or 0) + have
    mode = "ab" if have and r.status == 206 else "wb"
    if mode == "wb":
        have = 0
    t0, last = time.time(), 0
    with io.open(part, mode) as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            have += len(chunk)
            if time.time() - last > 5:
                pct = f"{100*have/total:.0f}%" if total else "?"
                rate = have / max(time.time() - t0, 1e-6) / 1048576
                print(f"    {name}: {human(have)} / {human(total)} ({pct}) {rate:.1f} MB/s", flush=True)
                last = time.time()
    os.replace(part, dst)
    print(f"[done] {name} {human(os.path.getsize(dst))} in {time.time()-t0:.0f}s", flush=True)
    return True


def main():
    os.makedirs(DEST, exist_ok=True)
    manifest = []
    ok = 0
    for name, why in FILES:
        print(f"\n=== {name} ===\n    {why}", flush=True)
        if fetch(name):
            ok += 1
            p = os.path.join(DEST, name)
            manifest.append({"file": name, "purpose": why,
                             "bytes": os.path.getsize(p) if os.path.exists(p) else None,
                             "source": BASE + name})
    json.dump({"dataset": "BUTTER-E", "doi": "10.25984/2329316",
               "landing": "https://data.openei.org/submissions/5991",
               "retrieved": "2026-08-18", "files": manifest},
              io.open(os.path.join(DEST, "MANIFEST.json"), "w", encoding="utf-8"), indent=1)
    print(f"\n{ok}/{len(FILES)} files retrieved into {DEST}", flush=True)
    return 0 if ok == len(FILES) else 1


if __name__ == "__main__":
    sys.exit(main())
