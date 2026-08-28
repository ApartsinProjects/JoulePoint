# -*- coding: utf-8 -*-
"""Harvest MLPerf Inference power submissions into a load x machine matrix."""
import io, json, os, sys, time, urllib.request
from collections import defaultdict

API = "https://api.github.com/repos/mlcommons/cm4mlperf-results/contents/experiment"
UA = {"User-Agent": "research-fetch"}
OUT = "data/mlperf-power"


def get(url):
    for attempt in range(4):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60))
        except Exception as e:
            if attempt == 3:
                print(f"  fail {url}: {e}")
                return None
            time.sleep(2 * (attempt + 1))


def main():
    os.makedirs(OUT, exist_ok=True)
    top = get(API)
    if top is None:
        sys.exit("cannot list experiment dir")
    power_dirs = [x["name"] for x in top if x["type"] == "dir" and "power" in x["name"]]
    print(f"{len(power_dirs)} power experiment directories")

    records = []
    for i, dname in enumerate(power_dirs, 1):
        sub = get(f"{API}/{dname}")
        if not sub:
            continue
        for run in [x for x in sub if x["type"] == "dir"]:
            files = get(f"{API}/{dname}/{run['name']}")
            if not files:
                continue
            for f in files:
                if f["name"] == "cm-result.json":
                    try:
                        raw = urllib.request.urlopen(
                            urllib.request.Request(f["download_url"], headers=UA), timeout=90).read()
                        recs = json.loads(raw)
                        for r in recs:
                            r["_experiment"] = dname
                        records.extend(recs)
                    except Exception as e:
                        print(f"  skip {dname}/{run['name']}: {e}")
        print(f"  [{i}/{len(power_dirs)}] {dname}: running total {len(records)}")

    json.dump(records, io.open(f"{OUT}/mlperf_power_records.json", "w", encoding="utf-8"))
    print(f"\n{len(records)} records saved")

    # what fields carry power?
    keys = defaultdict(int)
    for r in records:
        for k in r:
            keys[k] += 1
    pk = sorted(k for k in keys if any(t in k.lower() for t in ("power", "watt", "energy", "joule")))
    print("power-ish fields:", pk)

    acc = defaultdict(int); mod = defaultdict(int)
    for r in records:
        acc[r.get("accelerator_model_name")] += 1
        mod[r.get("MlperfModel")] += 1
    print(f"\ndistinct accelerators: {len(acc)}")
    print(f"distinct models: {len(mod)}")
    print("top accelerators:", sorted(acc.items(), key=lambda x: -x[1])[:12])
    print("models:", sorted(mod.items(), key=lambda x: -x[1]))


if __name__ == "__main__":
    main()
