# -*- coding: utf-8 -*-
"""
Fetch the measured-energy artifacts from Wilkins, Keshav and Mortier,
"Hybrid Heterogeneous Clusters Can Lower the Energy Consumption of LLM Inference
Workloads" (arXiv:2407.00010), repository github.com/grantwilkins/energy-inference.

LICENCE: the repository carries NO licence file. `GET /repos/grantwilkins/
energy-inference/license` returns 404 and the root tree contains no LICENSE /
COPYING / licence section in README.md. Under GitHub's terms the default is
all-rights-reserved: the work is publicly readable but NOT licensed for
redistribution. We therefore download only the aggregate CSVs we analyse, keep
them out of any published artifact, and cite the paper rather than redistribute
its data. Derived summary statistics we compute ourselves are fine to publish.

Only the ~1 MB of aggregate stats CSVs is fetched. The full repository is ~500 MB
(raw nvidia-smi traces up to 58 MB per file), so it is deliberately not cloned.
"""
import hashlib, io, json, os, urllib.request

BASE = "https://raw.githubusercontent.com/grantwilkins/energy-inference/main"
OUT = "data/wilkins"
FILES = {
    # aggregate, per-inference-call energy records with a System column
    "plots/independence-test/all-input-stats.csv": "independence-test/all-input-stats.csv",
    "plots/independence-test/all-output-stats.csv": "independence-test/all-output-stats.csv",
    "plots/independence-test/all-stats.csv": "independence-test/all-stats.csv",
    # copies kept only so the duplicate-file audit can hash them
    "plots/heterogeneous-datacenter/e2dc-artifacts/all-input-stats.csv": "e2dc-artifacts/all-input-stats.csv",
    "plots/heterogeneous-datacenter/e2dc-artifacts/all-output-stats.csv": "e2dc-artifacts/all-output-stats.csv",
    "plots/input-test/all-input-stats.csv": "input-test/all-input-stats.csv",
    "README.md": "README.md",
}

if __name__ == "__main__":
    manifest = {}
    for src, dst in FILES.items():
        path = os.path.join(OUT, dst)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with urllib.request.urlopen(BASE + "/" + src.replace(" ", "%20")) as fh:
                open(path, "wb").write(fh.read())
        blob = open(path, "rb").read()
        manifest[dst] = dict(source=src, bytes=len(blob),
                             md5=hashlib.md5(blob).hexdigest())
        print("{:52} {:9d} B  {}".format(dst, len(blob), manifest[dst]["md5"]))
    json.dump(dict(base=BASE, licence="NONE (no LICENSE file; all rights reserved)",
                   files=manifest),
              io.open(os.path.join(OUT, "MANIFEST.json"), "w", encoding="utf-8"), indent=1)
    print("\nwrote", os.path.join(OUT, "MANIFEST.json"))
