# -*- coding: utf-8 -*-
"""
Provenance audit: can every number in the paper be pointed at, and does every result file carry
the per-datapoint data it should?

Two failures this session motivate it. Three of the four headline lever figures matched no
numeric leaf in any result file: they were literals in a figure script. And
interaction_matrix.json has correct values but no producing script, so its numbers cannot be
regenerated.

This checks three things per result file:
  A. does a script exist that writes it (provenance)
  B. does it carry per-datapoint records, or only summaries (reproducibility)
  C. does it carry a sanity block, and did those checks pass (self-report)

And one thing per paper number: does a numeric leaf within tolerance exist somewhere.
"""
import io, json, os, re, glob, html
from collections import defaultdict

RES = "experiments/results"
SCRIPTS = glob.glob("experiments/*.py")
src = {}
for f in SCRIPTS:
    try:
        src[f] = io.open(f, encoding="utf-8", errors="replace").read()
    except Exception:
        pass

# ------------------------------------------------------------------ A. producers
producers = defaultdict(list)
for f, t in src.items():
    for m in re.finditer(r'results/([A-Za-z0-9_\-]+\.json)', t):
        producers[m.group(1)].append(os.path.basename(f))

files = sorted(os.path.basename(p) for p in glob.glob(RES + "/*.json"))
orphans = [f for f in files if not producers.get(f)]

# ------------------------------------------------------------------ B/C. content
def leaves(o, path="", out=None, depth=0):
    if out is None:
        out = []
    if depth > 14:
        return out
    if isinstance(o, dict):
        for k, v in o.items():
            leaves(v, path + "/" + str(k), out, depth + 1)
    elif isinstance(o, list):
        for i, v in enumerate(o[:600]):
            leaves(v, path + "[%d]" % i, out, depth + 1)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        out.append((path, float(o)))
    return out

PERDP = ("per_datapoint", "per_seed", "per_cell", "per_row", "per_fold", "rows", "runs",
         "cell_values", "per_composition", "sweep", "residuals", "per_round", "raw")
report = []
allleaves = {}
for f in files:
    try:
        d = json.load(io.open(os.path.join(RES, f), encoding="utf-8"))
    except Exception as e:
        report.append((f, "UNREADABLE", str(e)[:40], "", ""))
        continue
    lv = leaves(d)
    allleaves[f] = lv
    txt = json.dumps(d)[:400000]
    has_dp = any(k in txt for k in PERDP)
    san = d.get("sanity") if isinstance(d, dict) else None
    if isinstance(san, list) and san:
        npass = sum(1 for c in san if c.get("passed"))
        sstat = "{}/{}".format(npass, len(san))
    else:
        sstat = "none"
    report.append((f, "ok" if producers.get(f) else "ORPHAN",
                   producers.get(f, [""])[0][:28], "yes" if has_dp else "NO", sstat))

print("{:<34}{:<9}{:<30}{:<8}{}".format("result file", "producer", "written by", "per-dp", "sanity"))
print("-" * 96)
for r in sorted(report, key=lambda x: (x[1] != "ORPHAN", x[3] != "NO", x[0])):
    print("{:<34}{:<9}{:<30}{:<8}{}".format(r[0], r[1], r[2], r[3], r[4]))

print()
print("files: {}   orphans (no producing script): {}   no per-datapoint block: {}".format(
    len(files), sum(1 for r in report if r[1] == "ORPHAN"), sum(1 for r in report if r[3] == "NO")))
print("sanity blocks present: {} of {}".format(
    sum(1 for r in report if r[4] != "none"), len(files)))

# ------------------------------------------------------------------ paper numbers
paper = io.open("paper/greenmatch-paper.html", encoding="utf-8").read()
paper = re.sub(r"<svg.*?</svg>", " ", paper, flags=re.S)
txt = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", paper)))
body = txt[txt.index("1. Introduction"):txt.index("References")]
nums = sorted({float(m) for m in re.findall(r"\b(\d{1,3}\.\d{1,2})\b", body)
               if 0.05 <= float(m) <= 200})
print("\ndistinct numeric claims in the body: {}".format(len(nums)))
missing = []
for v in nums:
    tol = max(0.005, abs(v) * 0.002)
    hit = None
    for f, lv in allleaves.items():
        for p, x in lv:
            if abs(x - v) <= tol:
                hit = f; break
        if hit:
            break
    if not hit:
        missing.append(v)
print("numbers with NO matching leaf in any result file: {} of {}".format(len(missing), len(nums)))
print("  " + ", ".join("{:g}".format(v) for v in missing[:60]))

json.dump(dict(files=len(files),
               orphans=[r[0] for r in report if r[1] == "ORPHAN"],
               no_per_datapoint=[r[0] for r in report if r[3] == "NO"],
               no_sanity=[r[0] for r in report if r[4] == "none"],
               paper_numbers=len(nums), unmatched=missing,
               table=[dict(file=r[0], producer=r[1], by=r[2], per_dp=r[3], sanity=r[4])
                      for r in report]),
          io.open("experiments/results/provenance_audit.json", "w", encoding="utf-8"), indent=1)
print("\nsaved -> experiments/results/provenance_audit.json")
