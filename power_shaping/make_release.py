# -*- coding: utf-8 -*-
"""Assemble the ELF Zenodo release: copy the dataset CSVs and the AWS measurement scripts into elf_release/,
write the code README + MIT license, and zip it. Idempotent."""
import os, shutil, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.path.join(HERE, "elf_release")
DATA = os.path.join(REL, "data"); CODE = os.path.join(REL, "code")
os.makedirs(DATA, exist_ok=True); os.makedirs(CODE, exist_ok=True)

# raw per-sweep CSVs (the individual measurement files) live under data/raw
DATA_FILES = ["aws_sweep_spectrum.csv", "aws_sweep_a100.csv", "aws_sweep_batch.csv",
              "aws_sweep_a100_batch.csv", "aws_sweep_clock.csv", "aws_sweep_clock_a10g.csv",
              "reaction_time.csv"]
# the consolidated master table (headline artifact) + its dictionary live under data/processed
MASTER_FILES = ["elf_master.csv", "elf_master_dictionary.md"]
# the exact harnesses run on AWS to produce the data, plus the consolidation + verification code
CODE_FILES = ["aws_sweep_expand.py", "aws_powercap_probe.py", "aws_powercap_sweep.py",
              "aws_sweep_batch.py", "aws_sweep_a100_batch.py", "aws_sweep_a10g_shard.py",
              "merge_a10g_rep3.py", "aws_sweep_spectrum.py", "aws_sweep_clock.py",
              "reaction_time_probe.py", "monitor_aws_spend.py",
              "build_master_dataset.py", "verify_against_master.py"]

for fn in DATA_FILES:
    src = os.path.join(HERE, "data", "raw", fn)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(DATA, fn)); print("data +", fn)
    else:
        print("data MISSING", fn)
for fn in MASTER_FILES:
    src = os.path.join(HERE, "data", "processed", fn)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(DATA, fn)); print("master +", fn)
    else:
        print("master MISSING", fn)
# surface the dictionary at the release root too, as the "start here" file
if os.path.exists(os.path.join(DATA, "elf_master_dictionary.md")):
    shutil.copy2(os.path.join(DATA, "elf_master_dictionary.md"), os.path.join(REL, "elf_master_dictionary.md"))
for fn in CODE_FILES:
    src = os.path.join(HERE, fn)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(CODE, fn)); print("code +", fn)
    else:
        print("code MISSING", fn)

MIT = """MIT License

Copyright (c) 2026 Alexander Apartsin, Yehudit Aperstein

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without
limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
open(os.path.join(CODE, "LICENSE"), "w", encoding="utf-8").write(MIT)

CODE_README = """# ELF measurement code

These are the exact AWS harnesses used to produce the ELF dataset. Each launches a throwaway VPC and a single
GPU EC2 instance, sets power caps with `nvidia-smi -pl`, sweeps, exfiltrates results to S3 via a same-region
gateway endpoint, and tears everything down (self-cleaning).

- `aws_sweep_expand.py` - shared scaffold: bucket/endpoint setup, `run_one` launcher, user-data template.
- `aws_powercap_probe.py`, `aws_powercap_sweep.py` - probe + DLAMI helpers and per-run cleanup.
- `aws_sweep_spectrum.py` - g-card (T4/L4/A10G) cap sweep at batch 32 with the 4-phase latency split.
- `aws_sweep_batch.py` - g-card control (batch 32) + auto-calibrated saturate cap sweeps.
- `aws_sweep_a100_batch.py` - A100 (sharded 8-GPU p4d) control + saturate cap sweeps.
- `aws_sweep_a10g_shard.py` - A10G 3-rep sweep model-sharded across N single-GPU g5 instances (each does a
  disjoint slice of the 20 models), then merges the shard CSVs; keeps every run under the scaffold timers.
- `merge_a10g_rep3.py` - validates the sharded A10G 3-rep block and splices it into `aws_sweep_batch.csv`.
- `aws_sweep_clock.py` - clock/DVFS sweep (`nvidia-smi -lgc`) that reaches below the driver power-cap floor
  (T4/L4, and A10G with focused low-range sampling); produces `aws_sweep_clock*.csv`.
- `reaction_time_probe.py` - steps the cap on a live A10G and logs power/latency at ~50 Hz.
- `monitor_aws_spend.py` - cost watchdog / dead-man cleanup.
- `build_master_dataset.py` - consolidates every sweep above into the headline `data/elf_master.csv` (one row
  per operating point, both actuators and both load regimes, with provenance, saturation flags, canonical-subset
  flags, card specs, job features, and fitted response-law parameters). See `data/elf_master_dictionary.md`.
- `verify_against_master.py` - checks that the paper's headline figure/table numbers reproduce from the
  documented subset of the master table.

Requirements: an AWS account with GPU quota, `boto3`, and a recent Deep Learning AMI (PyTorch). Running these
incurs real charges. They set no secrets in the data; the account id in bucket names is cosmetic.

Reproduce, e.g.: `python aws_sweep_batch.py --parallel --region us-east-1`
"""
open(os.path.join(CODE, "README_code.md"), "w", encoding="utf-8").write(CODE_README)

# zip
zpath = os.path.join(HERE, "ELF_dataset.zip")
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(REL):
        for fn in files:
            fp = os.path.join(root, fn)
            z.write(fp, os.path.relpath(fp, os.path.dirname(REL)))
print(f"\nzipped -> {os.path.relpath(zpath, HERE)}  ({os.path.getsize(zpath)//1024} KB)")
