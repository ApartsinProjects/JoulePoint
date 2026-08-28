# -*- coding: utf-8 -*-
"""
Hand-curated hardware descriptors for the accelerators appearing in MLPerf power
submissions. These are the side features that let a model predict for an
accelerator it has never observed (machine cold start).

Values are nominal vendor specifications. Where a figure is not something I can
state with confidence it is left as None, and the downstream model treats it as
missing rather than guessing. `confident` marks rows whose core specs (TDP,
memory) I am confident in; the experiment reports results with and without the
non-confident rows so the reader can see whether they carry the result.
"""

# name -> dict(vendor, klass, tdp_w, mem_gb, bw_gbs, year, form, confident)
SPECS = {
    # ---- NVIDIA datacenter ----
    "NVIDIA H200-SXM-141GB":      dict(vendor="NVIDIA", klass="datacenter", tdp_w=700, mem_gb=141, bw_gbs=4800, year=2023, form="SXM",  confident=True),
    "NVIDIA H100-SXM-80GB":       dict(vendor="NVIDIA", klass="datacenter", tdp_w=700, mem_gb=80,  bw_gbs=3350, year=2022, form="SXM",  confident=True),
    "NVIDIA H100-NVL-94GB":       dict(vendor="NVIDIA", klass="datacenter", tdp_w=400, mem_gb=94,  bw_gbs=3900, year=2023, form="NVL",  confident=True),
    "NVIDIA H100-PCIe-80GB":      dict(vendor="NVIDIA", klass="datacenter", tdp_w=350, mem_gb=80,  bw_gbs=2000, year=2022, form="PCIe", confident=True),
    "NVIDIA A100-SXM-80GB":       dict(vendor="NVIDIA", klass="datacenter", tdp_w=400, mem_gb=80,  bw_gbs=2039, year=2020, form="SXM",  confident=True),
    "NVIDIA A100-SXM-80GB CTS":   dict(vendor="NVIDIA", klass="datacenter", tdp_w=400, mem_gb=80,  bw_gbs=2039, year=2020, form="SXM",  confident=True),
    "NVIDIA A100-PCIe-80GB":      dict(vendor="NVIDIA", klass="datacenter", tdp_w=300, mem_gb=80,  bw_gbs=1935, year=2020, form="PCIe", confident=True),
    "NVIDIA A100-SXM4-40GB":      dict(vendor="NVIDIA", klass="datacenter", tdp_w=400, mem_gb=40,  bw_gbs=1555, year=2020, form="SXM",  confident=True),
    "NVIDIA A30":                 dict(vendor="NVIDIA", klass="datacenter", tdp_w=165, mem_gb=24,  bw_gbs=933,  year=2021, form="PCIe", confident=True),
    "NVIDIA A2":                  dict(vendor="NVIDIA", klass="datacenter", tdp_w=60,  mem_gb=16,  bw_gbs=200,  year=2021, form="PCIe", confident=True),
    "NVIDIA L4":                  dict(vendor="NVIDIA", klass="datacenter", tdp_w=72,  mem_gb=24,  bw_gbs=300,  year=2023, form="PCIe", confident=True),
    "NVIDIA L40S":                dict(vendor="NVIDIA", klass="datacenter", tdp_w=350, mem_gb=48,  bw_gbs=864,  year=2023, form="PCIe", confident=True),
    # ---- NVIDIA consumer ----
    "NVIDIA GeForce RTX 4090 (Ada Lovelace)": dict(vendor="NVIDIA", klass="consumer", tdp_w=450, mem_gb=24, bw_gbs=1008, year=2022, form="PCIe", confident=True),
    # ---- NVIDIA edge / embedded ----
    "NVIDIA Jetson AGX Orin 64G": dict(vendor="NVIDIA", klass="edge", tdp_w=60, mem_gb=64, bw_gbs=205, year=2022, form="SoC", confident=True),
    "NVIDIA Jetson AGX Orin 32G": dict(vendor="NVIDIA", klass="edge", tdp_w=40, mem_gb=32, bw_gbs=205, year=2022, form="SoC", confident=True),
    "NVIDIA Jetson AGX Orin":     dict(vendor="NVIDIA", klass="edge", tdp_w=60, mem_gb=32, bw_gbs=205, year=2022, form="SoC", confident=True),
    "NVIDIA Orin NX 16G":         dict(vendor="NVIDIA", klass="edge", tdp_w=25, mem_gb=16, bw_gbs=102, year=2023, form="SoC", confident=True),
    "NVIDIA Orin":                dict(vendor="NVIDIA", klass="edge", tdp_w=40, mem_gb=32, bw_gbs=205, year=2022, form="SoC", confident=True),
    "NVIDIA Xavier":              dict(vendor="NVIDIA", klass="edge", tdp_w=30, mem_gb=32, bw_gbs=137, year=2018, form="SoC", confident=True),
    "NVIDIA Xavier NX":           dict(vendor="NVIDIA", klass="edge", tdp_w=15, mem_gb=8,  bw_gbs=51,  year=2019, form="SoC", confident=True),
    # ---- Qualcomm Cloud AI 100 family ----
    "QUALCOMM Cloud AI 100 Ultra":              dict(vendor="QUALCOMM", klass="datacenter", tdp_w=150, mem_gb=128, bw_gbs=None, year=2023, form="PCIe", confident=True),
    "QUALCOMM Cloud AI 100 PCIe/HHHL Pro":      dict(vendor="QUALCOMM", klass="datacenter", tdp_w=75,  mem_gb=32,  bw_gbs=None, year=2020, form="PCIe", confident=True),
    "QUALCOMM Cloud AI 100 PCIe/HHHL Standard": dict(vendor="QUALCOMM", klass="datacenter", tdp_w=75,  mem_gb=16,  bw_gbs=None, year=2020, form="PCIe", confident=True),
    "QUALCOMM Cloud AI 100 PCIe/HHHL Lite":     dict(vendor="QUALCOMM", klass="datacenter", tdp_w=75,  mem_gb=16,  bw_gbs=None, year=2020, form="PCIe", confident=False),
    "QUALCOMM Cloud AI 100 DM.2":               dict(vendor="QUALCOMM", klass="edge", tdp_w=25, mem_gb=16, bw_gbs=None, year=2021, form="M.2", confident=True),
    "QUALCOMM Cloud AI 100 DM.2e":              dict(vendor="QUALCOMM", klass="edge", tdp_w=15, mem_gb=16, bw_gbs=None, year=2021, form="M.2", confident=True),
    # ---- mobile SoC blocks: specs not reliably comparable, left mostly missing ----
    "QUALCOMM Hexagon 698 DSP":   dict(vendor="QUALCOMM", klass="mobile", tdp_w=None, mem_gb=None, bw_gbs=None, year=2020, form="SoC", confident=False),
    "QUALCOMM Adreno 650 GPU":    dict(vendor="QUALCOMM", klass="mobile", tdp_w=None, mem_gb=None, bw_gbs=None, year=2020, form="SoC", confident=False),
    "QUALCOMM NPU 230 AIP":       dict(vendor="QUALCOMM", klass="mobile", tdp_w=None, mem_gb=None, bw_gbs=None, year=None, form="SoC", confident=False),
    "ARM Mali G52MP8(8EE)":       dict(vendor="ARM", klass="mobile", tdp_w=None, mem_gb=None, bw_gbs=None, year=2018, form="SoC", confident=False),
    "ARM Mali-G610 MP4":          dict(vendor="ARM", klass="mobile", tdp_w=None, mem_gb=None, bw_gbs=None, year=2021, form="SoC", confident=False),
    "Arm Mali-G52 MP6 GPU":       dict(vendor="ARM", klass="mobile", tdp_w=None, mem_gb=None, bw_gbs=None, year=2018, form="SoC", confident=False),
    # ---- other vendors ----
    "DaVinci":                    dict(vendor="HUAWEI",  klass="edge",       tdp_w=None, mem_gb=None, bw_gbs=None, year=None, form=None,  confident=False),
    "RecAccel N3000":             dict(vendor="NEUCHIPS", klass="datacenter", tdp_w=None, mem_gb=None, bw_gbs=None, year=2022, form="PCIe", confident=False),
    "MLSOC DualM.2":              dict(vendor="SIMA",    klass="edge",       tdp_w=None, mem_gb=None, bw_gbs=None, year=2023, form="M.2", confident=False),
    "NPU 3.0":                    dict(vendor="OTHER",   klass="edge",       tdp_w=None, mem_gb=None, bw_gbs=None, year=None, form=None,  confident=False),
}
