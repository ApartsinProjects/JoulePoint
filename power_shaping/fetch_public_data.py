# -*- coding: utf-8 -*-
"""Fetch the public datasets for the power-shaping PoC into data/raw/ (gitignored).
Run once before the experiments. All sources are public and re-downloadable.

  emerald DVFS + power  : ai-emerald/emerald-ai-demo-may-2025 (Nature Energy 2025)  ~150 KB
  Azure LLM Inference   : Azure/AzurePublicDataset  dataset-llm-2024 (conv 1-week)  ~1.1 GB
"""
import os, urllib.request

RAW = os.path.join(os.path.dirname(__file__), "data", "raw")
os.makedirs(RAW, exist_ok=True)

EMERALD = "https://raw.githubusercontent.com/ai-emerald/emerald-ai-demo-may-2025/main/data"
EMERALD_FILES = ["dvfs_sweep.csv", "SRP_total_power.csv", "SRP_exp_performance.csv",
                 "CAISOexp_totalpower_cleaned.csv", "CAISO-netdemand-cleaned.csv"]
AZURE = ("https://github.com/Azure/AzurePublicDataset/releases/download/"
         "dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv")


def get(url, dst):
    if os.path.exists(dst):
        print(f"  have {os.path.basename(dst)}"); return
    print(f"  downloading {os.path.basename(dst)} ...")
    urllib.request.urlretrieve(url, dst)
    print(f"    -> {os.path.getsize(dst)} bytes")


def main():
    print("emerald DVFS + power:")
    for f in EMERALD_FILES:
        get(f"{EMERALD}/{f}", os.path.join(RAW, f))
    print("Azure LLM inference trace (~1.1 GB):")
    get(AZURE, os.path.join(RAW, "azure_llm_conv.csv"))
    print("done. Next: python prep_azure.py")


if __name__ == "__main__":
    main()
