#!/bin/bash
set -e
pip install -q nvidia-ml-py==12.560.30 2>/dev/null || pip install -q nvidia-ml-py
python3 -u runpod_powercap_probe.py
