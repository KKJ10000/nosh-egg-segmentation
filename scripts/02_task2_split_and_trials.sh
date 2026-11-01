#!/usr/bin/env bash
# Task 2: leakage-aware split + the experiment trials (each appends a row to results/experiment_log.csv).
set -euo pipefail
cd "$(dirname "$0")/../src"
python split.py
rm -f ../results/experiment_log.csv
bash ../scripts/trials.sh
