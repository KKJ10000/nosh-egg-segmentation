#!/usr/bin/env bash
# Task 3: evaluate results/final_config.json on dev + eval, write metrics CSV/JSON and figures.
set -euo pipefail
cd "$(dirname "$0")/../src"
python evaluate.py
