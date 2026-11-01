#!/usr/bin/env bash
# Task 1: SAM ViT-B automatic mask generation (cached proposals) -> egg filter -> instance masks + montage.
# ~35-40 s/image on a 2-core CPU (~70 min for 110 images). Re-running skips images already cached.
set -euo pipefail
cd "$(dirname "$0")/.."
python src/sam_amg.py --points 10
cd src && python sam_filter.py && python viz_gt.py
