#!/usr/bin/env bash
# Task 4: classical segmentation + Kalman/Hungarian tracking on the video -> outputs/egg_tracking.mp4
set -euo pipefail
cd "$(dirname "$0")/../src"
python track_video.py --width 640 --stride 1
