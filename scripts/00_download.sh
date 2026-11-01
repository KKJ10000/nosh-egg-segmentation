#!/usr/bin/env bash
# Download dataset, video and SAM weights.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data weights
# Dataset (Roboflow "egg" v1, folder export). Alternative zip from the assignment (Google Drive id 1a58rACcCFnaj0ZiDoNyGbDWXunRyWziv):
if [ ! -d data/raw ]; then
  pip install -q gdown
  gdown --id 1a58rACcCFnaj0ZiDoNyGbDWXunRyWziv -O data/data.zip
  unzip -q data/data.zip -d data/raw
fi
# Video (if YouTube blocks yt-dlp on your network, download it manually and save as data/video.mp4)
[ -f data/video.mp4 ] || yt-dlp -f 'b[height<=720][ext=mp4]/b' -o data/video.mp4 "https://www.youtube.com/watch?v=Y4JyV94y2_A"
# SAM ViT-B checkpoint (375 MB)
[ -f weights/sam_vit_b_01ec64.pth ] || curl -L -o weights/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
