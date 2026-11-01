# Egg segmentation and tracking (Nosh Robotics CV intern task)

Kshitij Kumar Jha. Repo: https://github.com/KKJ10000/nosh-egg-segmentation

- **Report:** [`REPORT.pdf`](REPORT.pdf) covers all four tasks, the full experiment log, the evaluation and the AI-tools disclosure.
- **Tracking output video (Google Drive, public):** https://drive.google.com/file/d/10-KmLy1rw8uT85Oxb9gxMyuJAdiJGkqu/view?usp=sharing

| Task | What it does | Code | Outputs |
|---|---|---|---|
| 1. Pretrained segmentation | SAM ViT-B automatic mask generation, then an egg filter (shape + brightness) | `src/sam_amg.py`, `src/sam_filter.py` | `outputs/sam_masks/*.png` (instance id maps), `results/figures/sam_masks_montage.jpg` |
| 2. Classical segmentation | Leakage-aware 4:1 split, then a whiteness map, multi-level threshold sweep, morphology, ellipse-shape scoring and NMS | `src/split.py`, `src/classical.py`, `src/run_trial.py`, `scripts/trials.sh` | `results/split.json`, `results/experiment_log.csv`, `results/final_config.json` |
| 3. Evaluation | IoU, Dice, precision, recall, boundary F, instance matching and count error against the SAM pseudo-GT, plus plots and failure visualisations | `src/metrics.py`, `src/evaluate.py` | `results/metrics_*.csv`, `results/metrics_summary.json`, `results/figures/*` |
| 4. Video tracking | Per-frame classical segmentation, Kalman filter + Hungarian matching (IoU and distance cost), and re-identification of lost tracks | `src/tracker.py`, `src/track_video.py` | `results/egg_tracking.mp4`, `results/tracks.csv` |

## Final results (Task 3, held-out eval split)

Final classical configuration: trial **T31** (`results/final_config.json`), chosen on the dev split only (best dev mean IoU among T29–T32). Ground truth = SAM pseudo-masks from Task 1.

| Metric | Dev (88 images) | Eval (22 images, held out) |
|---|---|---|
| Mean IoU | 0.583 | **0.543** |
| Mean Dice | 0.634 | **0.581** |
| Mean pixel precision | 0.709 | 0.573 |
| Mean pixel recall | 0.639 | 0.599 |
| Mean boundary F (9 px) | 0.553 | **0.502** |
| Pooled IoU | 0.601 | 0.604 |
| Instance F1 (IoU ≥ 0.5) | 0.757 | 0.588 |
| Exact egg-count accuracy | 0.693 | 0.545 |
| Count MAE (eggs/image) | 0.36 | 0.59 |

Eval results are bimodal: 14 of 22 images reach IoU 0.52–0.99 (9 at ≥ 0.85), and 8 are complete misses, mostly white eggs on white or bright textured backgrounds, an egg on a hand, and foil. See `results/figures/worst_eval_failures.jpg` and the report.

The tracking video (Task 4) was produced with an earlier, simpler version of the classical segmenter (`results/video_config.json`), not the final T31 configuration; it keeps exactly 2 IDs for the 2 eggs.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu   # only needed for Task 1
pip install -r requirements.txt
bash scripts/00_download.sh      # dataset zip, video, SAM ViT-B weights -> data/, weights/
```

Everything runs on CPU. It was developed on 2 cores and 7 GB of RAM with no GPU.

Large files are not in git (see `.gitignore`):
- **Dataset:** Roboflow `egg` v1 (https://universe.roboflow.com/fire-bf7qe/egg-vzhtr/dataset/1), or the assignment's Google Drive zip (id `1a58rACcCFnaj0ZiDoNyGbDWXunRyWziv`). Unzip it into `data/raw/`.
- **SAM weights:** `sam_vit_b_01ec64.pth` from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth, saved in `weights/`.
- **Video:** https://www.youtube.com/watch?v=Y4JyV94y2_A, saved as `data/video.mp4`. If YouTube blocks yt-dlp on your network, download it manually.

## Reproduce

```bash
bash scripts/01_task1_sam.sh                 # ~35 s/image on 2 CPU cores (~70 min); cached and resumable
bash scripts/02_task2_split_and_trials.sh    # split + every experiment-log trial (dev split only)
bash scripts/03_task3_evaluate.sh            # final config on dev + held-out eval; metrics and figures
bash scripts/04_task4_track.sh               # tracking video (~4 min at 640x360)
cd src && python video_figs.py && python make_report.py   # figures for Task 4 + REPORT.pdf
```

The SAM masks are committed in `outputs/sam_masks/`, so Tasks 2–4 can be reproduced without running SAM.

## Repository layout

```
src/        common.py, sam_amg.py, sam_filter.py, viz_gt.py, split.py, classical.py, metrics.py,
            run_trial.py, evaluate.py, tracker.py, track_video.py, video_figs.py, make_report.py
scripts/    one shell script per task + trials.sh (the exact trial commands behind the experiment log)
results/    split.json, experiment_log.csv, experiment_decisions.json, final_config.json, video_config.json,
            metrics_dev.csv, metrics_eval.csv, metrics_summary.json, tracks.csv, figures/
outputs/    sam_masks/ (Task 1), classical_masks/ (Task 2/3)
report_text.md  source text of REPORT.pdf
```

## AI tools

AI coding tools (Hark, an AI assistant) were used to implement the code and to draft this README and the report. The method choices, the experiment log and the numbers come from runs in this repo. See the "AI tools" section of the report.
