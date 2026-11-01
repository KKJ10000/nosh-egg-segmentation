# Egg segmentation and tracking: Nosh Robotics CV intern task

**Kshitij Kumar Jha**. Code: https://github.com/KKJ10000/nosh-egg-segmentation. Tracking video (public): https://drive.google.com/file/d/10-KmLy1rw8uT85Oxb9gxMyuJAdiJGkqu/view?usp=sharing

## Summary

- **Task 1 (pretrained):** SAM ViT-B automatic masks plus a shape/brightness egg filter give 144 egg instances on the 110 images. These masks are the pseudo ground truth for Tasks 2-3.
- **Task 2 (classical):** a leakage-aware 4:1 split by scene (88 dev / 22 eval images), then 32 logged trials on dev only. The final pipeline (trial T31) is a locally normalised Lab whiteness map, a multi-level threshold sweep plus Canny closed-contour candidates, morphology, distance-transform watershed splitting of non-elliptical blobs, ellipse-shape and contrast scoring, and marker-controlled watershed refinement of each egg's edge. Dev mean IoU went from 0.327 (gray + Otsu baseline, T01) to 0.583.
- **Task 3 (evaluation, held-out eval split, 22 images):** mean IoU **0.543**, mean Dice **0.581**, pixel precision 0.573, pixel recall 0.599, boundary F **0.502**, pooled IoU 0.604, instance F1 0.588, exact egg-count accuracy 0.545 (count MAE 0.59). Results are bimodal: 14 of 22 eval images reach IoU 0.52-0.99 (9 are at 0.85 or above), and 8 are complete misses (IoU 0), mostly white eggs on white or bright textured backgrounds.
- **Task 4 (tracking):** classical per-frame segmentation, Kalman filter + Hungarian matching and lost-track re-identification keep exactly 2 IDs for the 2 eggs over the whole 60 s video. The video was produced with an earlier version of the segmenter (see Task 4).

## Data

The dataset is the Roboflow "egg" v1 export from the assignment's zip: 110 JPEGs at 640x640, exported in folder (classification) format. The only labels are folder names (egg / Unlabeled), so there are **no boxes or masks**. The images are mostly one to three white eggs, photographed by phone on very varied backgrounds: sand and ash, steel bowls with water, pink plastic bowls, green leaves, keyboards, towels, egg trays, foil packets and hands. Many backgrounds are themselves white or bright, which is what makes classical segmentation hard here.

All work was done on a 2-core CPU with 7 GB of RAM and no GPU.

# Task 1: Segmentation with a pretrained model (SAM)

## Method

- **Model:** Segment Anything, ViT-B checkpoint (`sam_vit_b_01ec64.pth`), run on CPU. The image encoder alone takes about 16 s per image at SAM's 1024 px input.
- **Prompting:** with no boxes in the dataset, box prompts were not possible. I used SAM's automatic mask generator with a 10x10 point grid (`pred_iou_thresh=0.86`, `stability_score_thresh=0.90`, `min_mask_region_area=200`). That took about 30–45 s per image, about 70 min for all 110. A 16x16 grid took about 60 s per image, which did not fit the time budget.
- **Caching:** every raw proposal is cached (`outputs/sam_raw/*.npz`), so the egg filter can be revised without re-running SAM.
- **Egg filter (`src/sam_filter.py`):** SAM proposes everything (bowls, leaves, keyboard keys, highlights). A proposal is kept as an egg if:
  - its area is 0.4–60% of the image;
  - it touches at most one image border, which rejects wedge-shaped corner fragments;
  - its solidity is at least 0.93;
  - its IoU with its own least-squares ellipse is at least 0.88;
  - its minor/major axis ratio is at least 0.45;
  - its mean Lab L is at least 120, it is not darker than a 31 px ring around it, and its mean chroma is at most 40.
- **De-duplication:** survivors are processed largest first, and a proposal is dropped if more than 40% of it is already covered. This keeps the whole egg rather than SAM's sub-part masks (highlight, shadow half).
- **Fallback:** if nothing passes, a relaxed pass (solidity 0.88, ellipse IoU 0.80, L 100) keeps the single brightest candidate.
- **Output:** `outputs/sam_masks/<image>.png` (uint16 instance-id map, 0 = background), plus a montage of all 110 images.

**Result:** the filter kept 144 egg instances on the 110 images: 81 images with one egg, 10 with two, 13 with three, 1 with four, and 5 with no accepted egg (all 5 in dev). The eval split has 23 eggs in 22 images (21 single-egg images and one two-egg image); dev has 121 eggs in 88 images.

[[fig:results/figures/sam_masks_montage_small.jpg|SAM ViT-B + egg filter on all 110 images (each colour is one instance). These masks are the pseudo ground truth for Tasks 2-3.|13.5]]

## Caveats about the pseudo-GT

The egg filter uses shape and brightness priors similar to the ones the classical method uses. Because of that, the GT is biased toward bright, elliptical, mostly unoccluded eggs:
- Eggs that are partly occluded (in a plastic bag, among leaves, under water) are sometimes missed by the filter or only partly masked.
- In a few images the fallback picked the wrong region.

These GT errors are counted against the classical method in Task 3 and are pointed out in the failure analysis where they occur (none of the six worst eval images is a GT error). With more time I would prompt SAM with a single positive point at each egg (one click per egg) to get a clean GT.

[[pagebreak]]

# Task 2: Classical image-processing segmentation

## Dev/eval split with leakage reduction (4:1)

The export contains no Roboflow augmentation copies: all 110 original filenames (`not_damaged_<n>`) are unique, and only one pair is a near-duplicate (pHash distance 4). The real leakage risk is **same-scene photos**. The same egg on the same ash heap, steel bowl, pink bowl, keyboard or foil packet was photographed several times from slightly different angles. If such shots are split across dev and eval, eval effectively measures memorised backgrounds.

`src/split.py` therefore groups images before splitting:
1. **Background similarity:** an HSV colour histogram (12x6x6 bins) per image, Bhattacharyya distance, and complete-linkage clustering cut at 0.55.
2. **Near-duplicates:** images with pHash Hamming distance of 12 or less are merged into the same group.
3. **Whole groups go to one split:** groups are assigned to eval (seed 0) until it holds 20% of the images, preferring groups of 6 or fewer so eval keeps scene diversity.

I chose the cut of 0.55 by looking at montages of the groups. At 0.35, same-scene shots were still split apart (76 groups). At 0.55 the groups correspond to scenes (38 groups). After the split, the closest dev–eval pair has a pHash distance of 16, against 4 for the closest pair overall. The eval split ends up with keyboard, towel, hand, foil-packet, door and pink-bowl scenes; the ash, steel-bowl and egg-tray scenes are in dev.

[[table:split]]

I visually checked all 22 eval pseudo-GT masks: every egg is correctly segmented (21 single-egg images and one two-egg image). Dev contains about 9 images whose SAM masks are wrong or empty (eggs under water in steel bowls, eggs among leaves, backlit eggs, the egg tray). Those images lower the dev numbers for every trial equally, so they do not change which trial wins.

## Final algorithm

`src/classical.py` with `results/final_config.json`. No learning is involved; every step is a hand-designed operation:

- **Feature map:** Lab whiteness `L - 1.5*chroma`, minus 0.5x a Gaussian-blurred copy (sigma 60 px) to remove the local background brightness, blurred (5 px) and normalised to 0-255.
- **Candidates:** a sweep of 12 thresholds at the 40th-97th percentiles of the feature map. Each binary map is opened and closed with a 7 px elliptical kernel and hole-filled, and every connected component is a candidate. Closed contours from an auto-threshold Canny edge map are added as further candidates.
- **Splitting merged blobs:** a candidate that fails the shape test is split with a distance-transform watershed (markers at regional maxima above 0.6x the maximum distance), and each part is tested again. This is the T31 change.
- **Shape test:** area 0.4-60% of the image, solidity of at least 0.90, IoU with the fitted ellipse of at least 0.85, axis ratio of at least 0.4.
- **Scoring and selection:** score = ellipse IoU x solidity x min(1, ring contrast / 40). Candidates scoring below 0.85x the best in the image are dropped, then a greedy non-maximum suppression (overlap 0.3) keeps the best non-overlapping candidates.
- **Edge refinement:** each kept egg is snapped to the colour edge with a marker-controlled watershed: sure foreground is the candidate eroded by 0.35r, sure background is outside the candidate dilated by 0.7r (r = equivalent radius).
- **Speed:** about 1.8 s per image on the 2-core CPU (2 worker processes).

[[pagebreak]]

## Experiment log (dev split only, 88 images, SAM pseudo-GT)

Every row below is one real run of `src/run_trial.py`, exactly as logged in `results/experiment_log.csv`. The exact commands are in `scripts/trials.sh`. Configs are overrides of `classical.BASE`. **IoU, Dice, precision, recall and BF are per-image means.** Inst F1 is instance matching at IoU 0.5. s/img is wall time on the 2-core box (2 processes; CPU steal was high at times). The decision column was written after looking at each result.

Trials were run in stages, and each stage was designed from the error maps of the previous best. Stage A built the basic blocks. Stage B changed how candidates are scored and selected. Stage C combined winners and added boundary refinement. Stage D targeted the 23 dev images with IoU below 0.1. Stage E combined and tuned. Stage F did the final tuning, and Stage G made the last checks around the best configuration.

[[table:explog]]

**How the pipeline evolved.** The gray + Otsu baseline (T01) found most egg pixels but also whole bright backgrounds (precision 0.36). Shape filtering alone (T06) removed the background but also most eggs, because a single global threshold merges each egg with its surroundings. The multi-level threshold sweep (T07) fixed that by letting each egg separate from its background at some level (IoU 0.24 -> 0.41). Shape-and-contrast scoring with a relative keep threshold (T10, T12) cut false positives. Watershed edge refinement (T13) recovered the shaded half of the eggs (0.44 -> 0.49). Local background subtraction in the whiteness map (T17) was the largest single gain (0.49 -> 0.55), because bright paper and cloth no longer outrank the egg. Edge candidates, a wider refinement band and, finally, the watershed split of merged blobs (T31) took dev IoU to **0.583**. Several plausible ideas made things worse and were dropped: GrabCut instead of watershed (T15), a contrast gate (T08), a stronger chroma penalty (T19), lower thresholds (T20), smaller or larger background scales (T24, T25), and a looser keep threshold (T27).

**Stage G (last checks around T28).** T29 (looser ellipse test, 0.571), T30 (replace each egg by its fitted ellipse, 0.573), T31 (watershed split, 0.583) and T32 (wider refinement band, 0.571). T31 had the best dev mean IoU and also the best BF, instance F1 and count MAE among them, so it is the final configuration (`results/final_config.json`). The eval split was not used for any of these choices.

[[pagebreak]]

# Task 3: Evaluation

## Protocol

The final configuration (T31) was run once on the 22 held-out eval images, with the SAM masks from Task 1 as ground truth (`src/evaluate.py`). Metrics:
- **IoU and Dice** of the egg/background mask, per image, then averaged. Pooled IoU and Dice sum the pixels over all images first, so large eggs weigh more.
- **Pixel precision and recall:** the share of predicted pixels that are egg, and the share of egg pixels that were found.
- **Boundary F:** precision and recall of boundary pixels matched within 9 px (1% of the image diagonal).
- **Instance metrics:** predicted and SAM instances are matched with the Hungarian algorithm; a match needs IoU of at least 0.5. Count error is predicted eggs minus SAM eggs.

## Results

[[table:summary]]

On the held-out split, mean IoU is **0.543** (dev 0.583), Dice 0.581, boundary F 0.502, and instance F1 0.588. Pooled IoU is the same on both splits (0.604 eval, 0.601 dev), so the drop in mean IoU comes from a few complete misses rather than worse masks overall.

The per-image results are strongly bimodal. 14 of 22 eval images reach IoU 0.52-0.99 (median over all 22 images: 0.73), 9 of them at 0.85 or above with boundary F up to 1.00. 8 images are complete misses with IoU 0. Exact egg-count accuracy is 0.545: in 4 images the egg is missed altogether, and in 6 images there are extra detections (up to +3).

[[fig:results/figures/metric_boxplots.png|Per-image metric distributions on dev (88) and eval (22).|16]]

[[fig:results/figures/precision_recall_scatter.png|Pixel precision vs recall per image. Points at (0, 0) are complete misses; points with high recall and lower precision are background leaks.|9]]

[[fig:results/figures/count_error_hist.png|Instance count error (predicted - SAM) per image.|9]]

[[fig:results/figures/iou_vs_size.png|IoU vs egg size.|9]]

[[pagebreak]]

## Failure analysis

[[fig:results/figures/worst_eval_failures.jpg|The six worst eval images. Columns: image, SAM pseudo-GT, classical prediction, error map (green TP, red FP, blue FN).|15]]

All six worst images are true failures of the classical method; the SAM masks are correct in each of them. The common causes:
- **White egg on a white or bright background** (cloth with stripes, a textured wall, a door, a white bowl). After local background subtraction the egg has almost no contrast against its surroundings, so no threshold level isolates it. The method then either finds nothing, or a small elliptical patch of background (a stripe, a fold) wins the ranking instead of the egg.
- **Egg on a hand:** the skin is bright with low chroma, and the egg merges with the palm in every threshold level. Only a small highlight on the hand passes the shape test.
- **Specular foil:** bright, irregular reflections dominate the whiteness map. The egg itself is merged with the foil, while small elliptical reflections pass the shape test.
- **Extra detections** in otherwise good images (count error +1 to +3), from round bright objects next to the egg.

[[fig:results/figures/best_eval_examples.jpg|The three best eval images for contrast: eggs with clear contrast to the background are segmented almost exactly.|15]]

## What I would do next

- Rank candidates with edge evidence along the whole ellipse outline (gradient magnitude on the fitted ellipse), not only brightness contrast. That targets the white-on-white cases, where the outline is still visible.
- Use a skin-colour model to suppress hands, and a specularity test (saturated, high-gradient pixels) to suppress foil.
- Cleaner ground truth: SAM prompted with one click per egg, instead of the automatic generator plus a filter.
- With labels, a small learned model would be the obvious step. The classical pipeline is a reasonable baseline at about 1.8 s per image on CPU.

[[pagebreak]]

# Task 4: Video tracking

**Note on versions:** the tracking video was produced with an **earlier version of the segmenter** than the final Task 2 configuration. It uses the global whiteness map, threshold sweep, morphology and ellipse-shape test (roughly the Stage A-B pipeline), with thresholds adapted to the video (`results/video_config.json`). It does not use local background subtraction, edge candidates, watershed splitting or watershed refinement. On this video (two white eggs on a dark table) that simpler segmenter was already sufficient, and I did not re-render the video with the final configuration.

**Output video (public Google Drive link):** https://drive.google.com/file/d/10-KmLy1rw8uT85Oxb9gxMyuJAdiJGkqu/view?usp=sharing (local copy in the repo: `results/egg_tracking.mp4`, 640x360, 24 fps, 60 s, all 1440 frames processed).

The video (YouTube Y4JyV94y2_A, "Raw and Boiled Egg Spinning") shows two white eggs on a dark wooden table. A hand picks them up, spins them and puts them down. It also has title cards, white overlay text ("Plane surface", "Raw Egg"), table reflections under the eggs, a white-and-red checked shirt and a camera re-framing at about 30 s.

## Segmenter adaptation

I used the same classical building blocks as Task 2 (whiteness map, threshold sweep, morphology, ellipse-shape scoring), in the earlier version described above. Only the acceptance thresholds changed, because this domain differs from the photo dataset. On a few sampled frames I printed the features of every candidate that passed the shape test:
- Real eggs: ring contrast 139–168, mean whiteness 224–240.
- Table reflections and finger tips: contrast 26–46, mean whiteness 93–131.
- The green title card produced one huge elliptical blob (area 0.59 of the frame).

From those clearly separated clusters I set `min_contrast=70`, `min_mean_f=170`, `max_area=0.12`, `min_area=0.004` and `min_axis_ratio=0.5` (`results/video_config.json`). With these settings, reflections, text, fingers and the title card were no longer detected on the sampled frames. Frames are downscaled to 640x360, and the segmenter runs at about 0.15 s per frame on CPU.

## Tracker

`src/tracker.py` works as follows:
- Each track has a constant-velocity Kalman filter on the centroid.
- Every frame, detections are assigned to tracks with the Hungarian algorithm on a cost of `0.5*(1 - IoU(det, track mask shifted by the Kalman-predicted motion)) + 0.5*(centroid distance / 2.5 egg radii)`.
- A match is gated out if the distance exceeds 2.5 radii or the cost exceeds 0.85.
- A track gets an ID after 3 hits (this suppresses one-frame false positives) and survives 12 missed frames (0.5 s) of occlusion or motion blur.

**Version 1 (Kalman + Hungarian only)** produced 5 IDs for 2 physical eggs (`results/tracks_v1_no_reid.csv`). The cause: when the hand lifts an egg, the egg is occluded for 2–4 s and put down somewhere else, so no motion model can bridge the gap. The camera re-framing at about 30 s also moves both eggs at once.

**Version 2 (final)** adds a lost-track pool for re-identification. A deleted confirmed track stays in the pool for 10 s, and a newly confirmed track inherits the ID of the lost track whose last position is nearest. This gave exactly 2 IDs over the whole video. I checked the result by eye on sampled frames: egg 1 is lifted, carried and put down to the right, and keeps ID 1; egg 2 keeps ID 2 through the re-framing.

[[fig:results/figures/tracking_frames.jpg|Sampled frames from the tracking output (frames 300-990). Each egg keeps its ID through hand occlusion, a carry-and-place, and the camera re-framing.|17]]

[[fig:results/figures/timeline_tracks_v1_no_reid.png|Track timeline, v1 (no re-identification): 5 IDs for 2 eggs.|15]]

[[fig:results/figures/timeline_tracks.png|Track timeline, final: 2 persistent IDs. Gaps are frames where the egg is hidden in the hand or the frame is a title card.|15]]

## Limitations of the tracking

- An egg held in the hand or spinning fast is often not detected: the fingers break its elliptical outline and motion blur lowers its contrast. The track simply pauses until the egg is visible again.
- Re-identification by nearest last position is a heuristic. The two eggs look identical, so if both disappear and reappear at the same time, their IDs could be swapped. Nothing in the pixels can resolve that. Fixes would be a hand tracker that "carries" the ID with the hand, or a small re-ID embedding.
- There is no quantitative tracking metric (MOTA/IDF1), because the video has no ground-truth tracks. The checks were visual plus the ID count (2 eggs in the video, 2 IDs).

[[pagebreak]]

# AI tools disclosure

I used AI coding tools (Hark, an AI assistant) on this assignment:
- **Implementation:** most of the Python code was written with Hark. That covers the SAM wrapper and egg filter, the split script, the classical pipeline with its configurable stages, the metrics, the tracker, the plotting and the PDF builder.
- **Report drafting:** Hark drafted the text of this report and the README.
- **Not AI-generated:** every number in the tables and figures comes from scripts in the repository, run on the data described here. The experiment log is the raw output of `run_trial.py` (`results/experiment_log.csv`), including the trials that made things worse.

I have reviewed the methods and can explain them and change them in the interview. Those are the Otsu and multi-level thresholding, the Lab whiteness map, the morphology, the ellipse-fit shape scoring, the distance-transform watershed, GrabCut, Hungarian assignment with a Kalman filter, and the boundary F-score.

# Reproducing

See README.md. In short:
- `scripts/00_download.sh` gets the data, video and weights.
- `scripts/01_task1_sam.sh` runs SAM.
- `scripts/02_task2_split_and_trials.sh` does the split and all the trials.
- `scripts/03_task3_evaluate.sh` runs the evaluation.
- `scripts/04_task4_track.sh` runs the tracking.
- `src/make_report.py` builds this PDF.

Code: https://github.com/KKJ10000/nosh-egg-segmentation
