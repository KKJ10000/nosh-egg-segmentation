# Task 2: Classical image-processing segmentation

## Dev/eval split with leakage reduction (4:1)

The export contains no Roboflow augmentation copies: all 110 original filenames (`not_damaged_<n>`) are unique, and only one pair is a near-duplicate (pHash distance 4). The real leakage risk is **same-scene photos**. The same egg on the same ash heap, steel bowl, pink bowl, keyboard or foil packet was photographed several times from slightly different angles. If such shots are split across dev and eval, eval effectively measures memorised backgrounds.

`src/split.py` therefore groups images before splitting:
1. **Background similarity:** an HSV colour histogram (12x6x6 bins) per image, Bhattacharyya distance, and complete-linkage clustering cut at 0.55.
2. **Near-duplicates:** images with pHash Hamming distance of 12 or less are merged into the same group.
3. **Whole groups go to one split:** groups are assigned to eval (seed 0) until it holds 20% of the images, preferring groups of 6 or fewer so eval keeps scene diversity.

I chose the cut of 0.55 by looking at montages of the groups. At 0.35, same-scene shots were still split apart (76 groups). At 0.55 the groups correspond to scenes (38 groups). After the split, the closest dev–eval pair has a pHash distance of 16, against 4 for the closest pair overall. The eval split ends up with keyboard, towel, hand, foil-packet, door and pink-bowl scenes; the ash, steel-bowl and egg-tray scenes are in dev.

[[table:split]]

I visually checked all 22 eval pseudo-GT masks: each is one correctly segmented egg. Dev contains about 9 images whose SAM masks are wrong or empty (eggs under water in steel bowls, eggs among leaves, backlit eggs, the egg tray). Those images lower the dev numbers for every trial equally, so they do not change which trial wins.

## Final algorithm

`src/classical.py` with `results/final_config.json`. No learning is involved; every step is a hand-designed operation:

{FINAL_ALGO}

## Experiment log (dev split only, 88 images, SAM pseudo-GT)

Every row below is one real run of `src/run_trial.py`, exactly as logged in `results/experiment_log.csv`. The exact commands are in `scripts/trials.sh`. Configs are overrides of `classical.BASE`. **IoU, Dice, precision, recall and BF are per-image means.** Inst F1 is instance matching at IoU 0.5. s/img is wall time on the 2-core box (2 processes; CPU steal was high at times). The decision column was written after looking at each result.

Trials were run in stages, and each stage was designed from the error maps of the previous best. Stage A built the basic blocks. Stage B changed how candidates are scored and selected. Stage C combined winners and added boundary refinement. Stage D targeted the 23 dev images with IoU below 0.1. Stage E combined and tuned. Stage F did the final tuning.

[[table:explog]]

{EXPLOG_STORY}
