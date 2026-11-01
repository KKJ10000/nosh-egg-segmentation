# Egg segmentation and tracking: Nosh Robotics CV intern task

**Kshitij Kumar Jha**. Code: https://github.com/KKJ10000/nosh-egg-segmentation. Tracking video (public): https://drive.google.com/file/d/10-KmLy1rw8uT85Oxb9gxMyuJAdiJGkqu/view?usp=sharing

## Summary

{SUMMARY}

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

{TASK1_STATS}

[[fig:results/figures/sam_masks_montage_small.jpg|SAM ViT-B + egg filter on all 110 images (each colour is one instance). These masks are the pseudo ground truth for Tasks 2-3.|17.5]]

## Caveats about the pseudo-GT

The egg filter uses shape and brightness priors similar to the ones the classical method uses. Because of that, the GT is biased toward bright, elliptical, mostly unoccluded eggs:
- Eggs that are partly occluded (in a plastic bag, among leaves, under water) are sometimes missed by the filter or only partly masked.
- In a few images the fallback picked the wrong region.

These GT errors are counted against the classical method in Task 3 and are pointed out in the failure analysis where they occur. With more time I would prompt SAM with a single positive point at each egg (one click per egg) to get a clean GT.
