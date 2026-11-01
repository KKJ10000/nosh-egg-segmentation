"""Task 1 (stage A): run SAM ViT-B automatic mask generation on every image and cache ALL raw proposals.
Filtering to egg masks is done in sam_filter.py so the filter can be revised without re-running SAM."""
import argparse, glob, os, time
import numpy as np, cv2, torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/raw")
    ap.add_argument("--ckpt", default="weights/sam_vit_b_01ec64.pth")
    ap.add_argument("--out", default="outputs/sam_raw")
    ap.add_argument("--points", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(2)
    os.makedirs(a.out, exist_ok=True)
    sam = sam_model_registry["vit_b"](checkpoint=a.ckpt).eval()
    gen = SamAutomaticMaskGenerator(sam, points_per_side=a.points, points_per_batch=64,
                                    pred_iou_thresh=0.86, stability_score_thresh=0.90,
                                    min_mask_region_area=200)
    files = sorted(glob.glob(os.path.join(a.images, "*", "*", "*.jpg")))
    if a.limit: files = files[:a.limit]
    for i, f in enumerate(files):
        stem = os.path.splitext(os.path.basename(f))[0]
        dst = os.path.join(a.out, stem + ".npz")
        if os.path.exists(dst): continue
        t = time.time()
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            ms = gen.generate(img)
        masks = np.stack([m["segmentation"] for m in ms]) if ms else np.zeros((0,) + img.shape[:2], bool)
        np.savez_compressed(dst, masks=np.packbits(masks, axis=-1), shape=np.array(masks.shape),
                            iou=np.array([m["predicted_iou"] for m in ms]),
                            stab=np.array([m["stability_score"] for m in ms]), src=f)
        print(f"{i+1}/{len(files)} {stem} n={len(ms)} {time.time()-t:.1f}s", flush=True)

if __name__ == "__main__":
    main()
