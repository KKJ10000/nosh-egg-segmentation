"""Task 1 (stage B): turn cached SAM proposals into egg instance masks.
The dataset is a classification export (no boxes/polygons), so SAM cannot be box-prompted; instead we keep
SAM proposals that look like an egg:
  * area between 0.4% and 60% of the image, touching at most one image border
  * convex & elliptical: solidity >= 0.93 and IoU(mask, fitted ellipse) >= 0.88
  * axis ratio (minor/major) >= 0.45
  * bright and low-chroma relative to its surroundings (eggs here are white/cream)
Overlapping survivors are de-duplicated (largest first; a candidate is dropped if > 40% of it is already covered),
which removes SAM's sub-part masks (highlights, shadows halves) in favour of the whole egg.
Output: outputs/sam_masks/<stem>.png  (uint16 instance id map, 0 = background)."""
import glob, os, json, argparse
import numpy as np, cv2
from common import ROOT

P = dict(max_borders=1, min_area=0.004, max_area=0.60, min_solidity=0.93, min_ell_iou=0.88, min_axis_ratio=0.45,
         min_L=120, min_L_contrast=-5, max_chroma=40, max_overlap=0.40)

def load_raw(path):
    z = np.load(path, allow_pickle=True)
    shp = tuple(z["shape"])
    m = np.unpackbits(z["masks"], axis=-1)[..., :shp[-1]].astype(bool) if shp[0] else np.zeros(shp, bool)
    return m, z["iou"], str(z["src"])

def shape_feats(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs: return None
    c = max(cs, key=cv2.contourArea); a = cv2.contourArea(c)
    if len(c) < 5 or a < 10: return None
    hull = cv2.contourArea(cv2.convexHull(c))
    (cx, cy), (w, h), ang = cv2.fitEllipse(c)
    e = np.zeros(mask.shape, np.uint8)
    cv2.ellipse(e, ((cx, cy), (w, h), ang), 1, -1)
    inter = np.logical_and(e, mask).sum(); uni = np.logical_or(e, mask).sum()
    return dict(solidity=a / max(hull, 1), ell_iou=inter / max(uni, 1), axis_ratio=min(w, h) / max(w, h, 1e-6))

RELAX = dict(min_solidity=0.88, min_ell_iou=0.80, min_L=100)  # fallback pass when the strict pass finds nothing

def egg_masks(masks, img, p=P, _fallback=True):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] * 100 / 255 * 2.55  # keep 0..255 scale
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    H, W = img.shape[:2]; N = H * W
    cands = []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    for i, m in enumerate(masks):
        a = m.sum() / N
        if not (p["min_area"] <= a <= p["max_area"]): continue
        ys, xs = np.nonzero(m)   # reject wedge-like corner/edge fragments touching >= 2 image borders
        if (ys.min() == 0) + (xs.min() == 0) + (ys.max() == H - 1) + (xs.max() == W - 1) > p["max_borders"]: continue
        f = shape_feats(m)
        if f is None or f["solidity"] < p["min_solidity"] or f["ell_iou"] < p["min_ell_iou"] or f["axis_ratio"] < p["min_axis_ratio"]:
            continue
        ring = cv2.dilate(m.astype(np.uint8), k).astype(bool) & ~m
        mL, rL = L[m].mean(), (L[ring].mean() if ring.any() else 0)
        if mL < p["min_L"] or mL - rL < p["min_L_contrast"] or chroma[m].mean() > p["max_chroma"]: continue
        cands.append((m.sum(), i))
    out = np.zeros((H, W), np.uint16); taken = np.zeros((H, W), bool); k_id = 0
    for area, i in sorted(cands, reverse=True):
        m = masks[i]
        if (m & taken).sum() / area > p["max_overlap"]: continue
        k_id += 1; out[m & ~taken] = k_id; taken |= m
    if k_id == 0 and _fallback:
        # Fallback: relax shape/brightness limits and keep only the single brightest candidate.
        q = dict(p, **RELAX)
        lab2 = egg_masks(masks, img, q, _fallback=False)
        if lab2.max() > 0:
            best = max(range(1, int(lab2.max()) + 1), key=lambda j: L[lab2 == j].mean())
            out = (lab2 == best).astype(np.uint16)
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--raw", default=os.path.join(ROOT, "outputs/sam_raw"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs/sam_masks")); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); stats = {}
    for f in sorted(glob.glob(os.path.join(a.raw, "*.npz"))):
        masks, iou, src = load_raw(f)
        img = cv2.imread(src if os.path.isabs(src) else os.path.join(ROOT, src))
        lab = egg_masks(masks, img)
        st = os.path.basename(f)[:-4]
        cv2.imwrite(os.path.join(a.out, st + ".png"), lab)
        stats[st] = int(lab.max())
    json.dump({"params": P, "instances_per_image": stats}, open(os.path.join(ROOT, "results/sam_gt_stats.json"), "w"), indent=1)
    c = np.bincount(list(stats.values())); print("images:", len(stats), "instance-count histogram:", c.tolist())

if __name__ == "__main__":
    main()
