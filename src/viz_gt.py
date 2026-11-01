"""Montage of SAM egg masks over images (for checking Task 1 output)."""
import glob, os, sys, numpy as np, cv2
from common import ROOT, list_images, stem
def overlay(img, lab, color=None):
    o = img.copy(); rng = np.random.default_rng(1)
    for k in range(1, int(lab.max()) + 1):
        m = lab == k
        c = np.array(color if color else rng.integers(60, 255, 3), np.uint8)
        o[m] = (0.5 * o[m] + 0.5 * c).astype(np.uint8)
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(o, cs, -1, (0, 255, 0), 2)
    return o
if __name__ == "__main__":
    mdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "outputs/sam_masks")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results/figures/sam_masks_montage.jpg")
    ims = []
    for f in list_images():
        p = os.path.join(mdir, stem(f) + ".png")
        if not os.path.exists(p): continue
        lab = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        ims.append(cv2.resize(overlay(cv2.imread(f), lab), (200, 200)))
    cols = 10
    while len(ims) % cols: ims.append(np.zeros((200, 200, 3), np.uint8))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cv2.imwrite(dst, np.vstack([np.hstack(ims[i:i + cols]) for i in range(0, len(ims), cols)]))
