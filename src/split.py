"""Task 2: leakage-aware 4:1 dev/eval split.
The Roboflow export has no augmentation copies (110 unique source names), so the main leakage risk is
near-duplicate shots and *same-scene* shots (same egg/background photographed several times).
We group images by (a) perceptual hash distance and (b) background colour-histogram similarity, then
assign whole groups to dev or eval so no scene appears in both."""
import json, os, itertools
import numpy as np, cv2, imagehash
from PIL import Image
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from common import list_images, stem, source_id, ROOT

PHASH_T = 12      # Hamming distance <= 12 (of 64 bits) -> near duplicate
HIST_T = 0.55     # complete-linkage cut on Bhattacharyya distance of HSV histograms (chosen by inspecting group montages: 0.35 left same-scene shots split, 0.55 groups by scene)
EVAL_FRAC = 0.2
SEED = 0

def hsv_hist(img):
    hsv = cv2.cvtColor(cv2.resize(img, (128, 128)), cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [12, 6, 6], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(h, None).flatten()

def main():
    fs = list_images(); n = len(fs)
    ph = [imagehash.phash(Image.open(f)) for f in fs]
    hi = [hsv_hist(cv2.imread(f)) for f in fs]
    D = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        D[i, j] = D[j, i] = cv2.compareHist(hi[i], hi[j], cv2.HISTCMP_BHATTACHARYYA)
    lab = fcluster(linkage(squareform(D), "complete"), HIST_T, "distance")
    # union with pHash near-duplicates
    parent = list(range(n))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j in itertools.combinations(range(n), 2):
        if lab[i] == lab[j] or (ph[i] - ph[j]) <= PHASH_T:
            parent[find(i)] = find(j)
    groups = {}
    for i in range(n): groups.setdefault(find(i), []).append(i)
    groups = list(groups.values())
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(groups))
    target = round(n * EVAL_FRAC); ev = []
    # fill eval with whole groups, preferring small/medium groups so eval keeps scene diversity
    for gi in sorted(order, key=lambda g: (len(groups[g]) > 6, rng.random())):
        if len(ev) + len(groups[gi]) <= target + 1: ev += groups[gi]
        if len(ev) >= target: break
    evs = set(ev)
    out = {"dev": [stem(fs[i]) for i in range(n) if i not in evs],
           "eval": [stem(fs[i]) for i in range(n) if i in evs],
           "groups": [[stem(fs[i]) for i in g] for g in groups],
           "params": {"phash_t": PHASH_T, "hist_t": HIST_T, "eval_frac": EVAL_FRAC, "seed": SEED}}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results/split.json"), "w"), indent=1)
    sizes = sorted((len(g) for g in groups), reverse=True)
    print(f"{n} images -> {len(groups)} groups, sizes {sizes}; dev={len(out['dev'])} eval={len(out['eval'])}")
    # leakage check: closest cross-split pHash distance
    di = [i for i in range(n) if i not in evs]
    print("min cross-split pHash distance:", min(ph[i] - ph[j] for i in di for j in ev))

if __name__ == "__main__":
    main()
