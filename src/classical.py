"""Task 2: classical (no learning) egg segmentation.
Every stage is switchable through a config dict so each trial in the experiment log is reproducible:
  feature map  -> threshold (single Otsu or multi-level sweep) -> morphology -> components
  -> shape filtering (area / solidity / ellipse fit) -> optional watershed split -> optional ellipse fit / GrabCut.
Returns an instance label map (uint16, 0 = background)."""
import numpy as np, cv2
from scipy import ndimage as ndi

BASE = dict(
    feature="gray",        # gray | L | whiteness (L - chroma_w*chroma) | whiteness_local
    chroma_w=1.5,          # weight of Lab chroma in the whiteness map
    local_sigma=60,        # for whiteness_local: subtract a heavily blurred copy (illumination / background normalisation)
    local_w=0.5,
    blur=5,                # Gaussian blur kernel on the feature map (0 = off)
    thresh="otsu",         # otsu | sweep
    otsu_offset=0,         # added to the Otsu threshold
    sweep_levels=12,       # number of thresholds in the sweep
    sweep_lo=40, sweep_hi=97,  # sweep thresholds span these percentiles of the feature map
    open_k=0, close_k=0,   # morphology kernel sizes (0 = off)
    fill_holes=False,
    keep="all",            # all | largest
    min_area=0.0, max_area=1.0,          # as fraction of the image
    min_solidity=0.0, min_ell_iou=0.0, min_axis_ratio=0.0,
    min_contrast=0.0,      # mean feature inside minus mean in a surrounding ring
    min_mean_f=0,          # minimum mean of the (0-255 normalised) feature map inside the component
    max_overlap=0.3,       # NMS for sweep candidates
    score="shape",         # shape: ell_iou*solidity | shape_contrast: additionally * min(1, contrast/contrast_norm)
    contrast_norm=40.0,
    area_w=0.0,            # bonus area_w*sqrt(area): prefer the whole egg over an elliptical highlight inside it
    rel_keep=0.0,          # keep only candidates scoring >= rel_keep * best score in the image
    watershed=False, ws_ratio=0.6,       # markers = distance transform > ws_ratio * local max
    ellipse_fit=False,     # replace each instance by its fitted ellipse
    grabcut=False, gc_iters=3, gc_margin=0.15,
    ws_refine=False,       # marker-controlled watershed on the colour image to snap each candidate to the egg edge
    wr_in=0.5, wr_out=0.5, # inner marker = candidate eroded by wr_in*r; background = beyond candidate dilated by wr_out*r (r = equiv. radius)
    edge_cands=False,      # add candidates from closed Canny contours (helps white-on-white scenes where brightness fails)
    canny_sigma=0.33, edge_dilate=3,
    work_size=640,
)

def feature_map(img, c):
    if c["feature"] == "gray":
        f = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[..., 0]; chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
        if c["feature"] == "L": f = L
        else:
            f = L - c["chroma_w"] * chroma
            if c["feature"] == "whiteness_local":
                bg = cv2.GaussianBlur(f, (0, 0), c["local_sigma"])
                f = f - c["local_w"] * bg
    if c["blur"]: f = cv2.GaussianBlur(f, (c["blur"] | 1, c["blur"] | 1), 0)
    f = cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX)
    return f.astype(np.uint8)

def clean(b, c):
    b = b.astype(np.uint8)
    if c["open_k"]: b = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c["open_k"],) * 2))
    if c["close_k"]: b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c["close_k"],) * 2))
    if c["fill_holes"]: b = ndi.binary_fill_holes(b).astype(np.uint8)
    return b

def comp_feats(m, f, pad=12):
    """Shape/contrast features of one binary component, computed on its bounding-box crop for speed."""
    ys, xs = np.nonzero(m)
    if len(ys) < 20: return None
    y0, y1 = max(0, ys.min() - pad), min(m.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(m.shape[1], xs.max() + pad + 1)
    mc = m[y0:y1, x0:x1].astype(np.uint8); fc = f[y0:y1, x0:x1]
    cs, _ = cv2.findContours(mc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cs, key=cv2.contourArea); a = cv2.contourArea(cnt)
    if len(cnt) < 5 or a < 20: return None
    hull = cv2.contourArea(cv2.convexHull(cnt))
    ell = cv2.fitEllipse(cnt)
    e = np.zeros(mc.shape, np.uint8); cv2.ellipse(e, ell, 1, -1)
    mb = mc.astype(bool)
    inter = np.logical_and(e, mb).sum(); uni = np.logical_or(e, mb).sum()
    ring = cv2.dilate(mc, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))).astype(bool) & ~mb
    contrast = float(fc[mb].mean() - (fc[ring].mean() if ring.any() else 0))
    (cx, cy), (w, h), ang = ell
    return dict(mean_f=float(fc[mb].mean()), area=len(ys) / m.size, solidity=a / max(hull, 1), ell_iou=inter / max(uni, 1),
                axis_ratio=min(w, h) / max(w, h, 1e-6), contrast=contrast, ellipse=((cx + x0, cy + y0), (w, h), ang))

def passes(ft, c):
    return (ft is not None and c["min_area"] <= ft["area"] <= c["max_area"] and ft["solidity"] >= c["min_solidity"]
            and ft["ell_iou"] >= c["min_ell_iou"] and ft["axis_ratio"] >= c["min_axis_ratio"] and ft["contrast"] >= c["min_contrast"]
            and ft["mean_f"] >= c["min_mean_f"])

def watershed_split(m, c):
    """Split touching eggs: markers = regional maxima of the distance transform above ws_ratio*max."""
    m = m.astype(bool)
    d = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    if d.max() == 0: return [m]
    mx = ndi.maximum_filter(d, size=int(max(5, d.max() * 0.8)))
    peaks = (d == mx) & (d > c["ws_ratio"] * d.max())
    mk, n = ndi.label(peaks)
    if n <= 1: return [m]
    markers = np.zeros(m.shape, np.int32); markers[~m] = 1; markers[mk > 0] = mk[mk > 0] + 1
    relief = cv2.cvtColor((255 - d / d.max() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    lab = cv2.watershed(relief, markers)
    parts = [(lab == k) & m for k in range(2, n + 2)]
    return [p for p in parts if p.any()]

def candidates_single(f, c):
    t, _ = cv2.threshold(f, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b = clean(f > t + c["otsu_offset"], c)
    n, lab = cv2.connectedComponents(b)
    return [lab == k for k in range(1, n)]

def candidates_sweep(f, c):
    out = []
    for t in np.percentile(f, np.linspace(c["sweep_lo"], c["sweep_hi"], c["sweep_levels"])):
        b = clean(f > t, c)
        n, lab, st, _ = cv2.connectedComponentsWithStats(b)
        for k in range(1, n):
            if st[k, cv2.CC_STAT_AREA] >= c["min_area"] * f.size: out.append(lab == k)
    return out

def candidates_edges(img, c):
    """Closed-contour candidates: Canny edges on the blurred gray image (auto thresholds from the median),
    dilated to close small gaps, then hole-filled; every filled region is a candidate."""
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    v = np.median(g); lo, hi = int(max(0, (1 - c["canny_sigma"]) * v)), int(min(255, (1 + c["canny_sigma"]) * v))
    e = cv2.Canny(g, lo, hi)
    e = cv2.dilate(e, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c["edge_dilate"],) * 2))
    filled = ndi.binary_fill_holes(e > 0) & ~(e > 0)
    filled = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    filled = cv2.dilate(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c["edge_dilate"],) * 2))  # give back the edge width
    n, lab, st, _ = cv2.connectedComponentsWithStats(filled)
    return [lab == k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= c["min_area"] * filled.size]

def grabcut_refine(img, m, c):
    x, y, w, h = cv2.boundingRect(m.astype(np.uint8))
    mg = int(c["gc_margin"] * max(w, h))
    gm = np.full(m.shape, cv2.GC_BGD, np.uint8)
    gm[max(0, y - mg):y + h + mg, max(0, x - mg):x + w + mg] = cv2.GC_PR_BGD
    gm[m > 0] = cv2.GC_PR_FGD
    er = cv2.erode(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    gm[er > 0] = cv2.GC_FGD
    try:
        cv2.grabCut(img, gm, None, np.zeros((1, 65)), np.zeros((1, 65)), c["gc_iters"], cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return m
    r = (gm == cv2.GC_FGD) | (gm == cv2.GC_PR_FGD)
    n, lab, st, _ = cv2.connectedComponentsWithStats(r.astype(np.uint8))
    if n <= 1: return m
    k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA]); r = ndi.binary_fill_holes(lab == k)
    return r

def ws_refine(img, m, c):
    """Marker-controlled watershed: sure-foreground = eroded candidate, sure-background = outside a dilated band.
    The flood meets at the strongest colour edge inside the band, which recovers the shaded side of the egg that a
    brightness threshold cuts off."""
    m = m.astype(np.uint8); r = np.sqrt(m.sum() / np.pi)
    ki = max(1, int(c["wr_in"] * r)); ko = max(3, int(c["wr_out"] * r))
    fg = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ki + 1,) * 2))
    if fg.sum() < 20: fg = cv2.erode(m, np.ones((3, 3), np.uint8))
    band = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ko + 1,) * 2))
    markers = np.zeros(m.shape, np.int32); markers[band == 0] = 1; markers[fg > 0] = 2
    lab = cv2.watershed(cv2.GaussianBlur(img, (5, 5), 0), markers)
    out = ndi.binary_fill_holes(lab == 2)
    out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)
    return out if out.sum() > 0 else m.astype(bool)

def segment(img, cfg=None, return_debug=False):
    c = dict(BASE, **(cfg or {}))
    H, W = img.shape[:2]; s = c["work_size"] / max(H, W)
    im = cv2.resize(img, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA) if s != 1 else img
    f = feature_map(im, c)
    cands = candidates_sweep(f, c) if c["thresh"] == "sweep" else candidates_single(f, c)
    if c["edge_cands"]: cands += candidates_edges(im, c)
    if c["watershed"]:
        cands = [p for m in cands for p in (watershed_split(m, c) if comp_feats(m, f) is not None and not passes(comp_feats(m, f), c) else [m])]
    scored = []
    for m in cands:
        ft = comp_feats(m, f)
        if passes(ft, c):
            sc = ft["ell_iou"] * ft["solidity"]
            if c["score"] == "shape_contrast": sc *= min(1.0, max(0.0, ft["contrast"]) / c["contrast_norm"])
            sc += c["area_w"] * np.sqrt(ft["area"])
            scored.append((sc, m, ft))
    if c["rel_keep"] and scored:
        best = max(x[0] for x in scored); scored = [x for x in scored if x[0] >= c["rel_keep"] * best]
    if c["keep"] == "largest" and scored:
        scored = [max(scored, key=lambda x: x[2]["area"])]
    # greedy NMS: best shape score first
    out = np.zeros(f.shape, np.uint16); taken = np.zeros(f.shape, bool); k = 0
    for sc, m, ft in sorted(scored, key=lambda x: (-x[0], -x[2]["area"])):
        if c["ellipse_fit"]:
            e = np.zeros(f.shape, np.uint8); cv2.ellipse(e, ft["ellipse"], 1, -1); m = e.astype(bool)
        if c["ws_refine"]:
            m = ws_refine(im, m, c)
        if c["grabcut"]:
            m = grabcut_refine(im, m, c)
        if (m & taken).sum() / max(m.sum(), 1) > c["max_overlap"]: continue
        k += 1; out[m & ~taken] = k; taken |= m
    if s != 1: out = cv2.resize(out, (W, H), interpolation=cv2.INTER_NEAREST)
    return (out, f) if return_debug else out
