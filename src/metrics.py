"""Segmentation metrics. GT and prediction are instance label maps (0 = background)."""
import numpy as np, cv2
from scipy.optimize import linear_sum_assignment

def semantic(gt, pr):
    g, p = gt > 0, pr > 0
    tp = np.logical_and(g, p).sum(); fp = np.logical_and(~g, p).sum(); fn = np.logical_and(g, ~p).sum()
    if g.sum() == 0 and p.sum() == 0:
        return dict(iou=1.0, dice=1.0, precision=1.0, recall=1.0, tp=0, fp=0, fn=0)
    return dict(iou=tp / (tp + fp + fn), dice=2 * tp / (2 * tp + fp + fn),
                precision=tp / (tp + fp) if tp + fp else 0.0, recall=tp / (tp + fn) if tp + fn else 1.0,
                tp=int(tp), fp=int(fp), fn=int(fn))

def boundary_f(gt, pr, tol=None):
    """Boundary F-score (Martin et al./DAVIS style): boundary pixels matched within `tol` px.
    tol defaults to 1% of the image diagonal (~9 px at 640x640)."""
    g, p = (gt > 0).astype(np.uint8), (pr > 0).astype(np.uint8)
    if tol is None: tol = max(1, round(0.01 * np.hypot(*g.shape)))
    k = np.ones((3, 3), np.uint8)
    bg = g - cv2.erode(g, k); bp = p - cv2.erode(p, k)
    if bg.sum() == 0 and bp.sum() == 0: return 1.0
    if bg.sum() == 0 or bp.sum() == 0: return 0.0
    dg = cv2.distanceTransform(1 - bg, cv2.DIST_L2, 3); dp = cv2.distanceTransform(1 - bp, cv2.DIST_L2, 3)
    prec = (dg[bp > 0] <= tol).mean(); rec = (dp[bg > 0] <= tol).mean()
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)

def instances(gt, pr, thr=0.5):
    gi = [k for k in np.unique(gt) if k]; pi = [k for k in np.unique(pr) if k]
    M = np.zeros((len(gi), len(pi)))
    for a, x in enumerate(gi):
        gm = gt == x
        for b, y in enumerate(pi):
            pm = pr == y; inter = np.logical_and(gm, pm).sum()
            if inter: M[a, b] = inter / np.logical_or(gm, pm).sum()
    matched = []
    if M.size:
        r, c = linear_sum_assignment(-M)
        matched = [M[i, j] for i, j in zip(r, c) if M[i, j] >= thr]
    tp = len(matched)
    return dict(n_gt=len(gi), n_pred=len(pi), inst_tp=tp, inst_fp=len(pi) - tp, inst_fn=len(gi) - tp,
                matched_miou=float(np.mean(matched)) if matched else float("nan"),
                count_err=len(pi) - len(gi))

def all_metrics(gt, pr):
    d = semantic(gt, pr); d["bf"] = boundary_f(gt, pr); d.update(instances(gt, pr)); return d

def summarize(rows):
    """rows: list of dicts from all_metrics. Returns mean per-image + pooled + instance F1."""
    import math
    keys = ["iou", "dice", "precision", "recall", "bf"]
    s = {f"mean_{k}": float(np.mean([r[k] for r in rows])) for k in keys}
    TP = sum(r["tp"] for r in rows); FP = sum(r["fp"] for r in rows); FN = sum(r["fn"] for r in rows)
    s["pooled_iou"] = TP / (TP + FP + FN); s["pooled_dice"] = 2 * TP / (2 * TP + FP + FN)
    itp = sum(r["inst_tp"] for r in rows); ifp = sum(r["inst_fp"] for r in rows); ifn = sum(r["inst_fn"] for r in rows)
    s["inst_precision"] = itp / (itp + ifp) if itp + ifp else 0.0
    s["inst_recall"] = itp / (itp + ifn) if itp + ifn else 0.0
    s["inst_f1"] = 2 * itp / (2 * itp + ifp + ifn) if itp + ifp + ifn else 1.0
    s["count_mae"] = float(np.mean([abs(r["count_err"]) for r in rows]))
    s["count_exact_acc"] = float(np.mean([r["count_err"] == 0 for r in rows]))
    s["n_images"] = len(rows)
    return s
