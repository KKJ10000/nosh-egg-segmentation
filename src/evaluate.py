"""Task 3: evaluate the final classical segmenter against SAM pseudo-GT on the EVAL split (and dev for reference).
Writes results/metrics_{split}.csv, results/metrics_summary.json, and figures in results/figures/."""
import json, os, argparse
import numpy as np, cv2, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ROOT, list_images, stem, load_split, load_gt
from run_trial import evaluate
from viz_gt import overlay

FIG = os.path.join(ROOT, "results/figures")

def error_vis(img, gt, pr):
    """green = TP, red = FP (predicted egg, SAM says background), blue = FN (missed egg pixels)."""
    g, p = gt > 0, pr > 0; o = (img * 0.55).astype(np.uint8)
    o[g & p] = (0.45 * img[g & p] + 0.55 * np.array([0, 200, 0])).astype(np.uint8)
    o[~g & p] = (0.45 * img[~g & p] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
    o[g & ~p] = (0.45 * img[g & ~p] + 0.55 * np.array([255, 80, 0])).astype(np.uint8)
    return o

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cfg", default=os.path.join(ROOT, "results/final_config.json")); a = ap.parse_args()
    cfg = json.load(open(a.cfg))["segmenter"]
    os.makedirs(FIG, exist_ok=True); summ = {}
    paths = {stem(p): p for p in list_images()}
    for split in ["dev", "eval"]:
        rows, s = evaluate(cfg, split, save_dir=os.path.join(ROOT, f"outputs/classical_masks"))
        df = pd.DataFrame(rows).sort_values("iou"); df.to_csv(os.path.join(ROOT, f"results/metrics_{split}.csv"), index=False)
        summ[split] = s
    json.dump(summ, open(os.path.join(ROOT, "results/metrics_summary.json"), "w"), indent=1)
    ev = pd.read_csv(os.path.join(ROOT, "results/metrics_eval.csv"))
    dv = pd.read_csv(os.path.join(ROOT, "results/metrics_dev.csv"))
    # 1) per-image metric distributions
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ks = ["iou", "dice", "precision", "recall", "bf"]
    ax[0].boxplot([dv[k] for k in ks], tick_labels=ks); ax[0].set_title(f"Dev (n={len(dv)})"); ax[0].set_ylim(-0.02, 1.02)
    ax[1].boxplot([ev[k] for k in ks], tick_labels=ks); ax[1].set_title(f"Eval (n={len(ev)})"); ax[1].set_ylim(-0.02, 1.02)
    for x in ax: x.grid(alpha=.3)
    fig.suptitle("Per-image metrics vs SAM pseudo-GT"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "metric_boxplots.png"), dpi=130); plt.close(fig)
    # 2) precision vs recall scatter (failure mode map)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.scatter(dv["recall"], dv["precision"], alpha=.5, label="dev", s=22)
    ax.scatter(ev["recall"], ev["precision"], alpha=.9, label="eval", s=30, marker="s")
    ax.set_xlabel("recall (share of SAM egg pixels found)"); ax.set_ylabel("precision (share of predicted pixels that are egg)")
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03); ax.grid(alpha=.3); ax.legend()
    ax.set_title("Failure modes: low precision = background leak,\nlow recall = missed / under-segmented egg", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "precision_recall_scatter.png"), dpi=130); plt.close(fig)
    # 3) count error histogram
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    allc = pd.concat([dv.assign(split="dev"), ev.assign(split="eval")])
    bins = np.arange(allc["count_err"].min() - .5, allc["count_err"].max() + 1.5)
    ax.hist([dv["count_err"], ev["count_err"]], bins=bins, label=["dev", "eval"]); ax.legend()
    ax.set_xlabel("predicted eggs - SAM eggs"); ax.set_ylabel("images"); ax.set_title("Instance count error"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "count_error_hist.png"), dpi=130); plt.close(fig)
    # 4) IoU vs egg size
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for d, n in [(dv, "dev"), (ev, "eval")]:
        ax.scatter((d["tp"] + d["fn"]) / (640 * 640), d["iou"], label=n, alpha=.7, s=22)
    ax.set_xlabel("SAM egg area (fraction of image)"); ax.set_ylabel("IoU"); ax.grid(alpha=.3); ax.legend(); ax.set_title("IoU vs egg size")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "iou_vs_size.png"), dpi=130); plt.close(fig)
    # 5) worst eval failures + best for contrast
    worst = ev.nsmallest(6, "iou"); best = ev.nlargest(3, "iou")
    for name, sel in [("worst_eval_failures", worst), ("best_eval_examples", best)]:
        tiles = []
        for _, r in sel.iterrows():
            img = cv2.imread(paths[r["image"]]); gt = load_gt(r["image"])
            pr = cv2.imread(os.path.join(ROOT, "outputs/classical_masks", r["image"] + ".png"), cv2.IMREAD_UNCHANGED)
            t = [cv2.resize(x, (300, 300)) for x in (img, overlay(img, gt, (255, 0, 255)), overlay(img, pr, (0, 255, 255)), error_vis(img, gt, pr))]
            row = np.hstack(t)
            label = f"{r['image'].split('_jpg')[0]}  IoU={r['iou']:.2f}  P={r['precision']:.2f}  R={r['recall']:.2f}  #pred={r['n_pred']} #SAM={r['n_gt']}"
            bar = np.full((30, row.shape[1], 3), 255, np.uint8); cv2.putText(bar, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            tiles.append(np.vstack([bar, row]))
        head = np.full((34, 1200, 3), 255, np.uint8)
        for i, tx in enumerate(["image", "SAM mask (pseudo-GT)", "classical prediction", "error: green TP / red FP / blue FN"]):
            cv2.putText(head, tx, (8 + 300 * i, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imwrite(os.path.join(FIG, name + ".jpg"), np.vstack([head] + tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(json.dumps({k: {m: round(v, 4) if isinstance(v, float) else v for m, v in s.items()} for k, s in summ.items()}, indent=1))

if __name__ == "__main__":
    main()
