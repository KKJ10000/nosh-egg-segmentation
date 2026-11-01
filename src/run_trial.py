"""Run one named classical-segmentation trial on the DEV split and append it to results/experiment_log.csv.
Usage: python run_trial.py <trial_id> '<json cfg overrides>' "<what changed / why>" [--split dev|eval] [--nolog]"""
import sys, json, os, csv, time, argparse
import numpy as np, cv2
from multiprocessing import Pool
from common import ROOT, list_images, stem, load_split, load_gt
from classical import segment, BASE
from metrics import all_metrics, summarize

def _one(args):
    path, cfg = args
    gt = load_gt(stem(path)); pr = segment(cv2.imread(path), cfg)
    r = all_metrics(gt, pr); r["image"] = stem(path); return r, pr

def evaluate(cfg, split="dev", save_dir=None, procs=2):
    names = set(load_split()[split])
    paths = [p for p in list_images() if stem(p) in names]
    with Pool(procs) as pool: res = pool.map(_one, [(p, cfg) for p in paths])
    rows = [r for r, _ in res]
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for r, pr in res: cv2.imwrite(os.path.join(save_dir, r["image"] + ".png"), pr)
    return rows, summarize(rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("tid"); ap.add_argument("cfg"); ap.add_argument("why")
    ap.add_argument("--split", default="dev"); ap.add_argument("--nolog", action="store_true")
    a = ap.parse_args(); cfg = json.loads(a.cfg); t = time.time()
    rows, s = evaluate(cfg, a.split)
    dt = (time.time() - t) / len(rows)
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()}))
    if not a.nolog:
        f = os.path.join(ROOT, "results/experiment_log.csv"); new = not os.path.exists(f)
        with open(f, "a", newline="") as fh:
            w = csv.writer(fh)
            if new: w.writerow(["trial", "split", "change_and_reason", "config_overrides", "mean_iou", "mean_dice", "mean_precision",
                                "mean_recall", "mean_bf", "inst_f1", "count_mae", "sec_per_img", "n_images", "timestamp"])
            w.writerow([a.tid, a.split, a.why, json.dumps(cfg), *(round(s[k], 4) for k in ["mean_iou", "mean_dice", "mean_precision", "mean_recall", "mean_bf", "inst_f1", "count_mae"]),
                        round(dt, 3), s["n_images"], time.strftime("%Y-%m-%d %H:%M")])
