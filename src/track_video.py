"""Task 4: run the final classical segmenter + tracker on the video and write an annotated MP4."""
import argparse, json, os, subprocess, time, csv
import numpy as np, cv2
from common import ROOT
from classical import segment
from tracker import MultiTracker

COLORS = [tuple(int(x) for x in c) for c in np.random.default_rng(7).integers(40, 255, (64, 3))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(ROOT, "data/video.mp4"))
    ap.add_argument("--cfg", default=os.path.join(ROOT, "results/video_config.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs/egg_tracking.mp4"))
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--stride", type=int, default=1, help="process every n-th frame")
    ap.add_argument("--max_frames", type=int, default=0)
    a = ap.parse_args()
    cfg = json.load(open(a.cfg))
    seg_cfg, trk_cfg = cfg["segmenter"], cfg.get("tracker", {})
    cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS)
    tmp = a.out.replace(".mp4", "_raw.avi"); wr = None
    mt = MultiTracker(**trk_cfg); log = []; i = 0; t0 = time.time()
    while True:
        ok, fr = cap.read()
        if not ok or (a.max_frames and i >= a.max_frames): break
        if i % a.stride: i += 1; continue
        h, w = fr.shape[:2]; s = a.width / w
        fr = cv2.resize(fr, (a.width, round(h * s)), interpolation=cv2.INTER_AREA)
        lab = segment(fr, dict(seg_cfg, work_size=a.width))
        live = mt.step(lab)
        vis = fr.copy()
        for t in live:
            col = COLORS[t.id % len(COLORS)]
            ov = vis.copy(); ov[t.mask] = col; vis = cv2.addWeighted(ov, 0.35, vis, 0.65, 0)
            cs, _ = cv2.findContours(t.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cs, -1, col, 2)
            pts = np.array(t.trail, np.int32)
            if len(pts) > 1: cv2.polylines(vis, [pts], False, col, 2)
            x, y = int(t.c[0]), int(t.c[1])
            cv2.putText(vis, f"egg {t.id}", (x - 25, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(vis, f"egg {t.id}", (x - 25, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            log.append([i, round(i / fps, 3), t.id, x, y, int(t.mask.sum())])
        cv2.putText(vis, f"frame {i}  eggs tracked: {len(live)}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if wr is None:
            wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), fps / a.stride, (vis.shape[1], vis.shape[0]))
        wr.write(vis); i += 1
        if i % 100 == 0: print(f"frame {i} {time.time() - t0:.0f}s", flush=True)
    wr.release()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", a.out], check=True)
    os.remove(tmp)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results/tracks.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["frame", "time_s", "track_id", "cx", "cy", "area_px"]); w.writerows(log)
    ids = sorted({r[2] for r in log})
    print(json.dumps({"frames": i, "track_ids": ids, "seconds": round(time.time() - t0, 1), "out": a.out}))

if __name__ == "__main__":
    main()
