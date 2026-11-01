"""Figures for Task 4: sampled annotated frames + track timeline."""
import os, numpy as np, cv2, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import ROOT
FIG = os.path.join(ROOT, "results/figures"); os.makedirs(FIG, exist_ok=True)
cap = cv2.VideoCapture(os.path.join(ROOT, "outputs/egg_tracking.mp4"))
want = [300, 470, 520, 590, 700, 780, 870, 990]; tiles = []
for f in want:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, fr = cap.read()
    if ok: tiles.append(cv2.resize(fr, (480, 270)))
cv2.imwrite(os.path.join(FIG, "tracking_frames.jpg"), np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
for name in ["tracks.csv", "tracks_v1_no_reid.csv"]:
    p = os.path.join(ROOT, "results", name)
    if not os.path.exists(p): continue
    d = pd.read_csv(p); fig, ax = plt.subplots(figsize=(8, 2.4))
    for tid, g in d.groupby("track_id"):
        ax.scatter(g["time_s"], [tid] * len(g), s=2, label=f"ID {tid}")
    ax.set_xlabel("video time (s)"); ax.set_ylabel("track ID"); ax.set_yticks(sorted(d["track_id"].unique())); ax.set_xlim(0, 60); ax.grid(alpha=.3)
    ax.set_title("Track timeline " + ("(final: with re-identification)" if name == "tracks.csv" else "(v1: Kalman+Hungarian only, no re-ID)"), fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "timeline_" + name.replace(".csv", ".png")), dpi=130); plt.close(fig)
