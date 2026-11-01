"""Task 4: multi-object tracker for egg instances.
Detections = instance masks from the classical segmenter. Each track has a constant-velocity Kalman filter on
(cx, cy, vx, vy). Association per frame = Hungarian assignment on cost = 1 - IoU(mask, track's last mask shifted by
the Kalman-predicted motion) blended with normalised centroid distance; gated so implausible matches are rejected.
Unmatched detections start tentative tracks (confirmed after `min_hits`); tracks survive `max_age` missed frames
(occlusion by the hand, motion blur) before being deleted. Re-identification: a confirmed track that is deleted goes to a "lost" pool for `reid_window` frames. When a new
track is confirmed while the pool is non-empty, it inherits the ID of the lost track whose last position is
nearest (eggs in this video disappear while carried in the hand and reappear elsewhere, so a pure motion model
cannot bridge the gap; with visually identical eggs this is a heuristic, see report)."""
import numpy as np, cv2
from scipy.optimize import linear_sum_assignment

class Track:
    _next = 1
    def __init__(self, mask, c):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        self.kf.processNoiseCov = np.diag([1, 1, 4, 4]).astype(np.float32)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4
        self.kf.errorCovPost = np.diag([10, 10, 100, 100]).astype(np.float32)
        self.kf.statePost = np.array([[c[0]], [c[1]], [0], [0]], np.float32)
        self.mask, self.c, self.hits, self.miss, self.id = mask, c, 1, 0, None
        self.trail = [c]; self.area = mask.sum()
    def predict(self):
        p = self.kf.predict(); return float(p[0, 0]), float(p[1, 0])
    def update(self, mask, c):
        self.kf.correct(np.array([[c[0]], [c[1]]], np.float32))
        self.mask, self.c = mask, c; self.hits += 1; self.miss = 0
        self.area = 0.7 * self.area + 0.3 * mask.sum()
        self.trail = (self.trail + [c])[-30:]

def centroid(m):
    ys, xs = np.nonzero(m); return float(xs.mean()), float(ys.mean())

def shifted_iou(track, pred_c, m):
    dx, dy = int(round(pred_c[0] - track.c[0])), int(round(pred_c[1] - track.c[1]))
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    sm = cv2.warpAffine(track.mask.astype(np.uint8), M, (m.shape[1], m.shape[0])).astype(bool)
    inter = np.logical_and(sm, m).sum(); uni = np.logical_or(sm, m).sum()
    return inter / uni if uni else 0.0

class MultiTracker:
    def __init__(self, max_age=12, min_hits=3, gate=0.85, dist_w=0.5, reid_window=0):
        self.tracks, self.max_age, self.min_hits, self.gate, self.dist_w = [], max_age, min_hits, gate, dist_w
        self.reid_window, self.lost, self.frame = reid_window, [], 0
    def step(self, label_map):
        dets = [label_map == k for k in range(1, int(label_map.max()) + 1)]
        dets = [d for d in dets if d.sum() > 0]
        cents = [centroid(d) for d in dets]
        preds = [t.predict() for t in self.tracks]
        C = np.ones((len(self.tracks), len(dets)))
        for i, t in enumerate(self.tracks):
            r = np.sqrt(t.area / np.pi) + 1e-6   # track radius scale
            for j, d in enumerate(dets):
                dist = np.hypot(cents[j][0] - preds[i][0], cents[j][1] - preds[i][1]) / (2.5 * r)
                if dist > 1.0: continue          # gate: farther than 2.5 egg radii
                C[i, j] = (1 - self.dist_w) * (1 - shifted_iou(t, preds[i], d)) + self.dist_w * dist
        matched_t, matched_d = set(), set()
        if C.size:
            for i, j in zip(*linear_sum_assignment(C)):
                if C[i, j] <= self.gate:
                    self.tracks[i].update(dets[j], cents[j]); matched_t.add(i); matched_d.add(j)
        for i, t in enumerate(self.tracks):
            if i not in matched_t: t.miss += 1
        self.frame += 1
        for t in self.tracks:
            if t.miss > self.max_age and t.id is not None and self.reid_window:
                self.lost.append((self.frame, t.id, t.c))
        self.tracks = [t for t in self.tracks if t.miss <= self.max_age]
        self.lost = [l for l in self.lost if self.frame - l[0] <= self.reid_window]
        for j, d in enumerate(dets):
            if j not in matched_d: self.tracks.append(Track(d, cents[j]))
        for t in self.tracks:
            if t.id is None and t.hits >= self.min_hits:
                if self.lost:   # re-identify: inherit the nearest recently lost ID
                    k = min(range(len(self.lost)), key=lambda i: np.hypot(self.lost[i][2][0] - t.c[0], self.lost[i][2][1] - t.c[1]))
                    t.id = self.lost.pop(k)[1]
                else:
                    t.id = Track._next; Track._next += 1
        return [t for t in self.tracks if t.id is not None and t.miss == 0]
