# Task 4: Video tracking

**Output video (public Google Drive link):** https://drive.google.com/file/d/10-KmLy1rw8uT85Oxb9gxMyuJAdiJGkqu/view?usp=sharing (local copy: `outputs/egg_tracking.mp4`, 640x360, 24 fps, 60 s, all 1440 frames processed).

The video (YouTube Y4JyV94y2_A, "Raw and Boiled Egg Spinning") shows two white eggs on a dark wooden table. A hand picks them up, spins them and puts them down. It also has title cards, white overlay text ("Plane surface", "Raw Egg"), table reflections under the eggs, a white-and-red checked shirt and a camera re-framing at about 30 s.

## Segmenter adaptation

I used the same classical pipeline as Task 2 (whiteness map, threshold sweep, morphology, ellipse-shape scoring). Only the acceptance thresholds changed, because this domain differs from the photo dataset. On a few sampled frames I printed the features of every candidate that passed the shape test:
- Real eggs: ring contrast 139–168, mean whiteness 224–240.
- Table reflections and finger tips: contrast 26–46, mean whiteness 93–131.
- The green title card produced one huge elliptical blob (area 0.59 of the frame).

From those clearly separated clusters I set `min_contrast=70`, `min_mean_f=170`, `max_area=0.12`, `min_area=0.004` and `min_axis_ratio=0.5` (`results/video_config.json`). With these settings, reflections, text, fingers and the title card were no longer detected on the sampled frames. Frames are downscaled to 640x360, and the segmenter runs at about 0.15 s per frame on CPU.

## Tracker

`src/tracker.py` works as follows:
- Each track has a constant-velocity Kalman filter on the centroid.
- Every frame, detections are assigned to tracks with the Hungarian algorithm on a cost of `0.5*(1 - IoU(det, track mask shifted by the Kalman-predicted motion)) + 0.5*(centroid distance / 2.5 egg radii)`.
- A match is gated out if the distance exceeds 2.5 radii or the cost exceeds 0.85.
- A track gets an ID after 3 hits (this suppresses one-frame false positives) and survives 12 missed frames (0.5 s) of occlusion or motion blur.

**Version 1 (Kalman + Hungarian only)** produced 5 IDs for 2 physical eggs (`results/tracks_v1_no_reid.csv`). The cause: when the hand lifts an egg, the egg is occluded for 2–4 s and put down somewhere else, so no motion model can bridge the gap. The camera re-framing at about 30 s also moves both eggs at once.

**Version 2 (final)** adds a lost-track pool for re-identification. A deleted confirmed track stays in the pool for 10 s, and a newly confirmed track inherits the ID of the lost track whose last position is nearest. This gave exactly 2 IDs over the whole video. I checked the result by eye on sampled frames: egg 1 is lifted, carried and put down to the right, and keeps ID 1; egg 2 keeps ID 2 through the re-framing.

[[fig:results/figures/tracking_frames.jpg|Sampled frames from the tracking output (frames 300-990). Each egg keeps its ID through hand occlusion, a carry-and-place, and the camera re-framing.|17]]

[[fig:results/figures/timeline_tracks_v1_no_reid.png|Track timeline, v1 (no re-identification): 5 IDs for 2 eggs.|15]]

[[fig:results/figures/timeline_tracks.png|Track timeline, final: 2 persistent IDs. Gaps are frames where the egg is hidden in the hand or the frame is a title card.|15]]

## Limitations of the tracking

- An egg held in the hand or spinning fast is often not detected: the fingers break its elliptical outline and motion blur lowers its contrast. The track simply pauses until the egg is visible again.
- Re-identification by nearest last position is a heuristic. The two eggs look identical, so if both disappear and reappear at the same time, their IDs could be swapped. Nothing in the pixels can resolve that. Fixes would be a hand tracker that "carries" the ID with the hand, or a small re-ID embedding.
- There is no quantitative tracking metric (MOTA/IDF1), because the video has no ground-truth tracks. The checks were visual plus the ID count (2 eggs in the video, 2 IDs).
