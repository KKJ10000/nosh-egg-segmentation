#!/usr/bin/env bash
# Exact trial commands behind results/experiment_log.csv (run from src/). Each trial = config overrides on classical.BASE.
set -euo pipefail
cd "$(dirname "$0")/../src"
T() { python run_trial.py "$@"; }
# ---- Stage A: building blocks ----
T T01 '{}' "Baseline: grayscale, global Otsu, keep every connected component. Eggs are bright, so a single global threshold is the simplest possible detector."
T T02 '{"keep":"largest"}' "Keep only the largest component, since most images contain one egg; tests whether fragment false positives dominate the error."
T T03 '{"feature":"L"}' "Lab L channel instead of gray: perceptual lightness, less affected by colour casts."
T T04 '{"feature":"whiteness","chroma_w":1.5}' "Whiteness map L - 1.5*chroma: eggs are bright AND colourless, so penalising chroma should suppress bright coloured backgrounds (pink bowls, green leaves, skin)."
T T05 '{"feature":"whiteness","chroma_w":1.5,"open_k":7,"close_k":7,"fill_holes":true}' "Add morphology (open 7, close 7) + hole filling to remove speckle and fill specular/shadow holes inside eggs."
T T06 '{"feature":"whiteness","chroma_w":1.5,"open_k":7,"close_k":7,"fill_holes":true,"min_area":0.004,"max_area":0.6,"min_solidity":0.9,"min_ell_iou":0.8,"min_axis_ratio":0.4}' "Add shape filter on components (area, solidity, IoU with fitted ellipse, axis ratio): eggs are convex ellipses, background blobs are not."
T T07 '{"feature":"whiteness","chroma_w":1.5,"thresh":"sweep","open_k":7,"close_k":7,"fill_holes":true,"min_area":0.004,"max_area":0.6,"min_solidity":0.9,"min_ell_iou":0.8,"min_axis_ratio":0.4}' "Replace single Otsu with a 12-level threshold sweep (40th-97th percentile) and pick the most egg-shaped components across levels (greedy NMS by ellipse score). One global threshold cannot fit egg-on-white and egg-on-dark scenes alike."
# ---- Stage B: candidate scoring / selection (T07 = sweep is the new base) ----
S='"feature":"whiteness","chroma_w":1.5,"thresh":"sweep","open_k":7,"close_k":7,"fill_holes":true,"min_area":0.004,"max_area":0.6,"min_solidity":0.9,"min_ell_iou":0.8,"min_axis_ratio":0.4'
T T08 "{$S,\"min_contrast\":15}" "T07 + reject candidates not at least 15 grey-levels brighter than a 21px ring around them: many T07 false positives were flat elliptical background patches."
T T09 "{$S,\"score\":\"shape_contrast\"}" "T07 but rank candidates by shape score x min(1, contrast/40) so bright-against-background ellipses win the NMS over flat ones."
T T10 "{$S,\"rel_keep\":0.85}" "T07 + keep only candidates scoring >= 0.85 x the best one in the image (most images have 1-3 similar eggs; weaker extra candidates were mostly false positives)."
T T11 "{$S,\"area_w\":0.3}" "T07 + area bonus 0.3*sqrt(area) in the ranking: nested sweep candidates (highlight inside egg) are both elliptical; prefer the larger whole egg."
# ---- Stage C: combine winners, boundary refinement ----
T T12 "{$S,\"score\":\"shape_contrast\",\"rel_keep\":0.85}" "Combine the two kept Stage-B changes (T09 contrast-aware score + T10 relative keep)."
T T13 "{$S,\"score\":\"shape_contrast\",\"rel_keep\":0.85,\"ws_refine\":true}" "T12 + marker-controlled watershed refinement (fg = candidate eroded 0.5r, bg = outside candidate dilated 0.5r, flooded on the blurred colour image). Dev error maps showed the threshold keeps only the lit side of the egg; the shaded side is lost (low recall)."
T T14 "{$S,\"score\":\"shape_contrast\",\"rel_keep\":0.85,\"ws_refine\":true,\"area_w\":0.3}" "T13 + area bonus (T11) so the whole egg rather than its highlight seeds the refinement."
T T15 "{$S,\"score\":\"shape_contrast\",\"rel_keep\":0.85,\"grabcut\":true}" "T12 + GrabCut (GMM colour model + graph cut, 3 iters, init from candidate) instead of watershed, as an alternative boundary refinement."
# ---- Stage D: attack the total failures of T13 (23/88 dev images IoU<0.1: egg on white paper/box, ash-covered eggs, eggs in hand) ----
B="$S,\"score\":\"shape_contrast\",\"rel_keep\":0.85,\"ws_refine\":true"
T T16 "{$B,\"edge_cands\":true}" "T13 + extra candidates from closed Canny contours (auto thresholds, dilate 3, fill holes). On white paper/boxes the egg has the same whiteness as the background, but its outline is still a visible edge."
T T17 "{$B,\"feature\":\"whiteness_local\"}" "T13 with background-normalised whiteness (subtract 0.5 x Gaussian-blurred map, sigma 60): eggs on bright backgrounds and ash-covered eggs are locally brighter even if globally not."
T T18 "{$B,\"min_ell_iou\":0.85}" "T13 with stricter ellipse-fit threshold 0.8 -> 0.85: several wrong picks were roughly elliptical paper/cloth regions."
T T19 "{$B,\"chroma_w\":3.0}" "T13 with stronger chroma penalty 1.5 -> 3.0 to push coloured backgrounds (skin, pink bowls, leaves) further down."
T T20 "{$B,\"sweep_lo\":20,\"sweep_levels\":16}" "T13 with a wider/denser sweep (20th-97th percentile, 16 levels) so darker / ash-covered eggs get a threshold level too."
# ---- Stage E: combine Stage-D winners (T17 local whiteness, T18 ellipse 0.85, T16 edges) and tune the local normalisation ----
L="$B,\"feature\":\"whiteness_local\",\"min_ell_iou\":0.85"
T T21 "{$L}" "Combine T17 (local whiteness) + T18 (ellipse IoU >= 0.85)."
T T22 "{$L,\"edge_cands\":true}" "T21 + Canny closed-contour candidates (T16 gave +0.004 alone)."
T T23 "{$L,\"local_w\":0.8}" "T21 with stronger background subtraction (local_w 0.5 -> 0.8)."
T T24 "{$L,\"local_sigma\":30}" "T21 with a smaller background scale (sigma 60 -> 30 px), closer to egg size (r ~ 40-120 px)."
T T25 "{$L,\"local_sigma\":120}" "T21 with a larger background scale (sigma 60 -> 120 px)."
# ---- Stage F: final tuning around T22 ----
F="$B,\"feature\":\"whiteness_local\",\"edge_cands\":true"
T T26 "{$F}" "T22 but ellipse threshold back to 0.80 (T21 showed 0.85 adds nothing once local whiteness is used; looser may recover eggs with flattened outlines)."
T T27 "{$L,\"edge_cands\":true,\"rel_keep\":0.7}" "T22 with looser relative keep 0.85 -> 0.7 to recover second/third eggs in multi-egg images (instance recall is the weak point)."
T T28 "{$L,\"edge_cands\":true,\"wr_in\":0.35,\"wr_out\":0.7}" "T22 with a wider watershed band (inner 0.35r, outer 0.7r): lets the refinement reach further into the shaded side of large eggs."
# ---- Stage G: last checks around T28 ----
G="$L,\"edge_cands\":true,\"wr_in\":0.35,\"wr_out\":0.7"
T T29 "{$G,\"min_ell_iou\":0.8}" "T28 + looser ellipse threshold 0.8 (T26 was marginally better than T22 with it)."
T T30 "{$G,\"ellipse_fit\":true}" "T28 but replace each instance by its fitted ellipse before refinement: eggs are near-ellipses, so this should regularise ragged threshold outlines."
T T31 "{$G,\"watershed\":true,\"ws_ratio\":0.6}" "T28 + distance-transform watershed split of components that fail the shape test (touching eggs merge into one non-elliptical blob)."
T T32 "{$G,\"wr_in\":0.25,\"wr_out\":0.9}" "T28 with an even wider refinement band (inner 0.25r, outer 0.9r)."
