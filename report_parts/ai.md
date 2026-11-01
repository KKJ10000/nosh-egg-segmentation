# AI tools disclosure

I used AI coding tools (Hark, an AI assistant) on this assignment:
- **Implementation:** most of the Python code was written with Hark. That covers the SAM wrapper and egg filter, the split script, the classical pipeline with its configurable stages, the metrics, the tracker, the plotting and the PDF builder.
- **Report drafting:** Hark drafted the text of this report and the README.
- **Not AI-generated:** every number in the tables and figures comes from scripts in the repository, run on the data described here. The experiment log is the raw output of `run_trial.py` (`results/experiment_log.csv`), including the trials that made things worse.

I have reviewed the methods and can explain them and change them in the interview. Those are the Otsu and multi-level thresholding, the Lab whiteness map, the morphology, the ellipse-fit shape scoring, the distance-transform watershed, GrabCut, Hungarian assignment with a Kalman filter, and the boundary F-score.

# Reproducing

See README.md. In short:
- `scripts/00_download.sh` gets the data, video and weights.
- `scripts/01_task1_sam.sh` runs SAM.
- `scripts/02_task2_split_and_trials.sh` does the split and all the trials.
- `scripts/03_task3_evaluate.sh` runs the evaluation.
- `scripts/04_task4_track.sh` runs the tracking.
- `src/make_report.py` builds this PDF.

Code: https://github.com/KKJ10000/nosh-egg-segmentation
