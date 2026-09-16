# Are spatial auditory-attention decoders reading the eyes?

A gaze-confound analysis of EEG-based spatial auditory attention decoding on the
KU Leuven **AV-GC-AAD** dataset (Rotaru et al., 2024, *J. Neural Eng.* 21:016017;
Zenodo 10.5281/zenodo.11058711).

## Question

EEG decoders that predict *which direction* a listener attends often report high
accuracy. People usually look where they listen, and eye movements create large
electrical artifacts on the scalp. So is the decoder reading attention, or eye
position?

AV-GC-AAD lets us test this, because its four conditions change the relation between
gaze and attention:

| Condition | What the listener looks at | Gaze vs attended side |
|---|---|---|
| FixedVideo | video of the attended speaker, on the attended side | **confounded** |
| NoVisuals | black screen, central fixation | minimised |
| MovingVideo | video of the attended speaker moving randomly | **decorrelated** |
| MovingTargetNoise | cross-hair moving randomly, with background noise | **decorrelated** |

Each trial is 10 min. At 5 min the two speakers **swap sides**. The listener keeps
attending the same speaker, so the attended direction flips halfway through every
trial.

## Data facts verified in the files

* 128 Hz, 1-40 Hz Chebyshev band-pass, 50 Hz notch, unreferenced.
* 64 BioSemi EEG channels + EXG3-EXG6 (EOG).
* **HEOG = EXG5 − EXG6** and VEOG = EXG3 − EXG4. At the swap in FixedVideo,
  EXG5 − EXG6 steps by about ±10-15 µV, with opposite sign for L→R vs R→L.
* `initAttention` gives the attended side for 0-300 s. The side is opposite for 300-600 s.
* This copy contains **9 subjects** (01, 03, 04, 07-12). Subjects 01 and 03 have no
  MovingTargetNoise trials.

## Pipeline

```
avgc/                    library
  config.py              every setting in one place (nothing tuned on test data)
  data.py                loader for the .mat files, trial/block/label logic
  preprocess.py          referencing, band-pass, EOG regression, signal "views"
  features.py            non-overlapping windows -> OAS-shrunk covariance matrices
  models.py              Riemannian tangent space + LDA, CSP + LDA
  evaluation.py          trial-disjoint splits, decision windows, accuracy / AUROC
  stimrecon.py           ridge backward model (envelope reconstruction)
  stats.py               Wilcoxon, Holm correction
scripts/
  run_gaze_check.py              Exp 0  do the eyes follow attention?
  run_stimulus_reconstruction.py Exp 1  sanity check: is the neural signal intact?
  run_spatial_decoding.py        Exp 2  left/right decoding, leave-one-trial-out
                                 Exp 3  cross-condition transfer
  run_permutation.py             Exp 4  per-subject trial-level permutation tests
  run_leakage_controls.py        Exp 5  leaky CV vs trial-disjoint, first-vs-second-half decoding
  make_report.py                 figures/ + results/RESULTS.md
```

### Models (all linear, closed-form LDA with Ledoit-Wolf shrinkage)

| Key | Input | Purpose |
|---|---|---|
| `eog_ts` | 4 EOG channels only | **gaze baseline**: what eyes alone can do |
| `eeg_csp` | 64 EEG, CAR | classical BCI baseline |
| `eeg_ts` | 64 EEG, CAR | Riemannian tangent space (strong classical decoder) |
| `eeg_eogreg_ts` | EEG with EOG regressed out | gaze control 1 |
| `eeg_post_ts` | 29 posterior channels, own reference | gaze control 2 (far from the eyes; a whole-head CAR would spread eye artifacts into them) |
| `eeg_alpha_ts` | EEG 8-13 Hz | alpha lateralisation, a known neural marker of spatial attention |
| `gazectrl_ts` / `gazectrl_csp` | all three controls combined | best attempt at a gaze-free decoder |

### Evaluation rules

* **Trial-disjoint splits.** A test trial is never seen in training, not even the
  windows next to it. This is the most common mistake in the literature.
* Decision windows of 1, 2, 5, 10, 30 and 60 s. Classifiers train on windows of
  min(L, 5) s. Longer decisions average consecutive 5 s scores.
* No hyperparameter is tuned on test data. LDA shrinkage is analytic. Ridge strength
  for envelope decoding comes from an inner leave-one-trial-out on training trials.
* Results are reported **per subject and per condition**. Group tests are exact
  Wilcoxon signed-rank tests across subjects, with Holm correction.
* **Why no window-level binomial test:** consecutive windows from the same 5-min
  block are not independent, so a binomial test is anti-conservative. We permute
  labels at trial level instead (see `run_permutation.py`, which also documents the
  power limit of 2 trials per condition).

## Reproduce

```bash
pip install -r requirements.txt
cd scripts
python run_gaze_check.py
python run_stimulus_reconstruction.py
python run_spatial_decoding.py
python run_permutation.py
python run_leakage_controls.py
python make_report.py
```

On Windows, run these from PowerShell or cmd, not Git Bash. Python multiprocessing
fails under Git Bash.

## Results

Full tables are in `results/RESULTS.md` (generated) and figures are in `figures/`.
N = 9 subjects. All numbers are means across subjects unless stated otherwise.

### 1. The neural signal is intact (Fig 1)
Envelope reconstruction identifies the attended speaker with **69.5 %** accuracy at
10 s and **85.2 %** at 60 s. Every subject is above chance, Wilcoxon p = 0.002 at
every window length. Regressing out the EOG changes nothing (69.5 % / 86.6 %). The
recordings contain decodable attention information that does not depend on gaze.

### 2. Gaze follows attention only in FixedVideo (Fig 0)
At the side swap, HEOG moves toward the newly attended side by a median of
**+5.2 µV** in FixedVideo. The median is about 0 µV in NoVisuals, MovingVideo and
MovingTargetNoise. The conditions manipulate gaze as intended.
