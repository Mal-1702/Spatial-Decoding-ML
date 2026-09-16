"""Project-wide constants. Every analysis reads its settings from here so that
nothing is tuned by looking at test results."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Datasets"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
CACHE_DIR = ROOT / "cache"

FS = 128                    # sampling rate of the released data (Hz)
TRIAL_S = 600               # every trial is 10 min
SWAP_S = 300                # speakers swap sides at 5 min (listener keeps the same speaker)
SKIP_S = 2                  # seconds dropped at the start of each block (swap transient)

N_EEG = 64
EXG = ["EXG3", "EXG4", "EXG5", "EXG6"]   # EOG electrodes (columns 64..67)
# Verified from the data: at the side swap in FixedVideo, EXG5-EXG6 steps by
# ~ +/-10-15 uV with opposite sign for L->R vs R->L, so it is horizontal EOG.
HEOG = ("EXG5", "EXG6")
VEOG = ("EXG3", "EXG4")

# Condition names in the files carry a repetition suffix (MovingVideo1, ...).
CONDITIONS = ["FixedVideo", "NoVisuals", "MovingVideo", "MovingTargetNoise"]
GAZE_RELATION = {
    "FixedVideo": "gaze follows attention",
    "NoVisuals": "central fixation",
    "MovingVideo": "gaze decorrelated",
    "MovingTargetNoise": "gaze decorrelated",
}

# Posterior (parietal / occipital / centro-parietal) channels, far from the eyes.
POSTERIOR = ["CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "CPz", "TP7", "TP8",
             "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "Pz",
             "PO3", "PO4", "PO7", "PO8", "POz", "O1", "O2", "Oz", "Iz"]

BROAD_BAND = (1.0, 40.0)    # as released
ALPHA_BAND = (8.0, 13.0)

# Decision-window lengths (s). Classifiers are trained on windows of
# min(L, TRAIN_MAX_S) seconds; longer decisions average the scores of
# consecutive TRAIN_MAX_S sub-windows.
WINDOWS_S = [1, 2, 5, 10, 30, 60]
TRAIN_MAX_S = 5

# Stimulus reconstruction (backward model) settings, following common AAD practice.
SR_BAND = (1.0, 9.0)
SR_FS = 32
SR_LAGS_MS = (0, 250)
SR_ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]

RANDOM_SEED = 0
