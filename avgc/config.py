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
