"""Linear stimulus reconstruction (backward model) -- the standard AAD sanity check.

A ridge regression maps time-lagged EEG (0-250 ms after the stimulus) to the
speech envelope.  In a test window we reconstruct the envelope and pick the
speaker whose real envelope correlates best with it.  This decoder uses the
*content* of the speech, not its direction, so eye position cannot help it.
"""
import numpy as np

from . import config as C
from .preprocess import bandpass, common_average, regress_out_eog


def prepare(trial, eog_regression=False):
    eeg = regress_out_eog(trial.eeg, trial.exg) if eog_regression else trial.eeg
    step = C.FS // C.SR_FS
    x = bandpass(common_average(eeg), C.SR_BAND)[::step]
    a = bandpass(trial.env_att, C.SR_BAND)[::step]
    u = bandpass(trial.env_unatt, C.SR_BAND)[::step]
    return lag(x), a, u


def lag(x):
    """Row t holds EEG samples t+l for l in the lag range (EEG follows the stimulus)."""
    l0, l1 = (int(round(ms / 1000 * C.SR_FS)) for ms in C.SR_LAGS_MS)
    T, ch = x.shape
    out = np.zeros((T, ch * (l1 - l0 + 1)))
    for j, l in enumerate(range(l0, l1 + 1)):
        out[:T - l, j * ch:(j + 1) * ch] = x[l:]
    return out
