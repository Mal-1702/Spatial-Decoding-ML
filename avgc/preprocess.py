"""Signal views of one trial. Every operation here is label-free and applied
per trial, so no information can leak between train and test trials."""
import numpy as np
from scipy.signal import butter, sosfiltfilt

from . import config as C


def bandpass(x, band, fs=C.FS, order=4):
    sos = butter(order, band, btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def common_average(x):
    return x - x.mean(axis=1, keepdims=True)


def regress_out_eog(eeg, exg):
    """Least-squares removal of everything in the EEG that is linearly explained
    by the four EOG electrodes (fit within the trial, no labels involved)."""
    design = np.c_[exg, np.ones(len(exg))]
    coef, *_ = np.linalg.lstsq(design, eeg, rcond=None)
    return eeg - design @ coef
