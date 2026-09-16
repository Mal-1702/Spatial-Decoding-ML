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


def views(trial, eeg_labels):
    """Return {view_name: (T, n_channels) array} for the spatial decoders.

    EOG            4 EOG electrodes only                        -> gaze baseline
    EEG            64 EEG, common average reference, 1-40 Hz
    EEG_eogreg     EOG regressed out, then CAR
    EEG_posterior  29 posterior channels, referenced to their own mean
                   (a whole-head CAR would smear frontal eye artifacts into them)
    EEG_alpha      CAR, 8-13 Hz (alpha lateralisation band)
    EEG_gazectrl   EOG regression + posterior + alpha, all three controls combined
    """
    post = [eeg_labels.index(ch) for ch in C.POSTERIOR]
    clean = regress_out_eog(trial.eeg, trial.exg)
    return {
        "EOG": trial.exg,
        "EEG": common_average(trial.eeg),
        "EEG_eogreg": common_average(clean),
        "EEG_posterior": common_average(trial.eeg[:, post]),
        "EEG_alpha": bandpass(common_average(trial.eeg), C.ALPHA_BAND),
        "EEG_gazectrl": bandpass(common_average(clean[:, post]), C.ALPHA_BAND),
    }


def heog(trial):
    i, j = C.EXG.index(C.HEOG[0]), C.EXG.index(C.HEOG[1])
    return trial.exg[:, i] - trial.exg[:, j]
