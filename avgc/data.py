"""Loader for the KU Leuven AV-GC-AAD dataset (Rotaru et al., 2024).

File layout (verified on the released .mat files):
    data[i]            (76800, 68) float32 -- 64 EEG + EXG3..EXG6, 128 Hz, unreferenced
    conditionID[i]     e.g. 'MovingVideo1'
    initAttention[i]   'left' / 'right' -- side of the attended speaker for 0-300 s
    stimulus.attendedEnvelopes[i], .unattendedEnvelopes[i]  (76803,)
    metadata[i].FileHeader.Channels[k].Label

At 300 s the two speakers swap sides; the listener keeps attending the same
speaker, so the attended *direction* flips halfway through every trial.
"""
import re
from dataclasses import dataclass

import numpy as np
import scipy.io as sio

from . import config as C


@dataclass
class Trial:
    subject: str
    index: int              # position in the file (recording order)
    condition: str          # base condition name, e.g. 'MovingVideo'
    repetition: int         # 1 or 2
    init_side: str          # 'left' or 'right'
    eeg: np.ndarray         # (T, 64) float64, unreferenced
    exg: np.ndarray         # (T, 4)  float64
    env_att: np.ndarray     # (T,)
    env_unatt: np.ndarray   # (T,)

    @property
    def blocks(self):
        """[(start_sample, stop_sample, label)] with label 0 = left, 1 = right."""
        first = 0 if self.init_side == "left" else 1
        s, h, e = C.SKIP_S * C.FS, C.SWAP_S * C.FS, C.TRIAL_S * C.FS
        return [(s, h, first), (h + s, e, 1 - first)]


def subject_files():
    return sorted(C.DATA_DIR.glob("2024-AV-GC-AAD-sub*_preprocessed.mat"))


def subject_id(path):
    return re.search(r"sub(\d+)", path.name).group(1)


def load_subject(path):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    assert int(m["fs"]) == C.FS
    sid = subject_id(path)
    labels = [ch.Label for ch in m["metadata"][0].FileHeader.Channels]
    assert labels[C.N_EEG:] == C.EXG, labels[C.N_EEG:]
    n = C.TRIAL_S * C.FS
    trials = []
    for i, cond in enumerate(np.atleast_1d(m["conditionID"])):
        base, rep = re.match(r"([A-Za-z]+)(\d)", cond).groups()
        x = np.asarray(m["data"][i], dtype=np.float64)
        assert x.shape == (n, C.N_EEG + 4), x.shape
        trials.append(Trial(
            subject=sid, index=i, condition=base, repetition=int(rep),
            init_side=str(m["initAttention"][i]),
            eeg=x[:, :C.N_EEG], exg=x[:, C.N_EEG:],
            env_att=np.asarray(m["stimulus"].attendedEnvelopes[i][:n], dtype=np.float64),
            env_unatt=np.asarray(m["stimulus"].unattendedEnvelopes[i][:n], dtype=np.float64),
        ))
    return sid, labels[:C.N_EEG], trials
