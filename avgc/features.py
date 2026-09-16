"""Cut trials into non-overlapping windows inside each attention block and
estimate one covariance matrix per window."""
from dataclasses import dataclass

import numpy as np
from pyriemann.utils.covariance import covariances

from . import config as C
from .preprocess import views


@dataclass
class WindowSet:
    covs: dict            # view -> (n_windows, ch, ch)
    trial: np.ndarray     # index into the subject's trial list
    block: np.ndarray     # 0 = before swap, 1 = after swap
    label: np.ndarray     # 0 = attend left, 1 = attend right
    pos: np.ndarray       # window number inside its block


def block_windows(trial, win_s):
    """Yield (start_sample, block_id, label, position) for every full window."""
    w = int(win_s * C.FS)
    for b, (start, stop, label) in enumerate(trial.blocks):
        for k in range((stop - start) // w):
            yield start + k * w, b, label, k


def subject_windows(trials, eeg_labels, win_lengths):
    """Covariances for every view and every training window length."""
    out = {w: {"covs": {}, "trial": [], "block": [], "label": [], "pos": []}
           for w in win_lengths}
    for ti, tr in enumerate(trials):
        sig = views(tr, eeg_labels)
        for w in win_lengths:
            idx = list(block_windows(tr, w))
            starts = np.array([i[0] for i in idx])
            n = int(w * C.FS)
            for name, x in sig.items():
                segs = np.stack([x[s:s + n].T for s in starts])      # (n_win, ch, n)
                out[w]["covs"].setdefault(name, []).append(
                    covariances(segs, estimator="oas").astype(np.float64))
            out[w]["trial"] += [ti] * len(idx)
            out[w]["block"] += [i[1] for i in idx]
            out[w]["label"] += [i[2] for i in idx]
            out[w]["pos"] += [i[3] for i in idx]
    return {w: WindowSet(covs={k: np.concatenate(v) for k, v in d["covs"].items()},
                         trial=np.array(d["trial"]), block=np.array(d["block"]),
                         label=np.array(d["label"]), pos=np.array(d["pos"]))
            for w, d in out.items()}
