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
