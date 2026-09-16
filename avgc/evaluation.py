"""Trial-disjoint evaluation.

A *trial* is one 10-min recording (two 5-min blocks with opposite attended
sides).  Train and test sets never share a trial, so no window of a held-out
recording, nor its immediate neighbours, is ever seen during training.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as C
from .models import MODELS, build


def train_len(L):
    return min(L, C.TRAIN_MAX_S)


def fit_predict(key, ws, train_trials, test_trials, labels=None):
    """Fit on windows from train_trials, return decision scores on test windows."""
    view = MODELS[key][0]
    y = ws.label if labels is None else labels
    tr = np.isin(ws.trial, train_trials)
    te = np.isin(ws.trial, test_trials)
    model = build(key).fit(ws.covs[view][tr], y[tr])
    return model.decision_function(ws.covs[view][te]), te


def to_decisions(ws_by_len, scores_by_len, masks_by_len, L):
    """Decision windows of length L: direct scores if L <= TRAIN_MAX_S,
    otherwise the mean score of consecutive TRAIN_MAX_S sub-windows."""
    tl = train_len(L)
    ws, sc, te = ws_by_len[tl], scores_by_len[tl], masks_by_len[tl]
    df = pd.DataFrame({"trial": ws.trial[te], "block": ws.block[te],
                       "label": ws.label[te], "pos": ws.pos[te], "score": sc})
    if L == tl:
        return df
    group = L // tl
    df["dec"] = df["pos"] // group
    agg = (df.groupby(["trial", "block", "dec"])
             .agg(label=("label", "first"), score=("score", "mean"), n=("score", "size"))
             .reset_index())
    return agg[agg["n"] == group]          # drop incomplete windows at block ends
