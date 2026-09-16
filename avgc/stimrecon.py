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


def _stats(X, y):
    return X.T @ X, X.T @ y, X.sum(0), y.sum(), len(y)


def _solve(parts, alpha):
    XtX = sum(p[0] for p in parts); Xty = sum(p[1] for p in parts)
    sx = sum(p[2] for p in parts); sy = sum(p[3] for p in parts); n = sum(p[4] for p in parts)
    mx, my = sx / n, sy / n
    cov = XtX - n * np.outer(mx, mx)
    lam = alpha * np.trace(cov) / cov.shape[0]
    w = np.linalg.solve(cov + lam * np.eye(len(cov)), Xty - n * mx * my)
    return w, mx


def _corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.sqrt((a @ a) * (b @ b)) + 1e-12))


def fit(train):
    """train: list of (X, att, unatt).  Ridge strength chosen by inner
    leave-one-trial-out on the training trials only."""
    parts = [_stats(X, a) for X, a, _ in train]
    best, best_r = C.SR_ALPHAS[0], -np.inf
    if len(train) > 1:
        for alpha in C.SR_ALPHAS:
            rs = []
            for k, (X, a, _) in enumerate(train):
                w, mx = _solve(parts[:k] + parts[k + 1:], alpha)
                rs.append(_corr((X - mx) @ w, a))
            if np.mean(rs) > best_r:
                best, best_r = alpha, np.mean(rs)
    w, mx = _solve(parts, best)
    return w, mx, best


def evaluate(model, X, att, unatt, L):
    """Fraction of non-overlapping L-second windows where the attended
    envelope correlates better with the reconstruction."""
    w, mx, _ = model
    rec = (X - mx) @ w
    n = int(L * C.SR_FS)
    lag_max = int(round(C.SR_LAGS_MS[1] / 1000 * C.SR_FS))
    usable = len(rec) - lag_max                    # last rows have zero-padded lags
    wins = [(rec[s:s + n], att[s:s + n], unatt[s:s + n]) for s in range(0, usable - n + 1, n)]
    correct = sum(_corr(r, a) > _corr(r, u) for r, a, u in wins)
    return correct, len(wins)
