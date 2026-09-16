"""Spatial (left vs right) attention decoders.

All classifiers are linear discriminant analysis with Ledoit-Wolf shrinkage:
closed-form, no hyperparameter to tune, so nothing can be tuned on test data.

ts   Riemannian tangent space: each window covariance is projected onto the
     tangent plane at the mean covariance of the training windows, giving a
     vector of ch*(ch+1)/2 features.  The projection uses no labels.
csp  Common spatial patterns (6 filters, log-variance) -- the classical BCI
     baseline.  CSP uses labels, so it is refit inside every training fold.
"""
from pyriemann.spatialfilters import CSP
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline

# key -> (signal view, method, readable name)
MODELS = {
    "eog_ts":       ("EOG",           "ts",  "EOG only (gaze baseline)"),
    "eeg_csp":      ("EEG",           "csp", "EEG CSP+LDA"),
    "eeg_ts":       ("EEG",           "ts",  "EEG Riemann"),
    "eeg_eogreg_ts": ("EEG_eogreg",   "ts",  "EEG Riemann, EOG regressed"),
    "eeg_post_ts":  ("EEG_posterior", "ts",  "EEG Riemann, posterior channels"),
    "eeg_alpha_ts": ("EEG_alpha",     "ts",  "EEG Riemann, alpha band"),
    "gazectrl_ts":  ("EEG_gazectrl",  "ts",  "Gaze-controlled EEG Riemann"),
    "gazectrl_csp": ("EEG_gazectrl",  "csp", "Gaze-controlled CSP+LDA"),
}


def lda():
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def tangent_space():
    # Log-Euclidean reference mean (closed form, fast) with the affine-invariant map.
    return TangentSpace(metric={"mean": "logeuclid", "map": "riemann"})


def build(key):
    _, method, _ = MODELS[key]
    if method == "ts":
        return make_pipeline(tangent_space(), lda())
    return make_pipeline(CSP(nfilter=6, log=True, metric="euclid"), lda())
