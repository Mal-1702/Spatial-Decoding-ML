"""Experiment 5 -- why do trial-disjoint accuracies sit *below* chance?

Hypothesis: the EEG drifts slowly over a 10-min recording (electrode impedance,
fatigue, arousal).  Inside a trial, "attend left" and "attend right" occupy the
first and second half, so a classifier can separate them using time alone.
The two repetitions of each condition start on opposite sides, so drift learnt
on one repetition predicts the *wrong* side on the other -> below chance.

Two controls on 5 s windows:
  leaky_random_cv   5-fold CV over randomly shuffled windows, ignoring trials
                    (the evaluation mistake common in the literature).
  time_decoding     trial-disjoint leave-one-trial-out, but the label is
                    "first vs second half" instead of the attended side.
Output: results/leakage_controls.csv
"""
import _common  # noqa: F401  (sets thread limits and import path)
import numpy as np
import pandas as pd
from _common import parallel_map
from sklearn.model_selection import StratifiedKFold

from avgc import config as C, data, features
from avgc.models import MODELS, build

WIN_S = 5
KEYS = ["eog_ts", "eeg_csp", "eeg_ts", "gazectrl_ts"]


def accuracy_rows(sid, key, analysis, ws, trials, scores, labels):
    conds = np.array([trials[t].condition for t in ws.trial])
    pred = (scores > 0).astype(int)
    return [dict(subject=sid, model=key, analysis=analysis, condition=c,
                 accuracy=float((pred[conds == c] == labels[conds == c]).mean()))
            for c in np.unique(conds)]


def run_subject(path):
    sid, labels, trials = data.load_subject(path)
    ws = features.subject_windows(trials, labels, [WIN_S])[WIN_S]
    rows = []
    for key in KEYS:
        X = ws.covs[MODELS[key][0]]
        # leaky: windows of the same block land in train and test
        scores = np.zeros(len(ws.label))
        skf = StratifiedKFold(5, shuffle=True, random_state=C.RANDOM_SEED)
        for tr, te in skf.split(X, ws.label):
            scores[te] = build(key).fit(X[tr], ws.label[tr]).decision_function(X[te])
        rows += accuracy_rows(sid, key, "leaky_random_cv", ws, trials, scores, ws.label)
        # time decoding, trial-disjoint
        scores = np.zeros(len(ws.block))
        for k in range(len(trials)):
            tr, te = ws.trial != k, ws.trial == k
            scores[te] = build(key).fit(X[tr], ws.block[tr]).decision_function(X[te])
        rows += accuracy_rows(sid, key, "time_decoding_loto", ws, trials, scores, ws.block)
        # reference: attended side, trial-disjoint (same 5 s windows)
        scores = np.zeros(len(ws.label))
        for k in range(len(trials)):
            tr, te = ws.trial != k, ws.trial == k
            scores[te] = build(key).fit(X[tr], ws.label[tr]).decision_function(X[te])
        rows += accuracy_rows(sid, key, "side_decoding_loto", ws, trials, scores, ws.label)
        print(f"sub{sid} {key} done", flush=True)
    return rows


if __name__ == "__main__":
    out = parallel_map(run_subject, [(p,) for p in data.subject_files()], n_jobs=4)
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv(C.RESULTS_DIR / "leakage_controls.csv", index=False)
    print(df.pivot_table(index=["analysis", "model"], columns="condition",
                         values="accuracy", aggfunc="mean").round(3))
