"""Experiments 2 and 3 -- spatial (left/right) attention decoding.

pooled_loto      train on all other trials of the subject (every condition),
                 test on the held-out trial.
cross_condition  train on the trials of condition A, test on condition B.
                 A == B uses the other repetition of the same condition.
                 Training on FixedVideo (gaze follows attention) and testing on
                 MovingVideo / MovingTargetNoise (gaze decorrelated) is the key
                 test of whether a decoder relies on gaze.

Outputs: results/spatial_pooled.csv, results/spatial_crosscond.csv
"""
import _common  # noqa: F401  (sets thread limits and import path)
import numpy as np
import pandas as pd
from _common import parallel_map

from avgc import config as C, data, features, evaluation as E
from avgc.models import MODELS


def summarise_by_condition(dec_by_L, trials, meta):
    rows = []
    for L, dec in dec_by_L.items():
        conds = np.array([trials[t].condition for t in dec["trial"]])
        for cond in np.unique(conds):
            rows.append({**meta, "test_condition": cond, "window_s": L,
                         **E.summarise(dec[conds == cond])})
    return rows


def run_subject(path):
    sid, labels, trials = data.load_subject(path)
    ws = features.subject_windows(trials, labels, sorted({E.train_len(L) for L in C.WINDOWS_S}))
    idx = np.arange(len(trials))
    by_cond = {c: [i for i in idx if trials[i].condition == c] for c in C.CONDITIONS}
    by_cond = {c: v for c, v in by_cond.items() if v}
    pooled, cross = [], []
    for key in MODELS:
        # --- pooled leave-one-trial-out
        for k in idx:
            dec = E.run_split(key, ws, [i for i in idx if i != k], [k])
            pooled += summarise_by_condition(dec, trials, dict(subject=sid, model=key, test_trial=k))
        # --- cross-condition
        for a, a_trials in by_cond.items():
            others = [i for i in idx if trials[i].condition != a]
            dec = E.run_split(key, ws, a_trials, others)
            cross += summarise_by_condition(dec, trials, dict(subject=sid, model=key, train_condition=a))
            for k in a_trials:                     # same condition: other repetition
                dec = E.run_split(key, ws, [i for i in a_trials if i != k], [k])
                cross += summarise_by_condition(dec, trials, dict(subject=sid, model=key, train_condition=a))
        print(f"sub{sid} {key} done", flush=True)
    return pooled, cross


def collapse(df, keys):
    """Pool windows over test trials, then recompute accuracy (AUC is averaged)."""
    g = df.groupby(keys)
    out = g[["n_windows", "n_correct"]].sum()
    out["accuracy"] = out["n_correct"] / out["n_windows"]
    out["auc"] = g["auc"].mean()
    return out.reset_index()


if __name__ == "__main__":
    C.RESULTS_DIR.mkdir(exist_ok=True)
    res = parallel_map(run_subject, [(p,) for p in data.subject_files()])
    pooled = pd.DataFrame([r for p, _ in res for r in p])
    cross = pd.DataFrame([r for _, c in res for r in c])
    pooled.to_csv(C.RESULTS_DIR / "spatial_pooled_by_trial.csv", index=False)
    collapse(pooled, ["subject", "model", "test_condition", "window_s"]).to_csv(
        C.RESULTS_DIR / "spatial_pooled.csv", index=False)
    collapse(cross, ["subject", "model", "train_condition", "test_condition", "window_s"]).to_csv(
        C.RESULTS_DIR / "spatial_crosscond.csv", index=False)
    print("saved")
