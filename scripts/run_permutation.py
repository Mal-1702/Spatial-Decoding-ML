"""Experiment 4 -- per-subject significance by trial-level label permutation.

Why not a binomial test?  Windows from the same 5-min block are strongly
correlated, so treating them as independent coin flips gives p-values that are
far too small.

Null model: the attended side carries no decodable information.  Under the
null, the assignment "left first" vs "right first" of every 10-min trial is
exchangeable, so we flip it at random for each trial (train and test labels
alike) and rerun the full leave-one-trial-out analysis.  This keeps every
temporal dependency in the data -- including slow drifts from the first to
the second half of a trial -- which a window-level shuffle would destroy.

Power limitation (a property of the dataset, not of the code): each condition
has only 2 trials per subject.  A permutation that flips both trials of a
condition together leaves a condition-specific decoder's accuracy unchanged,
so the smallest reachable per-condition p-value is roughly 0.25.  We therefore
also test each subject over all conditions pooled ("ALL", 6-8 trials), and use
across-subject Wilcoxon tests for per-condition claims.

Decision window: 10 s.  Output: results/permutation.csv
"""
import _common  # noqa: F401  (sets thread limits and import path)
import sys

import numpy as np
import pandas as pd
from _common import parallel_map

from avgc import config as C, data, features, evaluation as E
from avgc.models import MODELS, build, lda, tangent_space

WINDOW_S = 10
N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 100


def run_subject(path, seed):
    sid, labels, trials = data.load_subject(path)
    tl = E.train_len(WINDOW_S)
    ws = features.subject_windows(trials, labels, [tl])[tl]
    idx = np.arange(len(trials))
    conds = np.array([t.condition for t in trials])
    rng = np.random.default_rng(seed)
    flips = np.vstack([np.zeros(len(idx), int), rng.integers(0, 2, (N_PERM, len(idx)))])

    rows = []
    for key, (view, method, _) in MODELS.items():
        covs = ws.covs[view]
        # label-free tangent-space features: computed once per fold
        feats = {}
        if method == "ts":
            for k in idx:
                tr = ws.trial != k
                ts = tangent_space().fit(covs[tr])
                feats[k] = ts.transform(covs)
        for p, flip in enumerate(flips):
            y = ws.label ^ flip[ws.trial]
            scores = np.zeros(len(y))
            for k in idx:
                tr, te = ws.trial != k, ws.trial == k
                if method == "ts":
                    clf = lda().fit(feats[k][tr], y[tr])
                    scores[te] = clf.decision_function(feats[k][te])
                else:
                    clf = build(key).fit(covs[tr], y[tr])
                    scores[te] = clf.decision_function(covs[te])
            ws_p = features.WindowSet(ws.covs, ws.trial, ws.block, y, ws.pos)
            dec = E.to_decisions({tl: ws_p}, {tl: scores}, {tl: np.ones(len(y), bool)}, WINDOW_S)
            dec_cond = conds[dec["trial"].to_numpy()]
            for c in [*np.unique(conds), "ALL"]:
                sel = dec if c == "ALL" else dec[dec_cond == c]
                rows.append(dict(subject=sid, model=key, condition=c, perm=p,
                                 accuracy=E.summarise(sel)["accuracy"]))
        print(f"sub{sid} {key} done", flush=True)
    return rows


if __name__ == "__main__":
    files = data.subject_files()
    out = parallel_map(run_subject, [(f, C.RANDOM_SEED + i) for i, f in enumerate(files)])
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv(C.RESULTS_DIR / "permutation_null.csv", index=False)
    obs = df[df.perm == 0].set_index(["subject", "model", "condition"])["accuracy"]
    null = df[df.perm > 0]
    pvals = (null.join(obs.rename("observed"), on=["subject", "model", "condition"])
                 .assign(ge=lambda d: d.accuracy >= d.observed - 1e-12)
                 .groupby(["subject", "model", "condition"])
                 .agg(n_perm=("ge", "size"), n_ge=("ge", "sum"),
                      null_mean=("accuracy", "mean"), null_95=("accuracy", lambda a: np.quantile(a, 0.95))))
    pvals["observed"] = obs
    pvals["p_value"] = (1 + pvals["n_ge"]) / (1 + pvals["n_perm"])
    pvals.reset_index().to_csv(C.RESULTS_DIR / "permutation.csv", index=False)
    print(pvals.groupby(["model", "condition"])["p_value"].apply(lambda p: (p < 0.05).sum()))
