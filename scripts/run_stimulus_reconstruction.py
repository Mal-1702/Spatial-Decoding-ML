"""Experiment 1 -- sanity check: is the neural signal intact?

Within-subject leave-one-trial-out linear stimulus reconstruction, with and
without EOG regression.  Output: results/stimrecon.csv
"""
import _common  # noqa: F401  (sets thread limits and import path)
import pandas as pd
from _common import parallel_map

from avgc import config as C, data, stimrecon as SR


def run_subject(path):
    sid, _, trials = data.load_subject(path)
    rows = []
    for eog_reg in (False, True):
        prepared = [SR.prepare(t, eog_regression=eog_reg) for t in trials]
        for k, tr in enumerate(trials):
            model = SR.fit([p for i, p in enumerate(prepared) if i != k])
            for L in C.WINDOWS_S:
                correct, n = SR.evaluate(model, *prepared[k], L)
                rows.append(dict(subject=sid, trial=k, condition=tr.condition,
                                 eog_regression=eog_reg, window_s=L, n_windows=n,
                                 n_correct=correct, ridge_alpha=model[2]))
    print(f"sub{sid} done", flush=True)
    return rows


if __name__ == "__main__":
    C.RESULTS_DIR.mkdir(exist_ok=True)
    out = parallel_map(run_subject, [(p,) for p in data.subject_files()])
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv(C.RESULTS_DIR / "stimrecon.csv", index=False)
    print(df.groupby(["eog_regression", "window_s"])[["n_correct", "n_windows"]].sum()
            .assign(acc=lambda d: d.n_correct / d.n_windows))
