"""Experiment 0 -- does gaze actually follow attention in each condition?

At the side swap (300 s) the attended speaker jumps to the other side.  If the
eyes follow, horizontal EOG shows a step at that moment.  We measure the HEOG
change (0.2-1.0 s after vs 2-0 s before the swap), signed so that a positive
value means the eyes moved toward the newly attended side.
Output: results/gaze_check.csv
"""
import _common  # noqa: F401  (sets thread limits and import path)
import pandas as pd

from avgc import config as C, data
from avgc.preprocess import heog

# Sign verified on the data: a left->right gaze shift makes EXG5-EXG6 negative.
TOWARD_RIGHT = -1


def run_subject(path):
    sid, _, trials = data.load_subject(path)
    rows = []
    for k, tr in enumerate(trials):
        h, s = heog(tr), C.SWAP_S * C.FS
        step = h[s + int(0.2 * C.FS):s + C.FS].mean() - h[s - 2 * C.FS:s].mean()
        new_side_right = tr.init_side == "left"
        toward = step * TOWARD_RIGHT * (1 if new_side_right else -1)
        rows.append(dict(subject=sid, trial=k, condition=tr.condition,
                         init_side=tr.init_side, heog_step_uv=step,
                         shift_toward_attended_uv=toward))
    return rows


if __name__ == "__main__":
    C.RESULTS_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame([r for p in data.subject_files() for r in run_subject(p)])
    df.to_csv(C.RESULTS_DIR / "gaze_check.csv", index=False)
    print(df.groupby("condition")["shift_toward_attended_uv"].describe().round(2))
