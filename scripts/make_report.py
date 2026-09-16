"""Statistics, figures and an auto-generated results report.

Reads results/*.csv, writes figures/*.png and results/RESULTS.md.
"""
import _common  # noqa: F401  (sets thread limits and import path)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from avgc import config as C, stats as S
from avgc.models import MODELS

# ---- visual system (validated reference palette, light surface) -----------------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
DIVERGING = LinearSegmentedColormap.from_list("acc", [ORANGE, "#f0efec", BLUE])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 5,
})
NAME = {k: v[2] for k, v in MODELS.items()}
MAIN = [("eog_ts", ORANGE), ("eeg_ts", BLUE), ("gazectrl_ts", AQUA)]
L_MAIN = 10


def save(fig, name):
    C.FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(C.FIG_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def chance_line(ax):
    ax.axhline(0.5, color=INK2, lw=1, ls="--", zorder=1)


def conditions_present(df, col):
    return [c for c in C.CONDITIONS if c in set(df[col])]


# ---- Experiment 0: gaze check --------------------------------------------------------
def fig_gaze(gz):
    fig, ax = plt.subplots(figsize=(6.6, 3.3), layout="constrained")
    conds = conditions_present(gz, "condition")
    rng = np.random.default_rng(0)
    for i, c in enumerate(conds):
        v = gz.loc[gz.condition == c, "shift_toward_attended_uv"]
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=14, color=BLUE, alpha=0.7, lw=0)
        ax.plot([i - 0.25, i + 0.25], [v.median()] * 2, color=INK, lw=2)
    ax.axhline(0, color=INK2, lw=1, ls="--")
    ax.set_xticks(range(len(conds)), [f"{c}\n{C.GAZE_RELATION[c]}" for c in conds], fontsize=8)
    ax.set_ylabel("HEOG shift toward newly\nattended side at swap (µV)")
    ax.set_title("Do the eyes follow the attended speaker?")
    save(fig, "fig0_gaze_check.png")


# ---- Experiment 1: stimulus reconstruction ---------------------------------------------
def stimrecon_table(sr):
    per_subj = (sr.groupby(["subject", "condition", "eog_regression", "window_s"])
                  [["n_correct", "n_windows"]].sum()
                  .assign(accuracy=lambda d: d.n_correct / d.n_windows).reset_index())
    return per_subj


def fig_stimrecon(ps):
    conds = conditions_present(ps, "condition")
    fig, axes = plt.subplots(1, len(conds), figsize=(2.6 * len(conds), 2.8), sharey=True, layout="constrained")
    for ax, c in zip(axes, conds):
        for reg, color, label in [(False, BLUE, "EEG"), (True, AQUA, "EEG, EOG regressed")]:
            d = ps[(ps.condition == c) & (ps.eog_regression == reg)]
            for _, s in d.groupby("subject"):
                ax.plot(s.window_s, s.accuracy, color=color, lw=0.6, alpha=0.25)
            m = d.groupby("window_s").accuracy.mean()
            ax.plot(m.index, m.values, color=color, marker="o", label=label)
        chance_line(ax)
        ax.set_xscale("log"); ax.set_xticks(C.WINDOWS_S, C.WINDOWS_S)
        ax.set_title(c); ax.set_xlabel("Decision window (s)")
    axes[0].set_ylabel("Accuracy (attended vs unattended)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Experiment 1 - envelope decoding works in every condition (thin lines = subjects)",
                 fontsize=10, x=0.01, ha="left")
    save(fig, "fig1_stimulus_reconstruction.png")


# ---- Experiment 2: pooled spatial decoding ----------------------------------------------
def fig_spatial_vs_window(sp):
    conds = conditions_present(sp, "test_condition")
    fig, axes = plt.subplots(1, len(conds), figsize=(2.6 * len(conds), 2.9), sharey=True, layout="constrained")
    for ax, c in zip(axes, conds):
        for key, color in MAIN:
            d = sp[(sp.test_condition == c) & (sp.model == key)]
            g = d.groupby("window_s").accuracy
            m, se = g.mean(), g.std() / np.sqrt(g.count())
            ax.fill_between(m.index, m - se, m + se, color=color, alpha=0.15, lw=0)
            ax.plot(m.index, m.values, color=color, marker="o", label=NAME[key])
        chance_line(ax)
        ax.set_xscale("log"); ax.set_xticks(C.WINDOWS_S, C.WINDOWS_S)
        ax.set_title(f"{c}\n{C.GAZE_RELATION[c]}"); ax.set_xlabel("Decision window (s)")
    axes[0].set_ylabel("Left/right accuracy (mean ± SEM)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.suptitle("Experiment 2 - spatial decoding, leave-one-trial-out", fontsize=10, x=0.01, ha="left")
    save(fig, "fig2_spatial_vs_window.png")


def fig_per_subject(sp, perm):
    conds = conditions_present(sp, "test_condition")
    keys = list(MODELS)
    fig, axes = plt.subplots(1, len(conds), figsize=(3.0 * len(conds), 3.6), sharey=True, layout="constrained")
    rng = np.random.default_rng(1)
    for ax, c in zip(axes, conds):
        d = sp[(sp.test_condition == c) & (sp.window_s == L_MAIN)]
        for i, k in enumerate(keys):
            v = d.loc[d.model == k, "accuracy"]
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=12, color=BLUE, alpha=0.6, lw=0)
            ax.plot([i - 0.3, i + 0.3], [v.mean()] * 2, color=INK, lw=2)
        chance_line(ax)
        ax.set_xticks(range(len(keys)), [NAME[k] for k in keys], rotation=60, ha="right", fontsize=7)
        ax.set_title(f"{c}\n{C.GAZE_RELATION[c]}")
    axes[0].set_ylabel(f"Accuracy at {L_MAIN} s")
    fig.suptitle("Every subject, every model (dot = subject, bar = mean)", fontsize=10, x=0.01, ha="left")
    save(fig, "fig4_per_subject.png")


# ---- Experiment 3: cross-condition ------------------------------------------------------
def fig_crosscond(cc):
    keys = ["eog_ts", "eeg_csp", "eeg_ts", "gazectrl_ts"]
    conds = conditions_present(cc, "train_condition")
    fig, axes = plt.subplots(1, len(keys), figsize=(3.3 * len(keys), 3.4), layout="constrained")
    for ax, k in zip(axes, keys):
        d = cc[(cc.model == k) & (cc.window_s == L_MAIN)]
        M = (d.groupby(["train_condition", "test_condition"]).accuracy.mean()
               .unstack().reindex(index=conds, columns=conds))
        im = ax.imshow(M.values, cmap=DIVERGING, vmin=0.25, vmax=0.75)
        for i in range(len(conds)):
            for j in range(len(conds)):
                if not np.isnan(M.values[i, j]):
                    ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=8, color=INK)
        short = [c.replace("MovingTargetNoise", "MovTgtNoise") for c in conds]
        ax.set_xticks(range(len(conds)), short, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(conds)), short, fontsize=7)
        ax.grid(False); ax.set_title(NAME[k], fontsize=9)
        ax.set_xlabel("Test condition")
    axes[0].set_ylabel("Train condition")
    cb = fig.colorbar(im, ax=axes, shrink=0.8)
    cb.set_label("Mean accuracy (0.5 = chance)")
    fig.suptitle(f"Experiment 3 - train on one condition, test on another ({L_MAIN} s windows)",
                 fontsize=10, x=0.01, ha="left")
    save(fig, "fig3_cross_condition.png")


# ---- Experiment 5: leakage and drift controls ------------------------------------------
LEAK_ANALYSES = [("leaky_random_cv", "Random window CV\n(leaky: ignores trials)"),
                 ("side_decoding_loto", "Attended side\n(trial-disjoint)"),
                 ("time_decoding_loto", "First vs second half\n(trial-disjoint)")]
LEAK_KEYS = ["eog_ts", "eeg_csp", "eeg_ts", "gazectrl_ts"]


def leakage_per_subject(lk):
    return lk.groupby(["analysis", "model", "subject"]).accuracy.mean().reset_index()


def fig_leakage(ls):
    fig, axes = plt.subplots(1, len(LEAK_ANALYSES), figsize=(10, 3.4), sharey=True, layout="constrained")
    rng = np.random.default_rng(2)
    for ax, (a, title) in zip(axes, LEAK_ANALYSES):
        for i, k in enumerate(LEAK_KEYS):
            v = ls[(ls.analysis == a) & (ls.model == k)].accuracy
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=14, color=BLUE, alpha=0.65, lw=0)
            ax.plot([i - 0.28, i + 0.28], [v.mean()] * 2, color=INK, lw=2)
            ax.text(i, 0.86, f"{v.mean():.2f}", ha="center", fontsize=8, color=INK2)
        chance_line(ax)
        ax.set_ylim(0.3, 0.9)
        ax.set_xticks(range(len(LEAK_KEYS)), [NAME[k] for k in LEAK_KEYS], rotation=35, ha="right", fontsize=7)
        ax.set_title(title)
    axes[0].set_ylabel("Accuracy, 5 s windows")
    fig.suptitle("Experiment 5 - leaky splits look good because EEG drifts over a trial "
                 "(dot = subject, bar = mean)", fontsize=10, x=0.01, ha="left")
    save(fig, "fig5_leakage_and_drift.png")


def leakage_table(ls):
    t = (ls.groupby(["analysis", "model"]).accuracy
           .agg(mean="mean", sd="std", min="min", max="max").reset_index())
    t["p_vs_chance"] = [S.vs_chance(ls[(ls.analysis == a) & (ls.model == m)].accuracy)
                        for a, m in zip(t.analysis, t.model)]
    t["model"] = t.model.map(NAME)
    return t


# ---- tables ------------------------------------------------------------------------------
def group_table(sp, L):
    rows = []
    for k in MODELS:
        for c in conditions_present(sp, "test_condition"):
            d = sp[(sp.model == k) & (sp.test_condition == c) & (sp.window_s == L)].sort_values("subject")
            eog = sp[(sp.model == "eog_ts") & (sp.test_condition == c) & (sp.window_s == L)].sort_values("subject")
            rows.append(dict(model=NAME[k], condition=c, n=len(d),
                             mean_acc=d.accuracy.mean(), sd=d.accuracy.std(),
                             min=d.accuracy.min(), max=d.accuracy.max(), mean_auc=d.auc.mean(),
                             p_vs_chance=S.vs_chance(d.accuracy),
                             p_vs_eog=np.nan if k == "eog_ts" else S.paired(d.accuracy.values, eog.accuracy.values)))
    t = pd.DataFrame(rows)
    t["p_holm"] = S.holm(t.p_vs_chance)
    return t


def cross_table(cc, L):
    rows = []
    decor = [c for c in ["MovingVideo", "MovingTargetNoise"] if c in set(cc.test_condition)]
    for k in MODELS:
        d = cc[(cc.model == k) & (cc.window_s == L)]
        same = d[(d.train_condition == "FixedVideo") & (d.test_condition == "FixedVideo")]
        to_dec = (d[(d.train_condition == "FixedVideo") & d.test_condition.isin(decor)]
                    .groupby("subject").accuracy.mean())
        rows.append(dict(model=NAME[k],
                         FixedVideo_to_FixedVideo=same.accuracy.mean(),
                         FixedVideo_to_gaze_decorrelated=to_dec.mean(),
                         p_transfer_vs_chance=S.vs_chance(to_dec)))
    return pd.DataFrame(rows)


def main():
    R = C.RESULTS_DIR
    md = ["# Results (auto-generated by scripts/make_report.py)", "",
          "All numbers are across subjects; every test uses N = number of subjects.",
          "Splits are trial-disjoint (a trial = one 10-min recording).", ""]

    gz = pd.read_csv(R / "gaze_check.csv", dtype={"subject": str})
    fig_gaze(gz)
    g = gz.groupby("condition").shift_toward_attended_uv.agg(["median", "mean", "std", "count"]).reset_index()
    md += ["## Experiment 0 - HEOG shift toward the attended side at the swap (µV)", "",
           S.markdown_table(g, "{:.2f}"), ""]

    sr = pd.read_csv(R / "stimrecon.csv", dtype={"subject": str})
    ps = stimrecon_table(sr)
    fig_stimrecon(ps)
    subj = (sr.groupby(["subject", "eog_regression", "window_s"])[["n_correct", "n_windows"]].sum()
              .assign(acc=lambda d: d.n_correct / d.n_windows).reset_index())
    t = (subj.groupby(["eog_regression", "window_s"]).acc
             .agg(mean="mean", sd="std", min="min", max="max").reset_index())
    t["p_vs_chance"] = [S.vs_chance(subj[(subj.eog_regression == r) & (subj.window_s == w)].acc)
                        for r, w in zip(t.eog_regression, t.window_s)]
    md += ["## Experiment 1 - stimulus reconstruction (all conditions pooled)", "", S.markdown_table(t), ""]

    sp_path = R / "spatial_pooled.csv"
    if sp_path.exists():
        sp = pd.read_csv(sp_path, dtype={"subject": str})
        cc = pd.read_csv(R / "spatial_crosscond.csv", dtype={"subject": str})
        perm = pd.read_csv(R / "permutation.csv", dtype={"subject": str}) if (R / "permutation.csv").exists() else None
        fig_spatial_vs_window(sp)
        fig_per_subject(sp, perm)
        fig_crosscond(cc)
        for L in (L_MAIN, 60):
            gt = group_table(sp, L)
            gt.to_csv(R / f"group_spatial_{L}s.csv", index=False)
            md += [f"## Experiment 2 - spatial decoding, leave-one-trial-out, {L} s windows", "",
                   "p_vs_chance: one-sided Wilcoxon; p_holm: Holm-corrected over the table; "
                   "p_vs_eog: paired two-sided Wilcoxon against the EOG-only model.", "",
                   S.markdown_table(gt), ""]
        ct = cross_table(cc, L_MAIN)
        md += [f"## Experiment 3 - cross-condition transfer ({L_MAIN} s windows)", "",
               "Trained on FixedVideo (gaze follows attention), tested on the same condition "
               "or on MovingVideo/MovingTargetNoise (gaze decorrelated).", "", S.markdown_table(ct), ""]
        if (R / "leakage_controls.csv").exists():
            ls = leakage_per_subject(pd.read_csv(R / "leakage_controls.csv", dtype={"subject": str}))
            fig_leakage(ls)
            md += ["## Experiment 5 - leakage and drift controls (5 s windows, all conditions)", "",
                   "leaky_random_cv: shuffled 5-fold CV over windows (train and test share blocks). "
                   "time_decoding_loto: trial-disjoint, label = first vs second half of the trial.", "",
                   S.markdown_table(leakage_table(ls)), ""]
        if perm is not None:
            pc = (perm.assign(sig=perm.p_value < 0.05)
                      .groupby(["model", "condition"])
                      .agg(n_subjects=("sig", "size"), n_significant=("sig", "sum"),
                           median_p=("p_value", "median")).reset_index())
            pc["model"] = pc.model.map(NAME)
            md += ["## Experiment 4 - per-subject permutation tests (10 s windows)", "",
                   "Trial-level label flips. 'ALL' pools the subject's conditions. "
                   "Per-condition p-values cannot go much below ~0.25 (2 trials per condition).", "",
                   S.markdown_table(pc), ""]
    (R / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
