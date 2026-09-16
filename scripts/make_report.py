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
