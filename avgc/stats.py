"""Group-level statistics.  With N = 9 subjects we use exact non-parametric
tests and always show every subject, never only the mean."""
import numpy as np
from scipy.stats import binom, wilcoxon


def vs_chance(acc, chance=0.5):
    """One-sided Wilcoxon signed-rank: are accuracies above chance across subjects?"""
    d = np.asarray(acc) - chance
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(d, alternative="greater").pvalue)


def paired(a, b):
    """Two-sided paired Wilcoxon signed-rank test between two models."""
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(d).pvalue)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def binomial_threshold(n_windows, alpha=0.05):
    """Accuracy needed for a window-level binomial test.  Shown for comparison
    with the literature only: it assumes independent windows, which is false
    for consecutive windows of the same block, so it is anti-conservative."""
    return (binom.ppf(1 - alpha, n_windows, 0.5) + 1) / n_windows
