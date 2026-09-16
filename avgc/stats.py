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
