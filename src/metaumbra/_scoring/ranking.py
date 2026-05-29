"""Ranking and p-value combination helpers for genome-presence scoring."""

import numpy as np


def fisher_p_2(p1: float, p2: float) -> float:
    """Fisher combine two p-values with the closed-form chi-square(df=4) survival."""
    p1 = float(min(max(p1, 1e-300), 1.0))
    p2 = float(min(max(p2, 1e-300), 1.0))
    stat = -2.0 * (np.log(p1) + np.log(p2))
    x = float(stat)
    return float(np.exp(-x / 2.0) * (1.0 + x / 2.0))


def bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values for a 1D array of p-values."""
    p = np.asarray(pvals, dtype=float)
    n = int(p.size)
    if n == 0:
        return p

    order = np.argsort(p)
    ranked = p[order]
    q = ranked * float(n) / (np.arange(1, n + 1, dtype=float))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out


def qvalues_to_presence_scores(qvals: np.ndarray) -> np.ndarray:
    """Convert q-values to the existing -log10 presence score scale."""
    scores = -np.log10(np.clip(np.asarray(qvals, dtype=float), 1e-300, 1.0))
    scores[np.isclose(scores, 0.0, atol=1e-12)] = 0.0
    return scores
