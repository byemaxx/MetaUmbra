"""Ranking and p-value combination helpers for genome-presence scoring."""

import numpy as np


HMP_K2_EXACT_CALIBRATION = 2.0
HMP_CALIBRATION_COMPONENT_COUNT = 2
HMP_CALIBRATION_BASIS = "exact_k2_arbitrary_dependence"
DEFAULT_PRESENCE_COMBINATION_METHOD = "simes-closed"
PRESENCE_COMBINATION_METHODS = (
    "simes-closed",
    "bonferroni-min",
    "harmonic-mean-calibrated",
    "fisher",
    "harmonic-mean",
    "unique-only",
)


def _normalize_presence_combination_method(method: str | None) -> str:
    """Normalize a configured two-component presence p-value combiner."""
    normalized = str(method or DEFAULT_PRESENCE_COMBINATION_METHOD).strip().lower()
    aliases = {
        "simes_closed": "simes-closed",
        "simes": "simes-closed",
        "harmonic_mean_calibrated": "harmonic-mean-calibrated",
        "calibrated-hmp": "harmonic-mean-calibrated",
        "calibrated_hmp": "harmonic-mean-calibrated",
        "harmonic_mean": "harmonic-mean",
        "hmp": "harmonic-mean",
        "bonferroni_min": "bonferroni-min",
        "bonferroni": "bonferroni-min",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in PRESENCE_COMBINATION_METHODS:
        choices = "', '".join(PRESENCE_COMBINATION_METHODS)
        raise ValueError(f"presence_combination_method must be one of '{choices}'.")
    return normalized


def fisher_p_2(p1: float, p2: float) -> float:
    """Fisher combine two p-values with the closed-form chi-square(df=4) survival."""
    p1 = float(min(max(p1, 1e-300), 1.0))
    p2 = float(min(max(p2, 1e-300), 1.0))
    stat = -2.0 * (np.log(p1) + np.log(p2))
    x = float(stat)
    return float(np.exp(-x / 2.0) * (1.0 + x / 2.0))


def harmonic_mean_p_2(
    p_unique: float,
    p_shared: float,
    *,
    require_unique_evidence: bool = False,
    unique_count: int | None = None,
) -> float:
    """Combine unique and shared p-values with the equal-weight two-test HMP.

    The optional evidence gate keeps shared evidence in a supporting role: a
    candidate with no observed panel-unique peptide receives p=1.  The HMP is
    computed as ``2*p_unique*p_shared/(p_unique+p_shared)`` to avoid unstable
    reciprocal arithmetic for very small p-values.
    """
    if require_unique_evidence and (unique_count is None or int(unique_count) <= 0):
        return 1.0
    p_unique = float(min(max(p_unique, 1e-300), 1.0))
    p_shared = float(min(max(p_shared, 1e-300), 1.0))
    return float(min(1.0, (2.0 * p_unique * p_shared) / (p_unique + p_shared)))


def calibrated_harmonic_mean_p_2(
    p_unique: float,
    p_shared: float,
    *,
    num_peptides_unique: int,
    floor: float = 1e-300,
) -> float:
    """Return the exact K=2 arbitrary-dependence calibrated HMP p-value.

    The exact K=2 calibration factor is prescribed as ``2`` and is never fitted
    to benchmark, simulation, or biological-study results. At least one
    observed panel-unique peptide is required, so shared-only evidence cannot
    produce a calibrated-HMP call.
    """
    if int(num_peptides_unique) <= 0:
        return 1.0
    p_unique = float(min(max(p_unique, floor), 1.0))
    p_shared = float(min(max(p_shared, floor), 1.0))
    raw_hmp = (2.0 * p_unique * p_shared) / (p_unique + p_shared)
    return float(min(1.0, HMP_K2_EXACT_CALIBRATION * raw_hmp))


def bonferroni_min_p_2(
    p_unique: float,
    p_shared: float,
    *,
    require_unique_evidence: bool = False,
    unique_count: int | None = None,
) -> float:
    """Return the two-test Bonferroni minimum-p p-value.

    Production presence scoring supplies the unique-evidence gate so shared
    evidence can support, but cannot independently establish, a genome call.
    The optional gate remains opt-in here for direct mathematical use.
    """
    if require_unique_evidence and (unique_count is None or int(unique_count) <= 0):
        return 1.0
    p_unique = float(min(max(p_unique, 1e-300), 1.0))
    p_shared = float(min(max(p_shared, 1e-300), 1.0))
    return float(min(1.0, 2.0 * min(p_unique, p_shared)))


def simes_intersection_p_2(p_unique: float, p_shared: float) -> float:
    """Return the two-component Simes intersection-test p-value.

    This is the established two-p-value Simes form
    ``min(1, 2*min(p_unique, p_shared), max(p_unique, p_shared))``.  It is
    exported separately from the genome-level closed-testing value because an
    intersection result alone does not impose MetaUmbra's unique-evidence
    requirement.
    """
    p_unique = float(min(max(p_unique, 1e-300), 1.0))
    p_shared = float(min(max(p_shared, 1e-300), 1.0))
    return float(min(1.0, 2.0 * min(p_unique, p_shared), max(p_unique, p_shared)))


def simes_closed_p_2(
    p_unique: float,
    p_shared: float,
    *,
    num_peptides_unique: int,
) -> float:
    """Return the unique-hypothesis closed-testing p-value for two components.

    Genome-specific support requires an observed unique peptide.  Conditional
    on that gate, closed testing of the unique component against the Simes
    intersection reduces to ``min(1, 2*p_unique, max(p_unique, p_shared))``.
    The result is never smaller than the unique-component p-value.
    """
    if int(num_peptides_unique) <= 0:
        return 1.0
    p_unique = float(min(max(p_unique, 1e-300), 1.0))
    p_shared = float(min(max(p_shared, 1e-300), 1.0))
    return float(min(1.0, 2.0 * p_unique, max(p_unique, p_shared)))


def combine_presence_pvalues(
    p_unique: float,
    p_shared: float,
    *,
    method: str = DEFAULT_PRESENCE_COMBINATION_METHOD,
    unique_count: int | None = None,
    hmp_require_unique_evidence: bool = True,
) -> float:
    """Combine component p-values using one configured presence rule."""
    normalized = _normalize_presence_combination_method(method)
    if normalized == "simes-closed":
        return simes_closed_p_2(
            p_unique,
            p_shared,
            num_peptides_unique=0 if unique_count is None else unique_count,
        )
    if normalized == "fisher":
        return fisher_p_2(p_unique, p_shared)
    if normalized == "harmonic-mean-calibrated":
        return calibrated_harmonic_mean_p_2(
            p_unique,
            p_shared,
            num_peptides_unique=0 if unique_count is None else unique_count,
        )
    if normalized == "harmonic-mean":
        return harmonic_mean_p_2(
            p_unique,
            p_shared,
            require_unique_evidence=hmp_require_unique_evidence,
            unique_count=unique_count,
        )
    if normalized == "unique-only":
        return float(min(max(p_unique, 1e-300), 1.0))
    return bonferroni_min_p_2(
        p_unique,
        p_shared,
        require_unique_evidence=True,
        unique_count=unique_count,
    )


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
