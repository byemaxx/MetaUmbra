"""Shared peptide knockoff Monte Carlo helpers."""

from collections import Counter
from typing import Dict, Optional, Tuple, Union

import numpy as np


BROWN_MOMENT_PREFIXES = (500, 1_000, 2_000, 5_000, 10_000, 20_000)


def _mc_sum_from_pool(
    pool: Optional[np.ndarray],
    K: int,
    c: int,
    rng: np.random.Generator,
    sample_block_size: int,
) -> np.ndarray:
    """Sample K null sums with replacement from one empirical contribution pool.

    Replacement is intentional: each shared peptide contribution is sampled
    independently from the stratum-specific empirical null distribution.
    """
    if c <= 0:
        return np.zeros(int(K), dtype=np.float64)
    if pool is None or pool.size == 0:
        return np.zeros(int(K), dtype=np.float64)

    K = int(K)
    c = int(c)
    block = int(max(1, sample_block_size))
    out = np.zeros(K, dtype=np.float64)
    i = 0
    while i < K:
        j = min(K, i + block)
        idx = rng.integers(0, int(pool.size), size=(j - i, c), endpoint=False)
        out[i:j] = pool[idx].sum(axis=1)
        i = j
    return out


def shared_knockoff_mc(
    gid: str,
    obs_shared_score: float,
    K: int,
    rng: np.random.Generator,
    pools: Optional[Dict[Union[int, Tuple[int, int]], np.ndarray]],
    counts_by_genome: Dict[str, Counter],
    sample_block_size: int,
) -> Tuple[float, float, float, float, float]:
    """Empirical p-value and null moments for shared evidence by knockoff MC."""
    counts = counts_by_genome.get(gid, None)
    if not counts:
        return 1.0, 0.0, 0.0, 0.0, 0.0

    null_sum = np.zeros(int(K), dtype=np.float64)
    for key, c in counts.items():
        pool = pools.get(key, None) if pools else None
        null_sum += _mc_sum_from_pool(
            pool=pool,
            K=int(K),
            c=int(c),
            rng=rng,
            sample_block_size=sample_block_size,
        )

    ge = float(np.sum(null_sum >= float(obs_shared_score)))
    p = (1.0 + ge) / (1.0 + float(K))
    mu = float(null_sum.mean())
    sd = float(null_sum.std(ddof=1)) if int(K) > 1 else 0.0
    p95 = float(np.quantile(null_sum, 0.95))
    p99 = float(np.quantile(null_sum, 0.99))
    return float(p), mu, sd, p95, p99


def _compound_shared_sum_draws(
    shared_counts: np.ndarray,
    contribution_pool: np.ndarray,
    rng: np.random.Generator,
    exact_count_cutoff: int,
    sample_block_size: int,
) -> Tuple[np.ndarray, int]:
    """Draw compound shared-score sums with bounded temporary allocations.

    Counts up to ``exact_count_cutoff`` use the empirical contribution pool
    directly. Larger counts use a non-negative Gamma approximation matching the
    mean and variance of an independent sum from that pool.
    """
    counts = np.asarray(shared_counts, dtype=int).ravel()
    if bool(np.any(counts < 0)):
        raise ValueError("shared_counts must be non-negative.")

    pool = np.asarray(contribution_pool, dtype=np.float64).ravel()
    pool = pool[np.isfinite(pool) & (pool >= 0.0)]
    out = np.zeros(counts.size, dtype=np.float64)
    if pool.size == 0 or counts.size == 0:
        return out, 0

    cutoff = int(max(0, exact_count_cutoff))
    block = int(max(1, sample_block_size))
    exact_mask = (counts > 0) & (counts <= cutoff)
    for count in np.unique(counts[exact_mask]):
        row_indices = np.flatnonzero(counts == int(count))
        for start in range(0, int(row_indices.size), block):
            selected = row_indices[start : start + block]
            draw_indices = rng.integers(
                0,
                int(pool.size),
                size=(int(selected.size), int(count)),
                endpoint=False,
            )
            out[selected] = pool[draw_indices].sum(axis=1)

    approximate_indices = np.flatnonzero(counts > cutoff)
    if approximate_indices.size:
        mean = float(pool.mean())
        variance = float(pool.var(ddof=0))
        approximate_counts = counts[approximate_indices].astype(np.float64)
        if mean <= 0.0:
            out[approximate_indices] = 0.0
        elif variance <= 1e-30:
            out[approximate_indices] = approximate_counts * mean
        else:
            shapes = approximate_counts * mean * mean / variance
            scale = variance / mean
            out[approximate_indices] = rng.gamma(shape=shapes, scale=scale)

    return out, int(approximate_indices.size)


def _empirical_upper_tail(
    calibration_values: np.ndarray,
    query_values: np.ndarray,
    *,
    add_one: bool,
) -> np.ndarray:
    """Return tie-conservative empirical upper-tail probabilities."""
    calibration = np.sort(np.asarray(calibration_values, dtype=np.float64).ravel())
    query = np.asarray(query_values, dtype=np.float64)
    if calibration.size == 0:
        return np.ones(query.shape, dtype=np.float64)
    first_equal = np.searchsorted(calibration, query, side="left")
    exceedances = int(calibration.size) - first_equal
    if add_one:
        return (1.0 + exceedances.astype(np.float64)) / (1.0 + float(calibration.size))
    return exceedances.astype(np.float64) / float(calibration.size)


def brown_scaled_chi_square_moment_fit(
    calibration_statistics: np.ndarray,
    query_statistics: np.ndarray,
) -> Dict[str, object]:
    """Fit ``c * chi2(nu)`` by moments and score query statistics.

    The mean and sample variance are estimated from synchronized joint-null
    Fisher statistics. Degenerate calibration samples are reported as not
    estimable and return p=1 rather than manufacturing a tail distribution.
    """
    calibration = np.asarray(calibration_statistics, dtype=np.float64).ravel()
    query = np.asarray(query_statistics, dtype=np.float64)
    if calibration.size < 2:
        raise ValueError("At least two calibration statistics are required.")
    if not bool(np.all(np.isfinite(calibration))) or bool(np.any(calibration < 0.0)):
        raise ValueError("Calibration statistics must be finite and non-negative.")
    if not bool(np.all(np.isfinite(query))) or bool(np.any(query < 0.0)):
        raise ValueError("Query statistics must be finite and non-negative.")

    mean = float(calibration.mean())
    variance = float(calibration.var(ddof=1))
    estimable = bool(mean > 0.0 and variance > 0.0)
    if not estimable:
        return {
            "null_fisher_mean": mean,
            "null_fisher_variance": variance,
            "brown_scale": 0.0,
            "brown_df": 0.0,
            "brown_estimable": False,
            "pvalues": np.ones(query.shape, dtype=np.float64),
        }

    scale = variance / (2.0 * mean)
    degrees_of_freedom = 2.0 * mean * mean / variance
    try:
        from scipy.stats import chi2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("scipy is required for Brown moment calibration.") from exc
    pvalues = np.asarray(
        chi2.sf(query / scale, degrees_of_freedom),
        dtype=np.float64,
    )
    pvalues = np.clip(pvalues, 0.0, 1.0)
    return {
        "null_fisher_mean": mean,
        "null_fisher_variance": variance,
        "brown_scale": float(scale),
        "brown_df": float(degrees_of_freedom),
        "brown_estimable": True,
        "pvalues": pvalues,
    }


def conditional_joint_null_fisher_mc(
    *,
    observed_unique_count: int,
    observed_shared_score: float,
    observed_matched_count: int,
    theoretical_total_peptides: int,
    theoretical_unique_peptides: int,
    iterations: int,
    validation_iterations: int,
    rng: np.random.Generator,
    shared_contribution_pool: np.ndarray,
    exact_shared_count_cutoff: int = 64,
    sample_block_size: int = 256,
) -> Dict[str, object]:
    """Experimental conditional opportunity/compound joint-null Fisher test.

    This function is deliberately separate from ``shared_knockoff_mc``. It does
    not change the production statistic. One null replicate allocates a fixed
    matched-peptide count between theoretical unique and shared opportunities,
    then generates the shared score for the remaining slots.
    """
    observed_unique_count = int(observed_unique_count)
    observed_matched_count = int(observed_matched_count)
    theoretical_total_peptides = int(theoretical_total_peptides)
    theoretical_unique_peptides = int(theoretical_unique_peptides)
    iterations = int(iterations)
    validation_iterations = int(validation_iterations)
    if observed_unique_count < 0 or observed_matched_count < 0:
        raise ValueError("Observed peptide counts must be non-negative.")
    if observed_unique_count > observed_matched_count:
        raise ValueError("observed_unique_count cannot exceed observed_matched_count.")
    if theoretical_total_peptides <= 0:
        raise ValueError("theoretical_total_peptides must be positive.")
    if not (0 <= theoretical_unique_peptides <= theoretical_total_peptides):
        raise ValueError(
            "theoretical_unique_peptides must be between zero and theoretical_total_peptides."
        )
    if iterations < 1 or validation_iterations < 1:
        raise ValueError("iterations and validation_iterations must both be positive.")
    if not np.isfinite(float(observed_shared_score)) or float(observed_shared_score) < 0.0:
        raise ValueError("observed_shared_score must be finite and non-negative.")

    conditional_draws = int(min(observed_matched_count, theoretical_total_peptides))
    if observed_unique_count > conditional_draws:
        raise ValueError(
            "observed_unique_count exceeds the conditional theoretical draw count."
        )

    def draw_replicates(count: int) -> Tuple[np.ndarray, np.ndarray, int]:
        unique = rng.hypergeometric(
            ngood=theoretical_unique_peptides,
            nbad=theoretical_total_peptides - theoretical_unique_peptides,
            nsample=conditional_draws,
            size=int(count),
        ).astype(int)
        shared_counts = conditional_draws - unique
        shared, approximate = _compound_shared_sum_draws(
            shared_counts=shared_counts,
            contribution_pool=shared_contribution_pool,
            rng=rng,
            exact_count_cutoff=exact_shared_count_cutoff,
            sample_block_size=sample_block_size,
        )
        return unique, shared, approximate

    unique_calibration, shared_calibration, approximate_calibration = draw_replicates(iterations)
    unique_validation, shared_validation, approximate_validation = draw_replicates(validation_iterations)

    try:
        from scipy.stats import hypergeom, kstest, spearmanr  # type: ignore
    except ImportError as exc:
        raise RuntimeError("scipy is required for the experimental joint-null pilot.") from exc

    def unique_tail(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=int)
        tails = hypergeom.sf(
            values - 1,
            theoretical_total_peptides,
            theoretical_unique_peptides,
            conditional_draws,
        )
        return np.clip(np.asarray(tails, dtype=np.float64), 1e-300, 1.0)

    observed_unique_p = float(unique_tail(np.asarray([observed_unique_count]))[0])
    calibration_unique_p = unique_tail(unique_calibration)
    validation_unique_p = unique_tail(unique_validation)
    observed_shared_p = float(
        _empirical_upper_tail(
            shared_calibration,
            np.asarray([float(observed_shared_score)]),
            add_one=True,
        )[0]
    )
    calibration_shared_p = np.clip(
        _empirical_upper_tail(
            shared_calibration,
            shared_calibration,
            add_one=False,
        ),
        1.0 / float(iterations),
        1.0,
    )

    calibration_fisher = -2.0 * (
        np.log(calibration_unique_p) + np.log(calibration_shared_p)
    )
    calibration_fisher[unique_calibration <= 0] = 0.0
    if observed_unique_count <= 0:
        observed_fisher = 0.0
        observed_joint_p = 1.0
    else:
        observed_fisher = float(
            -2.0 * (np.log(observed_unique_p) + np.log(observed_shared_p))
        )
        observed_joint_p = float(
            (1.0 + np.sum(calibration_fisher >= observed_fisher))
            / (1.0 + float(iterations))
        )

    validation_shared_p = np.clip(
        _empirical_upper_tail(
            shared_calibration,
            shared_validation,
            add_one=True,
        ),
        1.0 / (1.0 + float(iterations)),
        1.0,
    )
    validation_fisher = -2.0 * (
        np.log(validation_unique_p) + np.log(validation_shared_p)
    )
    validation_fisher[unique_validation <= 0] = 0.0
    validation_joint_p = np.clip(
        _empirical_upper_tail(
            calibration_fisher,
            validation_fisher,
            add_one=True,
        ),
        1.0 / (1.0 + float(iterations)),
        1.0,
    )

    dependence_estimable = bool(
        np.unique(unique_calibration).size > 1
        and np.unique(shared_calibration).size > 1
    )
    if dependence_estimable:
        statistic_dependence = float(
            spearmanr(unique_calibration, shared_calibration).statistic
        )
        component_dependence = float(
            spearmanr(-np.log(calibration_unique_p), -np.log(calibration_shared_p)).statistic
        )
    else:
        statistic_dependence = float("nan")
        component_dependence = float("nan")
    ks_result = kstest(validation_joint_p, "uniform")

    brown_prefix_results: Dict[str, Dict[str, object]] = {}
    brown_prefixes = sorted(
        set(prefix for prefix in BROWN_MOMENT_PREFIXES if prefix <= iterations)
        | {iterations}
    )
    for prefix in brown_prefixes:
        prefix_shared = shared_calibration[:prefix]
        prefix_unique_p = calibration_unique_p[:prefix]
        prefix_shared_p = np.clip(
            _empirical_upper_tail(
                prefix_shared,
                prefix_shared,
                add_one=False,
            ),
            1.0 / float(prefix),
            1.0,
        )
        prefix_fisher = -2.0 * (
            np.log(prefix_unique_p) + np.log(prefix_shared_p)
        )
        prefix_fisher[unique_calibration[:prefix] <= 0] = 0.0

        prefix_observed_shared_p = float(
            _empirical_upper_tail(
                prefix_shared,
                np.asarray([float(observed_shared_score)]),
                add_one=True,
            )[0]
        )
        prefix_observed_fisher = (
            0.0
            if observed_unique_count <= 0
            else float(
                -2.0
                * (
                    np.log(observed_unique_p)
                    + np.log(prefix_observed_shared_p)
                )
            )
        )
        prefix_validation_shared_p = np.clip(
            _empirical_upper_tail(
                prefix_shared,
                shared_validation,
                add_one=True,
            ),
            1.0 / (1.0 + float(prefix)),
            1.0,
        )
        prefix_validation_fisher = -2.0 * (
            np.log(validation_unique_p) + np.log(prefix_validation_shared_p)
        )
        prefix_validation_fisher[unique_validation <= 0] = 0.0
        fit = brown_scaled_chi_square_moment_fit(
            prefix_fisher,
            np.concatenate(
                [
                    np.asarray([prefix_observed_fisher], dtype=np.float64),
                    prefix_validation_fisher,
                ]
            ),
        )
        fit_pvalues = np.asarray(fit.pop("pvalues"), dtype=np.float64)
        observed_brown_p = (
            1.0 if observed_unique_count <= 0 else float(fit_pvalues[0])
        )
        validation_brown_p = fit_pvalues[1:]
        brown_ks = kstest(validation_brown_p, "uniform")
        brown_prefix_results[str(prefix)] = {
            **fit,
            "calibration_iterations": int(prefix),
            "validation_iterations": int(validation_iterations),
            "pvalue_brown": observed_brown_p,
            "observed_shared_p": prefix_observed_shared_p,
            "observed_fisher_statistic": prefix_observed_fisher,
            "validation_fraction_p_le_0_01": float(
                np.mean(validation_brown_p <= 0.01)
            ),
            "validation_fraction_p_le_0_05": float(
                np.mean(validation_brown_p <= 0.05)
            ),
            "validation_ks_uniform_statistic": float(brown_ks.statistic),
            "validation_ks_uniform_pvalue": float(brown_ks.pvalue),
        }

    return {
        "pvalue_unique_component": observed_unique_p,
        "pvalue_shared_component": observed_shared_p,
        "fisher_statistic_observed": observed_fisher,
        "pvalue_joint_null_fisher": observed_joint_p,
        "iterations": iterations,
        "validation_iterations": validation_iterations,
        "minimum_attainable_p": 1.0 / (1.0 + float(iterations)),
        "conditional_matched_count": conditional_draws,
        "null_unique_mean": float(unique_calibration.mean()),
        "null_shared_mean": float(shared_calibration.mean()),
        "null_shared_p95": float(np.quantile(shared_calibration, 0.95)),
        "null_statistic_spearman": statistic_dependence,
        "null_component_spearman": component_dependence,
        "dependence_estimable": dependence_estimable,
        "validation_fraction_p_le_0_01": float(np.mean(validation_joint_p <= 0.01)),
        "validation_fraction_p_le_0_05": float(np.mean(validation_joint_p <= 0.05)),
        "validation_ks_uniform_statistic": float(ks_result.statistic),
        "validation_ks_uniform_pvalue": float(ks_result.pvalue),
        "gamma_approximation_fraction": float(
            (approximate_calibration + approximate_validation)
            / float(iterations + validation_iterations)
        ),
        "exact_shared_count_cutoff": int(max(0, exact_shared_count_cutoff)),
        "gate_applied": bool(observed_unique_count <= 0),
        "null_model": "conditional-opportunity-compound-v0",
        "brown_prefix_results": brown_prefix_results,
    }
