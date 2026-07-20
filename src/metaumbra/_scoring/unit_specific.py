"""Unit-specific scoring worker helpers.

This module is intentionally self-contained so Windows spawn-based
multiprocessing can import worker initializers and worker functions without
importing ``metaumbra.scoring`` back into a child process.
"""

import json
from collections import Counter
from typing import Dict, List, Set, Tuple, Union

import numpy as np
import pandas as pd

from .empirical import (
    DEFAULT_UNIQUE_EMPIRICAL_BACKGROUND_THRESHOLD_QUANTILE,
    _compute_empirical_background_stats_for_table,
)
from .knockoff import shared_knockoff_mc
from .ranking import bh_qvalues, fisher_p_2, qvalues_to_presence_scores
from .stats import (
    DEFAULT_UNIQUE_COUNT_POWER,
    DEFAULT_UNIQUE_PEPTIDE_ERROR_SOURCE,
    _clip_pvalue,
    _normalize_unique_pvalue_mode,
    _tempered_unique_error_product_pvalue,
)


UNIT_EMPIRICAL_BACKGROUND_INITIAL_EXCLUDE_FRACTION = 0.05
UNIT_EMPIRICAL_BACKGROUND_MIN_EXCLUDE_FRACTION = 0.02
UNIT_EMPIRICAL_BACKGROUND_MAX_EXCLUDE_FRACTION = 0.20
UNIT_EMPIRICAL_BACKGROUND_CANDIDATE_Q = 0.20
UNIT_EMPIRICAL_BACKGROUND_MAX_ITERATIONS = 3
UNIT_EMPIRICAL_BACKGROUND_SMALL_UNIT_MIN_ACTIVE_GENOMES = 100
UNIT_EMPIRICAL_BACKGROUND_THRESHOLD_QUANTILE = DEFAULT_UNIQUE_EMPIRICAL_BACKGROUND_THRESHOLD_QUANTILE

# The cohort-wide unit intentionally uses a moderately more permissive
# empirical-background calibration.  Pooling every sample raises the absolute
# unique-peptide background as well as the signal, so using the grouped-unit
# profile unchanged can erase real cohort-wide excess evidence.  These bounds
# remain below the legacy pooled scorer's 0.10--0.30 range.
ALL_SAMPLES_EMPIRICAL_BACKGROUND_INITIAL_EXCLUDE_FRACTION = 0.10
ALL_SAMPLES_EMPIRICAL_BACKGROUND_MIN_EXCLUDE_FRACTION = 0.05
ALL_SAMPLES_EMPIRICAL_BACKGROUND_MAX_EXCLUDE_FRACTION = 0.20
ALL_SAMPLES_EMPIRICAL_BACKGROUND_MAX_ITERATIONS = 5
UNIT_EMPIRICAL_BACKGROUND_OUTPUT_COLUMNS = (
    "unique_empirical_background_bin",
    "unique_empirical_background_size",
    "unique_empirical_background_threshold",
    "unique_empirical_excess_count",
    "p_unique_empirical_tail",
)
UNIT_EMPIRICAL_BACKGROUND_INTERNAL_COLUMNS = (
    "unit_empirical_background_iteration_trace",
    "unit_empirical_background_threshold_quantile",
    "unit_empirical_background_final_exclude_fraction",
    "unit_empirical_background_iterations",
    "unit_empirical_background_active_genomes",
    "unit_empirical_background_warning",
)


def _empirical_background_calibration_for_unit_mode(unit_mode: str) -> Dict[str, object]:
    """Return the explicit empirical-null calibration profile for a unit mode."""
    if str(unit_mode).strip() == "all-samples":
        return {
            "profile": "all-samples-moderately-permissive",
            "initial_exclude_fraction": ALL_SAMPLES_EMPIRICAL_BACKGROUND_INITIAL_EXCLUDE_FRACTION,
            "min_exclude_fraction": ALL_SAMPLES_EMPIRICAL_BACKGROUND_MIN_EXCLUDE_FRACTION,
            "max_exclude_fraction": ALL_SAMPLES_EMPIRICAL_BACKGROUND_MAX_EXCLUDE_FRACTION,
            "candidate_q": UNIT_EMPIRICAL_BACKGROUND_CANDIDATE_Q,
            "max_iterations": ALL_SAMPLES_EMPIRICAL_BACKGROUND_MAX_ITERATIONS,
        }
    return {
        "profile": "grouped-unit-conservative",
        "initial_exclude_fraction": UNIT_EMPIRICAL_BACKGROUND_INITIAL_EXCLUDE_FRACTION,
        "min_exclude_fraction": UNIT_EMPIRICAL_BACKGROUND_MIN_EXCLUDE_FRACTION,
        "max_exclude_fraction": UNIT_EMPIRICAL_BACKGROUND_MAX_EXCLUDE_FRACTION,
        "candidate_q": UNIT_EMPIRICAL_BACKGROUND_CANDIDATE_Q,
        "max_iterations": UNIT_EMPIRICAL_BACKGROUND_MAX_ITERATIONS,
    }


def _unit_knock_deg_bin(d: int, degeneracy_bin_edges: List[int]) -> int:
    d = int(max(d, 1))
    edges = list(degeneracy_bin_edges)
    if d <= edges[0]:
        return 0
    for i, e in enumerate(edges[1:], start=1):
        if d <= e:
            return i
    return len(edges)


def _unit_knock_len_bin(length: int, peptide_length_bin_edges: List[int]) -> int:
    length = int(max(length, 0))
    edges = list(peptide_length_bin_edges)
    if length <= edges[0]:
        return 0
    for i, e in enumerate(edges[1:], start=1):
        if length <= e:
            return i
    return len(edges)


def _unit_knock_stratum(
    d: int,
    pep_len: int,
    use_length_strata: bool,
    degeneracy_bin_edges: List[int],
    peptide_length_bin_edges: List[int],
) -> Union[int, Tuple[int, int]]:
    deg_bin = _unit_knock_deg_bin(d=d, degeneracy_bin_edges=degeneracy_bin_edges)
    if not use_length_strata:
        return int(deg_bin)
    len_bin = _unit_knock_len_bin(length=pep_len, peptide_length_bin_edges=peptide_length_bin_edges)
    return (int(deg_bin), int(len_bin))


def _unit_compute_weight(d: int) -> float:
    return 1.0 / float(int(max(d, 1)))


def _unit_build_knockoff_pools_for_peptides(
    observed_peptides: Set[str],
    peptide_deg: Dict[str, int],
    peptide_score: Dict[str, float],
    use_length_strata: bool,
    degeneracy_bin_edges: List[int],
    peptide_length_bin_edges: List[int],
) -> Dict[Union[int, Tuple[int, int]], np.ndarray]:
    pools: Dict[Union[int, Tuple[int, int]], List[float]] = {}
    for peptide in sorted(observed_peptides):
        d = int(peptide_deg.get(peptide, 0))
        if d <= 1:
            continue
        score = float(peptide_score.get(peptide, 1.0))
        weight = _unit_compute_weight(d=d)
        key = _unit_knock_stratum(
            d=d,
            pep_len=len(peptide),
            use_length_strata=use_length_strata,
            degeneracy_bin_edges=degeneracy_bin_edges,
            peptide_length_bin_edges=peptide_length_bin_edges,
        )
        pools.setdefault(key, []).append(float(weight * score))
    return {key: np.asarray(values, dtype=np.float32) for key, values in pools.items()}


def _unit_p_shared_knockoff_mc(
    gid: str,
    obs_shared_score: float,
    K: int,
    rng: np.random.Generator,
    pools: Dict[Union[int, Tuple[int, int]], np.ndarray],
    counts_by_genome: Dict[str, Counter],
    sample_block_size: int,
) -> Tuple[float, float, float, float, float]:
    return shared_knockoff_mc(
        gid=gid,
        obs_shared_score=obs_shared_score,
        K=K,
        rng=rng,
        pools=pools,
        counts_by_genome=counts_by_genome,
        sample_block_size=sample_block_size,
    )


def _unit_fisher_p_2(p1: float, p2: float) -> float:
    return fisher_p_2(p1=p1, p2=p2)


def _unit_bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    return bh_qvalues(pvals)


def _unit_hypergeom_tail_pvalue(
    observed: int,
    universe_size: int,
    success_states: int,
    draws: int,
) -> float:
    observed = int(observed)
    universe_size = int(max(universe_size, 0))
    success_states = int(max(success_states, 0))
    draws = int(max(draws, 0))
    if observed <= 0:
        return 1.0
    if universe_size <= 0 or success_states <= 0 or draws <= 0:
        return 1.0
    draws = min(draws, universe_size)
    success_states = min(success_states, universe_size)
    try:
        from scipy.stats import hypergeom  # type: ignore

        return _clip_pvalue(hypergeom.sf(observed - 1, universe_size, success_states, draws))
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for hypergeometric-opportunity unique p-values. "
            "Install scipy or use alpha-upper-bound mode."
        ) from exc


def _unit_unique_pvalue_stats_for_genome(
    gid: str,
    matched_peptides: Set[str],
    observed_unique: int,
    observed_unique_pool_size: int,
    mode: str,
    peptide_deg: Dict[str, int],
    genome_theoretical_unique_peptides: Dict[str, int],
    total_theoretical_unique_peptides_all_genomes: int,
    single_peptide_error_rate_upper_bound: float,
    peptide_error_upper_by_peptide: Dict[str, float],
    unique_peptide_error_source: str = DEFAULT_UNIQUE_PEPTIDE_ERROR_SOURCE,
    unique_count_power: float = DEFAULT_UNIQUE_COUNT_POWER,
) -> dict:
    U = int(observed_unique)
    mode = _normalize_unique_pvalue_mode(mode)

    alpha = float(min(max(single_peptide_error_rate_upper_bound, 1e-12), 1.0))
    S = int(observed_unique_pool_size)
    A = int(genome_theoretical_unique_peptides.get(gid, 0))
    A_total = int(total_theoretical_unique_peptides_all_genomes)
    theoretical_unique: object = pd.NA
    p_unique = 1.0
    p_unique_depth = 1.0
    expected = 0.0
    fold = 0.0
    has_unique_evidence = bool(U > 0)
    null_model = ""
    unique_effective_count = float(U)
    count_model = "raw"

    if mode == "hypergeometric-opportunity":
        theoretical_unique = int(A)
        expected = float(S) * float(A) / float(A_total) if A_total > 0 else 0.0
        fold = float(U) / max(expected, 1e-12)
        null_model = "hypergeometric"
        if U > 0 and S > 0 and A > 0 and A_total > 0:
            p_unique_depth = _unit_hypergeom_tail_pvalue(
                observed=U,
                universe_size=A_total,
                success_states=A,
                draws=S,
            )
            p_unique = p_unique_depth
    elif mode == "alpha-upper-bound":
        unique_peptides = sorted(
            peptide
            for peptide in matched_peptides
            if int(peptide_deg.get(peptide, 1)) == 1
        )
        p_unique, unique_effective_count, error_source_used = _tempered_unique_error_product_pvalue(
            unique_peptides=unique_peptides,
            alpha=alpha,
            peptide_error_upper_by_peptide=peptide_error_upper_by_peptide,
            error_source=unique_peptide_error_source,
            unique_count_power=unique_count_power,
        )
        count_model = f"power:{float(unique_count_power):g}"
        p_unique_depth = p_unique
        null_model = error_source_used

    return {
        "p_unique": _clip_pvalue(p_unique),
        "p_unique_depth": _clip_pvalue(p_unique_depth),
        "unique_observed": int(U),
        "unique_expected_null": float(expected),
        "unique_depth_fold": float(fold),
        "unique_depth_null_model": null_model,
        "unique_pvalue_mode": mode,
        "unique_peptide_error_source": null_model if mode == "alpha-upper-bound" else "",
        "has_unique_evidence": bool(has_unique_evidence),
        "theoretical_unique_peptides": theoretical_unique,
        "unique_effective_count": float(unique_effective_count),
        "unique_pvalue_count_model": count_model,
        "unique_empirical_background_bin": "",
        "unique_empirical_background_size": 0,
        "unique_empirical_background_threshold": 0.0,
        "unique_empirical_excess_count": 0.0,
        "p_unique_empirical_tail": 1.0,
    }


def _unit_empirical_unique_stats_from_row(row: pd.Series, mode: str = "empirical-background") -> dict:
    U = int(row.get("num_peptides_unique", 0))
    expected = float(row.get("expected_unique_null", 0.0))
    excess = float(row.get("unique_empirical_excess_count", 0.0))
    p_unique = _clip_pvalue(float(row.get("p_unique_empirical_background_excess", 1.0)))
    return {
        "p_unique": p_unique,
        "p_unique_depth": p_unique,
        "unique_observed": int(U),
        "unique_expected_null": float(expected),
        "unique_depth_fold": float(row.get("unique_depth_fold", 0.0)),
        "unique_depth_null_model": "empirical-background",
        "unique_pvalue_mode": mode,
        "unique_peptide_error_source": "",
        "has_unique_evidence": bool(U > 0),
        "theoretical_unique_peptides": pd.NA,
        "unique_effective_count": float(excess),
        "unique_pvalue_count_model": "background-excess",
        "unique_empirical_background_bin": str(row.get("unique_empirical_background_bin", "")),
        "unique_empirical_background_size": int(row.get("unique_empirical_background_size", 0)),
        "unique_empirical_background_threshold": float(row.get("unique_empirical_background_threshold", 0.0)),
        "unique_empirical_excess_count": float(excess),
        "p_unique_empirical_tail": _clip_pvalue(float(row.get("p_unique_empirical_tail", 1.0))),
    }


def _unit_shared_metrics_for_genome(
    genome_id: str,
    matched_peptides: Set[str],
    peptide_deg: Dict[str, int],
    peptide_score: Dict[str, float],
    genome_total_theoretical_peptides: Dict[str, int],
    use_length_strata: bool,
    degeneracy_bin_edges: List[int],
    peptide_length_bin_edges: List[int],
) -> dict:
    total_matched = int(len(matched_peptides))
    unique_count = 0
    shared_count = 0
    peptide_scores: List[float] = []
    peptide_weights: List[float] = []
    weighted_contributions: List[float] = []
    shared_weights: List[float] = []
    shared_contributions: List[float] = []
    unique_weighted_evidence = 0.0
    strata_counter = Counter()

    for peptide in sorted(matched_peptides):
        d = int(peptide_deg.get(peptide, 1))
        score = float(peptide_score.get(peptide, 1.0))
        weight = float(_unit_compute_weight(d=d))
        contribution = float(weight * score)

        peptide_scores.append(score)
        peptide_weights.append(weight)
        weighted_contributions.append(contribution)

        if d == 1:
            unique_count += 1
            unique_weighted_evidence += score
        else:
            shared_count += 1
            shared_weights.append(weight)
            shared_contributions.append(contribution)
            strata_counter[
                _unit_knock_stratum(
                    d=d,
                    pep_len=len(peptide),
                    use_length_strata=use_length_strata,
                    degeneracy_bin_edges=degeneracy_bin_edges,
                    peptide_length_bin_edges=peptide_length_bin_edges,
                )
            ] += 1

    total_theoretical = int(genome_total_theoretical_peptides.get(genome_id, 0))
    effective_peptide_count = float(np.sum(peptide_weights)) if peptide_weights else 0.0
    weighted_evidence = float(np.sum(weighted_contributions)) if weighted_contributions else 0.0
    effective_shared = float(np.sum(shared_weights)) if shared_weights else 0.0
    weighted_shared = float(np.sum(shared_contributions)) if shared_contributions else 0.0

    return {
        "num_peptides_matched": int(total_matched),
        "num_peptides_unique": int(unique_count),
        "total_peptide_count": int(total_theoretical),
        "peptide_match_ratio": float(total_matched) / float(max(total_theoretical, 1)),
        "average_peptide_score": float(np.mean(peptide_scores)) if peptide_scores else 0.0,
        "effective_peptide_count": float(effective_peptide_count),
        "weighted_evidence": float(weighted_evidence),
        "unique_weighted_evidence": float(unique_weighted_evidence),
        "shared_fraction": float(shared_count) / float(total_matched) if total_matched else 0.0,
        "matched_peptide_count_shared": int(shared_count),
        "effective_peptide_count_shared": float(effective_shared),
        "weighted_evidence_shared": float(weighted_shared),
        "shared_stratum_counts": strata_counter,
    }


def _unit_knockoff_target_indices(
    genome_ids: List[str],
    metrics_by_genome: Dict[str, dict],
    matched_counts: np.ndarray,
    top_n_targets: int | None,
) -> np.ndarray:
    """Select per-unit knockoff targets using the legacy evidence ordering."""
    active_indices = [
        int(index)
        for index, count in enumerate(np.asarray(matched_counts, dtype=int).ravel())
        if int(count) >= 1
    ]
    if top_n_targets is None:
        return np.asarray(active_indices, dtype=int)

    limit = max(0, int(top_n_targets))

    def evidence_key(index: int) -> tuple:
        genome_id = str(genome_ids[index])
        metrics = metrics_by_genome[genome_id]
        return (
            -int(metrics["num_peptides_unique"]),
            -float(metrics["unique_weighted_evidence"]),
            -float(metrics["weighted_evidence"]),
            -float(metrics["effective_peptide_count"]),
            -float(metrics["peptide_match_ratio"]),
            -int(metrics["num_peptides_matched"]),
            genome_id,
        )

    ranked_indices = sorted(active_indices, key=evidence_key)
    return np.asarray(ranked_indices[:limit], dtype=int)


def _unit_in_any_stage2_range(value: float, ranges: List[Tuple[float, float]]) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    for a, b in ranges:
        lo = float(min(a, b))
        hi = float(max(a, b))
        if lo <= float(value) <= hi:
            return True
    return False


_UNIT_WORKER_CONTEXT: Dict[str, object] = {}


def _init_unit_specific_worker(context: Dict[str, object]) -> None:
    _UNIT_WORKER_CONTEXT.clear()
    _UNIT_WORKER_CONTEXT.update(context)


def _compute_unit_specific_single_unit_worker(args: Dict[str, object]) -> Dict[str, object]:
    context = _UNIT_WORKER_CONTEXT
    unit_idx = int(args["unit_idx"])
    unit_id = str(args["analysis_unit_id"])
    genome_ids = [str(x) for x in context["genome_ids"]]
    n_genomes = len(genome_ids)
    matched_counts = np.asarray(args["matched_counts"], dtype=int).ravel()
    unit_observed_peptides = set(str(x) for x in args["unit_observed_peptides"])
    genome_matched_peptides: Dict[str, Set[str]] = context["genome_matched_peptides"]  # type: ignore[assignment]
    peptide_deg: Dict[str, int] = context["peptide_deg"]  # type: ignore[assignment]
    peptide_score: Dict[str, float] = context["peptide_score"]  # type: ignore[assignment]
    genome_total_theoretical_peptides: Dict[str, int] = context["genome_total_theoretical_peptides"]  # type: ignore[assignment]
    genome_theoretical_unique_peptides: Dict[str, int] = context["genome_theoretical_unique_peptides"]  # type: ignore[assignment]
    peptide_error_upper_by_peptide: Dict[str, float] = context["peptide_error_upper_by_peptide"]  # type: ignore[assignment]
    lineage_map: Dict[str, object] = context["lineage_map"]  # type: ignore[assignment]
    mode = str(context["mode"])
    K1 = int(context["knockoff_mc_iterations"])
    K2_raw = context.get("knockoff_stage2_mc_iterations")
    K2 = int(K2_raw) if K2_raw is not None else None
    ranges: List[Tuple[float, float]] = context["knockoff_stage2_p_exist_ranges"]  # type: ignore[assignment]
    sample_block_size = int(context["knockoff_sample_block_size"])
    knockoff_random_seed = int(context["knockoff_random_seed"])
    single_peptide_error_rate_upper_bound = float(context["single_peptide_error_rate_upper_bound"])
    unique_peptide_error_source = str(
        context.get("unique_peptide_error_source", DEFAULT_UNIQUE_PEPTIDE_ERROR_SOURCE)
    )
    unique_count_power = float(
        context.get("unique_count_power", DEFAULT_UNIQUE_COUNT_POWER)
    )
    unit_empirical_initial_exclude_fraction = float(
        context.get(
            "unit_empirical_background_initial_exclude_fraction",
            UNIT_EMPIRICAL_BACKGROUND_INITIAL_EXCLUDE_FRACTION,
        )
    )
    unit_empirical_min_exclude_fraction = float(
        context.get("unit_empirical_background_min_exclude_fraction", UNIT_EMPIRICAL_BACKGROUND_MIN_EXCLUDE_FRACTION)
    )
    unit_empirical_max_exclude_fraction = float(
        context.get("unit_empirical_background_max_exclude_fraction", UNIT_EMPIRICAL_BACKGROUND_MAX_EXCLUDE_FRACTION)
    )
    unit_empirical_candidate_q = float(
        context.get("unit_empirical_background_candidate_q", UNIT_EMPIRICAL_BACKGROUND_CANDIDATE_Q)
    )
    unit_empirical_max_iterations = int(
        context.get("unit_empirical_background_max_iterations", UNIT_EMPIRICAL_BACKGROUND_MAX_ITERATIONS)
    )
    unit_empirical_threshold_quantile = float(
        context.get("unit_empirical_background_threshold_quantile", UNIT_EMPIRICAL_BACKGROUND_THRESHOLD_QUANTILE)
    )
    knockoff_top_n_raw = context.get("knockoff_top_n_targets")
    knockoff_top_n_targets = int(knockoff_top_n_raw) if knockoff_top_n_raw is not None else None
    total_theoretical_unique_peptides_all_genomes = int(context["total_theoretical_unique_peptides_all_genomes"])
    n_samples = int(args["n_samples_in_unit"])
    use_length_strata = bool(context["use_length_strata"])
    degeneracy_bin_edges: List[int] = context["degeneracy_bin_edges"]  # type: ignore[assignment]
    peptide_length_bin_edges: List[int] = context["peptide_length_bin_edges"]  # type: ignore[assignment]

    unit_pools = _unit_build_knockoff_pools_for_peptides(
        observed_peptides=unit_observed_peptides,
        peptide_deg=peptide_deg,
        peptide_score=peptide_score,
        use_length_strata=use_length_strata,
        degeneracy_bin_edges=degeneracy_bin_edges,
        peptide_length_bin_edges=peptide_length_bin_edges,
    )
    unit_shared_strata_by_genome: Dict[str, Counter] = {}
    unit_metrics_by_genome: Dict[str, dict] = {}
    unit_matched_peptides_by_genome: Dict[str, Set[str]] = {}
    unit_unique_stats_by_genome: Dict[str, dict] = {}

    p_shared_values = np.ones(n_genomes, dtype=float)
    p_unique_values = np.ones(n_genomes, dtype=float)
    p_combined_values = np.ones(n_genomes, dtype=float)
    unique_counts = np.zeros(n_genomes, dtype=int)
    null_mean_values = np.zeros(n_genomes, dtype=float)
    null_sd_values = np.zeros(n_genomes, dtype=float)
    null_p95_values = np.zeros(n_genomes, dtype=float)
    null_p99_values = np.zeros(n_genomes, dtype=float)
    z_shared_values = np.zeros(n_genomes, dtype=float)
    unit_empirical_calibration: Dict[str, object] = {}

    for genome_idx, genome_id in enumerate(genome_ids):
        matched_peptides = set(genome_matched_peptides.get(genome_id, set())).intersection(unit_observed_peptides)
        metrics = _unit_shared_metrics_for_genome(
            genome_id=genome_id,
            matched_peptides=matched_peptides,
            peptide_deg=peptide_deg,
            peptide_score=peptide_score,
            genome_total_theoretical_peptides=genome_total_theoretical_peptides,
            use_length_strata=use_length_strata,
            degeneracy_bin_edges=degeneracy_bin_edges,
            peptide_length_bin_edges=peptide_length_bin_edges,
        )
        unit_matched_peptides_by_genome[genome_id] = matched_peptides
        unit_metrics_by_genome[genome_id] = metrics
        unit_shared_strata_by_genome[genome_id] = metrics["shared_stratum_counts"]
        unique_counts[genome_idx] = int(metrics["num_peptides_unique"])

    knockoff_target_indices = _unit_knockoff_target_indices(
        genome_ids=genome_ids,
        metrics_by_genome=unit_metrics_by_genome,
        matched_counts=matched_counts,
        top_n_targets=knockoff_top_n_targets,
    )
    knockoff_target_mask = np.zeros(n_genomes, dtype=bool)
    knockoff_target_mask[knockoff_target_indices] = True

    observed_unique_pool_size = int(unique_counts.sum())
    seed_seq = np.random.SeedSequence([int(knockoff_random_seed), int(unit_idx)])
    stage_children = seed_seq.spawn(2)
    rng_stage1 = np.random.default_rng(stage_children[0])
    rng_stage2 = np.random.default_rng(stage_children[1])

    for genome_idx, genome_id in enumerate(genome_ids):
        metrics = unit_metrics_by_genome[genome_id]
        if mode == "empirical-background":
            unique_stats = _unit_empirical_unique_stats_from_row(
                pd.Series(
                    {
                        "num_peptides_unique": int(metrics["num_peptides_unique"]),
                        "p_unique_empirical_background_excess": 1.0,
                    }
                )
            )
        else:
            unique_stats = _unit_unique_pvalue_stats_for_genome(
                gid=genome_id,
                matched_peptides=unit_matched_peptides_by_genome[genome_id],
                observed_unique=int(metrics["num_peptides_unique"]),
                observed_unique_pool_size=observed_unique_pool_size,
                mode=mode,
                peptide_deg=peptide_deg,
                genome_theoretical_unique_peptides=genome_theoretical_unique_peptides,
                total_theoretical_unique_peptides_all_genomes=total_theoretical_unique_peptides_all_genomes,
                single_peptide_error_rate_upper_bound=single_peptide_error_rate_upper_bound,
                peptide_error_upper_by_peptide=peptide_error_upper_by_peptide,
                unique_peptide_error_source=unique_peptide_error_source,
                unique_count_power=unique_count_power,
            )
        unit_unique_stats_by_genome[genome_id] = unique_stats

        is_knockoff_target = bool(knockoff_target_mask[genome_idx])
        p_unique = float(unique_stats["p_unique"]) if is_knockoff_target else 1.0
        p_shared = 1.0
        mu = sd = p95 = p99 = 0.0
        if is_knockoff_target:
            p_shared, mu, sd, p95, p99 = shared_knockoff_mc(
                gid=genome_id,
                obs_shared_score=float(metrics["weighted_evidence_shared"]),
                K=K1,
                rng=rng_stage1,
                pools=unit_pools,
                counts_by_genome=unit_shared_strata_by_genome,
                sample_block_size=sample_block_size,
            )

        p_shared_values[genome_idx] = _clip_pvalue(float(p_shared))
        p_unique_values[genome_idx] = _clip_pvalue(float(p_unique))
        p_combined_values[genome_idx] = (
            _clip_pvalue(
                _unit_fisher_p_2(p1=p_shared_values[genome_idx], p2=p_unique_values[genome_idx])
            )
            if is_knockoff_target
            else 1.0
        )
        null_mean_values[genome_idx] = float(mu)
        null_sd_values[genome_idx] = float(sd)
        null_p95_values[genome_idx] = float(p95)
        null_p99_values[genome_idx] = float(p99)
        z_shared_values[genome_idx] = (
            (float(metrics["weighted_evidence_shared"]) - float(mu)) / (float(sd) + 1e-12)
            if int(metrics["matched_peptide_count_shared"]) > 0
            else 0.0
        )

    if mode == "empirical-background":
        unit_metric_df = pd.DataFrame(
            [
                {
                    "genome_id": genome_id,
                    "_genomes_with_any_match": bool(matched_counts[genome_idx] >= 1),
                    "num_peptides_unique": int(unit_metrics_by_genome[genome_id]["num_peptides_unique"]),
                    "total_peptide_count": int(unit_metrics_by_genome[genome_id]["total_peptide_count"]),
                    "unique_weighted_evidence": float(unit_metrics_by_genome[genome_id]["unique_weighted_evidence"]),
                    "weighted_evidence": float(unit_metrics_by_genome[genome_id]["weighted_evidence"]),
                }
                for genome_idx, genome_id in enumerate(genome_ids)
            ]
        )
        target_mask_empirical = unit_metric_df["_genomes_with_any_match"].astype(bool).to_numpy(dtype=bool)
        active_genomes = int(np.sum(target_mask_empirical))
        small_unit_warning = ""
        min_fraction = float(np.clip(unit_empirical_min_exclude_fraction, 0.0, 1.0))
        max_fraction = float(np.clip(unit_empirical_max_exclude_fraction, min_fraction, 1.0))
        exclude_fraction = float(np.clip(unit_empirical_initial_exclude_fraction, min_fraction, max_fraction))
        max_iterations = int(max(1, unit_empirical_max_iterations))
        candidate_q = float(np.clip(unit_empirical_candidate_q, 0.0, 1.0))
        threshold_quantile = float(np.clip(unit_empirical_threshold_quantile, 0.0, 1.0))
        n_bins = 8
        if active_genomes < UNIT_EMPIRICAL_BACKGROUND_SMALL_UNIT_MIN_ACTIVE_GENOMES:
            exclude_fraction = 0.0
            n_bins = 1
            small_unit_warning = (
                "active matched genomes < "
                f"{UNIT_EMPIRICAL_BACKGROUND_SMALL_UNIT_MIN_ACTIVE_GENOMES}; "
                "using top_exclude_fraction=0.0 and n_bins=1"
            )
        iteration_trace: List[dict] = []

        def _apply_unit_empirical_stats(stats_df: pd.DataFrame) -> float:
            stats_ordered = stats_df.reset_index(drop=True)
            for genome_idx, row in stats_ordered.iterrows():
                genome_id = genome_ids[int(genome_idx)]
                unique_stats = _unit_empirical_unique_stats_from_row(row, mode=mode)
                unit_unique_stats_by_genome[genome_id] = unique_stats
                is_knockoff_target = bool(knockoff_target_mask[int(genome_idx)])
                p_unique_values[int(genome_idx)] = (
                    _clip_pvalue(float(unique_stats["p_unique"]))
                    if is_knockoff_target
                    else 1.0
                )
                if is_knockoff_target:
                    p_combined_values[int(genome_idx)] = _clip_pvalue(
                        _unit_fisher_p_2(
                            p1=float(p_shared_values[int(genome_idx)]),
                            p2=float(p_unique_values[int(genome_idx)]),
                        )
                    )
                else:
                    p_combined_values[int(genome_idx)] = 1.0

            q_tmp = np.ones(n_genomes, dtype=float)
            if bool(np.any(target_mask_empirical)):
                q_tmp[target_mask_empirical] = _unit_bh_qvalues(p_combined_values[target_mask_empirical])
                return float(np.sum(q_tmp[target_mask_empirical] <= candidate_q)) / float(
                    max(int(np.sum(target_mask_empirical)), 1)
                )
            return 0.0

        if bool(np.any(target_mask_empirical)) and not small_unit_warning:
            for _ in range(max_iterations):
                empirical_df, _ = _compute_empirical_background_stats_for_table(
                    unit_metric_df,
                    alpha=single_peptide_error_rate_upper_bound,
                    top_exclude_fraction=exclude_fraction,
                    threshold_quantile=threshold_quantile,
                    n_bins=n_bins,
                    min_bin_size=50,
                )
                candidate_fraction = _apply_unit_empirical_stats(empirical_df)
                new_exclude_fraction = float(np.clip(candidate_fraction, min_fraction, max_fraction))
                iteration_trace.append(
                    {
                        "iteration": int(len(iteration_trace) + 1),
                        "exclude_fraction": float(exclude_fraction),
                        "candidate_fraction": float(candidate_fraction),
                        "new_exclude_fraction": float(new_exclude_fraction),
                    }
                )
                if abs(new_exclude_fraction - exclude_fraction) < 0.01:
                    exclude_fraction = new_exclude_fraction
                    break
                exclude_fraction = new_exclude_fraction

        empirical_df, _ = _compute_empirical_background_stats_for_table(
            unit_metric_df,
            alpha=single_peptide_error_rate_upper_bound,
            top_exclude_fraction=exclude_fraction,
            threshold_quantile=threshold_quantile,
            n_bins=n_bins,
            min_bin_size=50,
        )
        _apply_unit_empirical_stats(empirical_df)
        if small_unit_warning:
            iteration_trace.append(
                {
                    "iteration": 1,
                    "exclude_fraction": 0.0,
                    "candidate_fraction": None,
                    "new_exclude_fraction": 0.0,
                    "warning": small_unit_warning,
                }
            )
        unit_empirical_calibration = {
            "analysis_unit_id": unit_id,
            "unit_empirical_background_iteration_trace": json.dumps(iteration_trace, separators=(",", ":")),
            "unit_empirical_background_threshold_quantile": float(threshold_quantile),
            "unit_empirical_background_final_exclude_fraction": float(exclude_fraction),
            "unit_empirical_background_iterations": int(len(iteration_trace)),
            "unit_empirical_background_active_genomes": int(active_genomes),
            "unit_empirical_background_warning": small_unit_warning,
        }

    if K2 is not None and ranges:
        candidate_mask = knockoff_target_mask & np.asarray(
            [_unit_in_any_stage2_range(value, ranges) for value in p_combined_values],
            dtype=bool,
        )
        candidate_indices = np.asarray(
            [index for index in knockoff_target_indices if bool(candidate_mask[int(index)])],
            dtype=int,
        )
        for genome_idx in candidate_indices:
            genome_id = genome_ids[int(genome_idx)]
            metrics = unit_metrics_by_genome[genome_id]
            p_shared, mu, sd, p95, p99 = shared_knockoff_mc(
                gid=genome_id,
                obs_shared_score=float(metrics["weighted_evidence_shared"]),
                K=K2,
                rng=rng_stage2,
                pools=unit_pools,
                counts_by_genome=unit_shared_strata_by_genome,
                sample_block_size=sample_block_size,
            )
            p_shared_values[genome_idx] = _clip_pvalue(float(p_shared))
            p_combined_values[genome_idx] = _clip_pvalue(
                _unit_fisher_p_2(p1=p_shared_values[genome_idx], p2=p_unique_values[genome_idx])
            )
            null_mean_values[genome_idx] = float(mu)
            null_sd_values[genome_idx] = float(sd)
            null_p95_values[genome_idx] = float(p95)
            null_p99_values[genome_idx] = float(p99)
            z_shared_values[genome_idx] = (
                (float(metrics["weighted_evidence_shared"]) - float(mu)) / (float(sd) + 1e-12)
                if int(metrics["matched_peptide_count_shared"]) > 0
                else 0.0
            )

    qvals = np.ones(n_genomes, dtype=float)
    target_mask = matched_counts >= 1
    if bool(np.any(target_mask)):
        qvals[target_mask] = _unit_bh_qvalues(p_combined_values[target_mask])
    presence_scores = qvalues_to_presence_scores(qvals)
    rank_order = np.lexsort((np.asarray(genome_ids, dtype=object), -matched_counts, -unique_counts, qvals))
    ranks = np.empty(n_genomes, dtype=int)
    ranks[rank_order] = np.arange(1, n_genomes + 1, dtype=int)

    rows = []
    for genome_idx, genome_id in enumerate(genome_ids):
        metrics = unit_metrics_by_genome[genome_id]
        unique_stats = unit_unique_stats_by_genome[genome_id]
        rows.append(
            {
                "analysis_unit_id": unit_id,
                "genome_id": genome_id,
                "Lineage": lineage_map.get(genome_id, pd.NA),
                "num_peptides_matched": int(metrics["num_peptides_matched"]),
                "num_peptides_unique": int(metrics["num_peptides_unique"]),
                "unique_effective_count": float(unique_stats["unique_effective_count"]),
                "matched_peptide_count_shared": int(metrics["matched_peptide_count_shared"]),
                "effective_peptide_count_shared": float(metrics["effective_peptide_count_shared"]),
                "weighted_evidence_shared": float(metrics["weighted_evidence_shared"]),
                "effective_peptide_count": float(metrics["effective_peptide_count"]),
                "weighted_evidence": float(metrics["weighted_evidence"]),
                "unique_weighted_evidence": float(metrics["unique_weighted_evidence"]),
                "shared_fraction": float(metrics["shared_fraction"]),
                "theoretical_unique_peptides": unique_stats["theoretical_unique_peptides"],
                "observed_unique_peptide_pool_size": int(observed_unique_pool_size),
                "expected_unique_null": float(unique_stats["unique_expected_null"]),
                "unique_depth_fold": float(unique_stats["unique_depth_fold"]),
                "unique_depth_null_model": str(unique_stats["unique_depth_null_model"]),
                "unique_pvalue_count_model": str(unique_stats["unique_pvalue_count_model"]),
                "has_unique_evidence": bool(unique_stats["has_unique_evidence"]),
                "pvalue_unique": float(p_unique_values[genome_idx]),
                "pvalue_unique_depth": float(unique_stats["p_unique_depth"]),
                "unique_empirical_background_bin": str(unique_stats.get("unique_empirical_background_bin", "")),
                "unique_empirical_background_size": int(unique_stats.get("unique_empirical_background_size", 0)),
                "unique_empirical_background_threshold": float(
                    unique_stats.get("unique_empirical_background_threshold", 0.0)
                ),
                "unique_empirical_excess_count": float(unique_stats.get("unique_empirical_excess_count", 0.0)),
                "p_unique_empirical_tail": float(unique_stats.get("p_unique_empirical_tail", 1.0)),
                "pvalue_shared": float(p_shared_values[genome_idx]),
                "knockoff_target": bool(knockoff_target_mask[genome_idx]),
                "pvalue": float(p_combined_values[genome_idx]),
                "qvalue": float(qvals[genome_idx]),
                "presence_score": float(presence_scores[genome_idx]),
                "presence_rank": int(ranks[genome_idx]),
                "pass_q_0_01": bool(qvals[genome_idx] <= 0.01 and int(metrics["num_peptides_matched"]) >= 1),
                "pass_q_0_05": bool(qvals[genome_idx] <= 0.05 and int(metrics["num_peptides_matched"]) >= 1),
                "null_mean_shared": float(null_mean_values[genome_idx]),
                "null_sd_shared": float(null_sd_values[genome_idx]),
                "null_p95_shared": float(null_p95_values[genome_idx]),
                "null_p99_shared": float(null_p99_values[genome_idx]),
                "z_shared": float(z_shared_values[genome_idx]),
                "n_samples_in_unit": int(n_samples),
                "unit_presence_rule": "union",
                "unit_shared_mode": "per-unit",
            }
        )

    return {
        "unit_idx": int(unit_idx),
        "analysis_unit_id": unit_id,
        "rows": rows,
        "knockoff_target_genomes": int(np.sum(knockoff_target_mask)),
        "unit_empirical_background_calibration": unit_empirical_calibration,
    }
