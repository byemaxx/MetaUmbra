# Genome existence scoring from a peptide list using peptide-space knockoff null
# Version: 4.6
# Date: 2026-02-13
#
# Workflow:
# 1) Read observed peptide list (optional peptide-level error-probability filter, e.g. PEP/FDR).
# 2) For each genome (theoretical digest peptides), compute matched peptides = observed ∩ theoretical.
# 3) Compute peptide degeneracy d(p) across TARGET genomes (recommended).
# 4) Compute shared-aware evidence:
#       w(p)=1/d(p)
#       weighted_evidence = Σ w(p)*score(p)
#       unique_weighted_evidence = Σ score(p) for d(p)=1
#       weighted_evidence_shared = Σ w(p)*score(p) for d(p)>1
# 5) Peptide-space knockoff (no second database matching):
#    - Build pools of shared peptide contributions (w*s) stratified by degeneracy (and optional length).
#    - For each genome, sample from these pools according to that genome's shared-stratum counts to get
#      an empirical null for weighted_evidence_shared, yielding p_shared_knock.
#    - Unique evidence p-value uses an adaptive peptide-depth null by default.
#    - Combine with Fisher (2 p-values) => p_presence; BH => q_presence (per-genome existence q-value).
#
# Outputs:
# - Main result table: concise, publication-facing columns with standardized names.
# - Artifacts folder: full internal metrics and diagnostics tables.

import os
import time
import pickle
import random
import logging
import hashlib
import tempfile
import multiprocessing as mp
import concurrent.futures
from pathlib import Path
from typing import Optional, List, Dict, Set, Tuple, Union, FrozenSet
from collections import Counter, defaultdict
import json
import sys
import platform

import numpy as np
import pandas as pd
from tqdm import tqdm


WINDOWS_MAX_PROCESS_POOL_WORKERS = 60
MIN_PVALUE = 1e-300
THEORETICAL_OPPORTUNITY_CACHE_VERSION = 2
THEORETICAL_OPPORTUNITY_MAX_SHARDS = 256
COUNT_DTYPE = np.int32
DEFAULT_UNIQUE_PVALUE_MODE = "alpha-upper-bound"
DEFAULT_UNIQUE_COUNT_POWER = 0.6
DEFAULT_UNIQUE_PEPTIDE_ERROR_SOURCE = "global-alpha"
UNIQUE_PVALUE_CANONICAL_MODES = (
    "hypergeometric-opportunity",
    "alpha-upper-bound",
)
UNIQUE_PEPTIDE_ERROR_SOURCES = (
    "global-alpha",
    "peptide-error-column",
)


def _normalize_unique_pvalue_mode(mode: Optional[str]) -> str:
    normalized = str(mode or DEFAULT_UNIQUE_PVALUE_MODE).strip().lower()
    if normalized not in UNIQUE_PVALUE_CANONICAL_MODES:
        choices = "', '".join(UNIQUE_PVALUE_CANONICAL_MODES)
        raise ValueError(f"unique_pvalue_mode must be one of '{choices}'.")
    return normalized


def _normalize_unique_peptide_error_source(source: Optional[str]) -> str:
    normalized = str(source or DEFAULT_UNIQUE_PEPTIDE_ERROR_SOURCE).strip().lower()
    if normalized not in UNIQUE_PEPTIDE_ERROR_SOURCES:
        choices = "', '".join(UNIQUE_PEPTIDE_ERROR_SOURCES)
        raise ValueError(f"unique_peptide_error_source must be one of '{choices}'.")
    return normalized


def _effective_unique_count(
    u_raw: int,
    power: float = DEFAULT_UNIQUE_COUNT_POWER,
    cap: Optional[float] = None,
) -> float:
    """Convert a raw unique peptide count into a tempered effective count."""
    u_raw = int(max(u_raw, 0))
    if u_raw == 0:
        return 0.0

    power = float(power)
    if not np.isfinite(power) or not (0 < power <= 1):
        raise ValueError("unique_count_power must be in the interval (0, 1].")

    u_eff = min(float(u_raw), float(u_raw) ** power)
    if cap is not None:
        cap_value = float(cap)
        if not np.isfinite(cap_value) or cap_value <= 0:
            raise ValueError("unique_count_cap must be positive or None.")
        u_eff = min(u_eff, cap_value)

    return float(max(u_eff, 0.0))


def _tempered_unique_error_product_pvalue(
    unique_peptides: List[str],
    alpha: float,
    peptide_error_upper_by_peptide: Dict[str, float],
    error_source: str,
    unique_count_power: float,
    unique_count_cap: Optional[float],
) -> Tuple[float, float, str]:
    u_raw = int(len(unique_peptides))
    if u_raw <= 0:
        return 1.0, 0.0, "none"

    error_source = _normalize_unique_peptide_error_source(error_source)
    u_eff = _effective_unique_count(
        u_raw=u_raw,
        power=unique_count_power,
        cap=unique_count_cap,
    )
    temper = float(u_eff) / float(max(u_raw, 1))

    if error_source == "global-alpha":
        log_eps_sum = float(u_raw) * float(np.log(alpha))
    else:
        missing = [peptide for peptide in unique_peptides if peptide not in peptide_error_upper_by_peptide]
        if missing:
            preview = ", ".join(sorted(missing)[:10])
            suffix = " ..." if len(missing) > 10 else ""
            raise ValueError(
                "unique_peptide_error_source='peptide-error-column' requires an error value for every unique peptide. "
                f"Missing {len(missing)} peptide(s): {preview}{suffix}"
            )
        errs = np.asarray([peptide_error_upper_by_peptide[peptide] for peptide in unique_peptides], dtype=float)
        errs = np.clip(errs, 1e-12, 1.0)
        log_eps_sum = float(np.sum(np.log(errs)))

    log_p = float(log_eps_sum) * temper
    p_unique = float(np.exp(max(log_p, np.log(MIN_PVALUE))))
    return _clip_pvalue(p_unique), float(u_eff), error_source


def _strip_raw_suffix_from_sample_ids(values: pd.Series) -> pd.Series:
    """Normalize DIA-NN parquet sample IDs such as sample.raw to sample."""
    return values.astype("string").str.strip().str.replace(r"\.raw$", "", case=False, regex=True)


def _drop_duplicate_pairs_with_pyarrow(
    df: pd.DataFrame,
    first_col: str,
    second_col: str,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Use Arrow's threaded hash group-by as a parallel distinct for two string columns."""
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required for threaded parquet unit-aware pair construction."
        ) from exc

    if logger is not None:
        logger.info(
            f"Using pyarrow threaded group-by to deduplicate {len(df)} peptide-sample row(s) ..."
        )
    table = pa.Table.from_pandas(df[[first_col, second_col]], preserve_index=False)
    unique_table = table.group_by([first_col, second_col], use_threads=True).aggregate([])
    return unique_table.to_pandas()


def _clip_pvalue(p: float) -> float:
    return float(np.clip(float(p), MIN_PVALUE, 1.0))


def _resolve_worker_count(num_workers: Optional[int], logger: Optional[logging.Logger] = None) -> int:
    """Clamp worker count to platform-supported and CPU-backed limits."""
    cpu_count = mp.cpu_count() or 1

    if num_workers is None:
        resolved = max(1, cpu_count - 1)
    else:
        resolved = max(1, int(num_workers))

    if sys.platform == "win32" and resolved > WINDOWS_MAX_PROCESS_POOL_WORKERS:
        if logger is not None:
            logger.warning(
                "Windows system detected: num_workers=%s exceeds ProcessPoolExecutor limit; adjusted to %s",
                resolved,
                WINDOWS_MAX_PROCESS_POOL_WORKERS,
            )
        resolved = WINDOWS_MAX_PROCESS_POOL_WORKERS

    if resolved > cpu_count:
        if logger is not None:
            logger.warning(
                "num_workers=%s exceeds available CPU cores=%s; adjusted to %s",
                resolved,
                cpu_count,
                cpu_count,
            )
        resolved = cpu_count

    return resolved


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


def _unit_mc_sum_from_pool(
    pool: Optional[np.ndarray],
    K: int,
    c: int,
    rng: np.random.Generator,
    sample_block_size: int,
) -> np.ndarray:
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


def _unit_p_shared_knockoff_mc(
    gid: str,
    obs_shared_score: float,
    K: int,
    rng: np.random.Generator,
    pools: Dict[Union[int, Tuple[int, int]], np.ndarray],
    counts_by_genome: Dict[str, Counter],
    sample_block_size: int,
) -> Tuple[float, float, float, float, float]:
    counts = counts_by_genome.get(gid, None)
    if not counts:
        return 1.0, 0.0, 0.0, 0.0, 0.0

    null_sum = np.zeros(int(K), dtype=np.float64)
    for key, c in counts.items():
        pool = pools.get(key, None)
        null_sum += _unit_mc_sum_from_pool(
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


def _unit_fisher_p_2(p1: float, p2: float) -> float:
    p1 = float(min(max(p1, 1e-300), 1.0))
    p2 = float(min(max(p2, 1e-300), 1.0))
    stat = -2.0 * (np.log(p1) + np.log(p2))
    x = float(stat)
    return float(np.exp(-x / 2.0) * (1.0 + x / 2.0))


def _unit_bh_qvalues(pvals: np.ndarray) -> np.ndarray:
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
    min_unique_for_unique_pvalue: int,
    mode: str,
    peptide_deg: Dict[str, int],
    genome_theoretical_unique_peptides: Dict[str, int],
    total_theoretical_unique_peptides_all_genomes: int,
    single_peptide_error_rate_upper_bound: float,
    peptide_error_upper_by_peptide: Dict[str, float],
    unique_peptide_error_source: str = DEFAULT_UNIQUE_PEPTIDE_ERROR_SOURCE,
    unique_count_power: float = DEFAULT_UNIQUE_COUNT_POWER,
    unique_count_cap: Optional[float] = None,
) -> dict:
    U = int(observed_unique)
    mode = _normalize_unique_pvalue_mode(mode)

    alpha = float(min(max(single_peptide_error_rate_upper_bound, 1e-12), 1.0))
    S = int(observed_unique_pool_size)
    A = int(genome_theoretical_unique_peptides.get(gid, 0))
    A_total = int(max(total_theoretical_unique_peptides_all_genomes, 1))
    theoretical_unique: object = pd.NA
    p_unique = 1.0
    p_unique_depth = 1.0
    expected = 0.0
    fold = 0.0
    gate_pass = False
    null_model = ""
    unique_effective_count = float(U)
    count_model = "raw"

    if mode == "hypergeometric-opportunity":
        theoretical_unique = int(A)
        expected = float(S) * float(A) / float(A_total) if A_total > 0 else 0.0
        fold = float(U) / max(expected, 1e-12)
        gate_pass = bool(U >= int(min_unique_for_unique_pvalue))
        null_model = "hypergeometric"
        if gate_pass and S > 0 and A > 0 and A_total > 0:
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
            unique_count_cap=unique_count_cap,
        )
        count_model = f"power:{float(unique_count_power):g}"
        p_unique_depth = p_unique
        gate_pass = U > 0
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
        "unique_gate_pass": bool(gate_pass),
        "theoretical_unique_peptides": theoretical_unique,
        "unique_effective_count": float(unique_effective_count),
        "unique_pvalue_count_model": count_model,
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


def _init_unit_aware_worker(context: Dict[str, object]) -> None:
    _UNIT_WORKER_CONTEXT.clear()
    _UNIT_WORKER_CONTEXT.update(context)


def _compute_unit_aware_single_unit_worker(args: Dict[str, object]) -> Dict[str, object]:
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
    gate_min = int(context["min_unique_for_unique_pvalue"])
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
    unique_count_cap = context.get("unique_count_cap")
    total_theoretical_unique_peptides_all_genomes = int(context["total_theoretical_unique_peptides_all_genomes"])
    n_samples = int(args["n_samples_in_unit"])
    use_length_strata = bool(context["use_length_strata"])
    degeneracy_bin_edges: List[int] = context["degeneracy_bin_edges"]  # type: ignore[assignment]
    peptide_length_bin_edges: List[int] = context["peptide_length_bin_edges"]  # type: ignore[assignment]
    shared_knockoff_func = context.get("shared_knockoff_func")

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

    observed_unique_pool_size = int(unique_counts.sum())
    seed_seq = np.random.SeedSequence([int(knockoff_random_seed), int(unit_idx)])
    stage_children = seed_seq.spawn(2)
    rng_stage1 = np.random.default_rng(stage_children[0])
    rng_stage2 = np.random.default_rng(stage_children[1])

    for genome_idx, genome_id in enumerate(genome_ids):
        metrics = unit_metrics_by_genome[genome_id]
        unique_stats = _unit_unique_pvalue_stats_for_genome(
            gid=genome_id,
            matched_peptides=unit_matched_peptides_by_genome[genome_id],
            observed_unique=int(metrics["num_peptides_unique"]),
            observed_unique_pool_size=observed_unique_pool_size,
            min_unique_for_unique_pvalue=gate_min,
            mode=mode,
            peptide_deg=peptide_deg,
            genome_theoretical_unique_peptides=genome_theoretical_unique_peptides,
            total_theoretical_unique_peptides_all_genomes=total_theoretical_unique_peptides_all_genomes,
            single_peptide_error_rate_upper_bound=single_peptide_error_rate_upper_bound,
            peptide_error_upper_by_peptide=peptide_error_upper_by_peptide,
            unique_peptide_error_source=unique_peptide_error_source,
            unique_count_power=unique_count_power,
            unique_count_cap=unique_count_cap,
        )
        unit_unique_stats_by_genome[genome_id] = unique_stats

        p_unique = float(unique_stats["p_unique"])
        p_shared = 1.0
        mu = sd = p95 = p99 = 0.0
        if int(metrics["num_peptides_matched"]) > 0:
            if callable(shared_knockoff_func):
                result = shared_knockoff_func(
                    gid=genome_id,
                    obs_shared_score=float(metrics["weighted_evidence_shared"]),
                    K=K1,
                    rng=rng_stage1,
                    return_moments=True,
                    pools=unit_pools,
                    counts_by_genome=unit_shared_strata_by_genome,
                )
                p_shared, mu, sd, p95, p99 = (
                    result if isinstance(result, tuple) else (float(result), 0.0, 0.0, 0.0, 0.0)
                )
            else:
                p_shared, mu, sd, p95, p99 = _unit_p_shared_knockoff_mc(
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

    if K2 is not None and ranges:
        target_mask = matched_counts >= 1
        candidate_mask = target_mask & np.asarray(
            [_unit_in_any_stage2_range(value, ranges) for value in p_combined_values],
            dtype=bool,
        )
        candidate_indices = np.where(candidate_mask)[0]
        for genome_idx in candidate_indices:
            genome_id = genome_ids[int(genome_idx)]
            metrics = unit_metrics_by_genome[genome_id]
            if callable(shared_knockoff_func):
                result = shared_knockoff_func(
                    gid=genome_id,
                    obs_shared_score=float(metrics["weighted_evidence_shared"]),
                    K=K2,
                    rng=rng_stage2,
                    return_moments=True,
                    pools=unit_pools,
                    counts_by_genome=unit_shared_strata_by_genome,
                )
                p_shared, mu, sd, p95, p99 = (
                    result if isinstance(result, tuple) else (float(result), 0.0, 0.0, 0.0, 0.0)
                )
            else:
                p_shared, mu, sd, p95, p99 = _unit_p_shared_knockoff_mc(
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
    presence_scores = -np.log10(np.clip(qvals, 1e-300, 1.0))
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
                "unique_gate_pass": bool(unique_stats["unique_gate_pass"]),
                "pvalue_unique": float(p_unique_values[genome_idx]),
                "pvalue_unique_depth": float(unique_stats["p_unique_depth"]),
                "pvalue_shared": float(p_shared_values[genome_idx]),
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
    }


# =========================
# Logging
# =========================
def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    """Set up logger with console (and optional file) handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# =========================
# Genome scan workers
# =========================
_OBS_PEPTIDES_WORKER: Union[Set[str], FrozenSet[str]] = set()
_AA_ONLY_PATTERN = r"^[ACDEFGHIKLMNPQRSTVWYBXZJUO]+$"


def _init_genome_batch_worker(obs_peptides: Union[Set[str], FrozenSet[str]]) -> None:
    """ProcessPool initializer: set read-only observed peptide universe once per worker."""
    global _OBS_PEPTIDES_WORKER
    _OBS_PEPTIDES_WORKER = obs_peptides


def _read_unique_peptides_from_digest(genome_peptides_path: Union[str, os.PathLike]) -> Set[str]:
    """Read unique theoretical peptides from a digest TSV."""
    genome_peptides_path = str(genome_peptides_path)
    fallback_to_first_col = False
    fallback_available_columns: Optional[List[str]] = None
    try:
        chunk_iter = pd.read_csv(
            genome_peptides_path,
            sep="\t",
            usecols=["Peptide"],
            dtype={"Peptide": "string"},
            engine="c",
            chunksize=50000,
        )
        peptide_column_name = "Peptide"
    except ValueError:
        fallback_to_first_col = True
        try:
            fallback_available_columns = pd.read_csv(genome_peptides_path, sep="\t", nrows=0).columns.tolist()
        except Exception:
            fallback_available_columns = None
        chunk_iter = pd.read_csv(
            genome_peptides_path,
            sep="\t",
            usecols=[0],
            dtype="string",
            engine="c",
            chunksize=50000,
        )
        peptide_column_name = None

    seen_theoretical: Set[str] = set()
    fallback_sanity_checked = False
    for chunk_df in chunk_iter:
        col_name = peptide_column_name if peptide_column_name is not None else chunk_df.columns[0]
        col_series = chunk_df[col_name].dropna().astype(str).str.strip()

        if fallback_to_first_col and not fallback_sanity_checked:
            sample = col_series[col_series != ""].head(200)
            if not sample.empty:
                is_aa_only = sample.str.upper().str.fullmatch(_AA_ONLY_PATTERN)
                if not bool(is_aa_only.all()):
                    bad_examples = sample[~is_aa_only].head(5).tolist()
                    cols_for_error = fallback_available_columns if fallback_available_columns else chunk_df.columns.tolist()
                    available_cols = ", ".join(map(str, cols_for_error))
                    raise ValueError(
                        "Fallback to first column failed sanity check: values are not peptide-like "
                        f"(AA letters only). Available columns: [{available_cols}]. "
                        f"Please provide a 'Peptide' column. Non-peptide examples: {bad_examples}"
                    )
                fallback_sanity_checked = True

        chunk_unique = set(col_series[col_series != ""].values.tolist())
        if chunk_unique:
            seen_theoretical.update(chunk_unique)

    return seen_theoretical


def _stable_theoretical_shard_index(peptide: str, shard_count: int) -> int:
    """Return a deterministic shard index for a peptide string."""
    if int(shard_count) <= 1:
        return 0
    digest = hashlib.blake2b(str(peptide).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) % int(shard_count)


def _build_theoretical_opportunity_batch_worker(
    batch_index: int,
    file_paths: List[Union[str, os.PathLike]],
    shard_count: int,
    temp_dir: str,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Read digest TSV files and write peptide/genome pairs into local shard files."""
    genome_total_theoretical_peptides: Dict[str, int] = {}
    shard_paths: Dict[int, str] = {}
    open_handles: Dict[int, object] = {}

    try:
        for genome_peptides_path in file_paths:
            genome_peptides_path = str(genome_peptides_path)
            genome_id = Path(genome_peptides_path).stem
            peptides = _read_unique_peptides_from_digest(genome_peptides_path)
            genome_total_theoretical_peptides[genome_id] = int(len(peptides))

            for peptide in peptides:
                shard_index = _stable_theoretical_shard_index(peptide, shard_count)
                handle = open_handles.get(shard_index)
                if handle is None:
                    shard_path = os.path.join(
                        temp_dir,
                        f"batch_{int(batch_index):05d}_shard_{int(shard_index):05d}.tsv",
                    )
                    handle = open(shard_path, "w", encoding="utf-8", newline="")
                    open_handles[shard_index] = handle
                    shard_paths[shard_index] = shard_path
                handle.write(f"{peptide}\t{genome_id}\n")
    finally:
        for handle in open_handles.values():
            try:
                handle.close()
            except Exception:
                pass

    return genome_total_theoretical_peptides, shard_paths


def _process_theoretical_opportunity_shard_worker(
    shard_index: int,
    shard_paths: List[Union[str, os.PathLike]],
) -> Tuple[int, Dict[str, int]]:
    """Reduce one theoretical-opportunity shard into unique counts by genome."""
    del shard_index
    peptide_owner: Dict[str, str] = {}
    genome_theoretical_unique_peptides: Counter = Counter()
    shared_marker = ""

    for shard_path in shard_paths:
        with open(shard_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                peptide, genome_id = line.split("\t", 1)
                owner = peptide_owner.get(peptide)
                if owner is None:
                    peptide_owner[peptide] = genome_id
                    genome_theoretical_unique_peptides[genome_id] += 1
                elif owner != genome_id and owner != shared_marker:
                    genome_theoretical_unique_peptides[owner] -= 1
                    peptide_owner[peptide] = shared_marker

    return int(len(peptide_owner)), {
        str(genome_id): int(count)
        for genome_id, count in genome_theoretical_unique_peptides.items()
        if int(count) != 0
    }


def _process_genome_batch_worker(file_paths: List[Union[str, os.PathLike]]) -> List[Tuple[str, Set[str], int, Optional[str]]]:
    """
    Process a batch of genome peptide files.

    Returns tuples:
    - (genome_id, matched_peptides, total_theoretical_unique_peptides, error_message)
    - error_message is None on success; non-empty on failure.
    """
    results: List[Tuple[str, Set[str], int, Optional[str]]] = []

    for genome_peptides_path in file_paths:
        genome_peptides_path = str(genome_peptides_path)
        genome_id = Path(genome_peptides_path).stem
        try:
            seen_theoretical = _read_unique_peptides_from_digest(genome_peptides_path)
            matched_peptides = set(seen_theoretical).intersection(_OBS_PEPTIDES_WORKER)
            results.append((genome_id, matched_peptides, len(seen_theoretical), None))
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            results.append((genome_id, set(), 0, err))

    return results


# =========================
# Main Calculator
# =========================
class GenomePresenceScorer:
    """
    Genome-level existence scoring from an observed peptide table by mapping onto per-genome theoretical peptide files.

    Clean v4.1:
    - Shared peptide degeneracy d(p) is computed across TARGET genomes (recommended).
    - A peptide-space knockoff null is used to estimate a per-genome existence p/q-value without requiring
      any second database matching.
    - Extra paper-friendly artifacts exported into output_dir/<result_stem>_artifacts/.
    """

    def __init__(self, num_workers: Optional[int] = None, log_file: Optional[str] = None):
        self.logger = setup_logger("GenomePresenceScorer", log_file)
        self.num_workers = _resolve_worker_count(num_workers, logger=self.logger)

        # Core states
        self.peptide_score: Dict[str, float] = {}  # peptide -> normalized score in [0,1] (or 1.0)
        self.peptide_error_cutoff: float = 0.05    # input peptide error/FDR filtering threshold
        # Upper bound on per-peptide false match probability used for unique-evidence p-value bound.
        self.single_peptide_error_rate_upper_bound: float = 1.0
        self.peptide_table_dir: Optional[str] = None

        self.genome_matched_peptides: Dict[str, Set[str]] = {}  # genome -> matched peptides (observed ∩ theoretical)
        self.genome_total_theoretical_peptides: Dict[str, int] = {}  # genome -> total theoretical peptides count
        self.genome_theoretical_unique_peptides: Dict[str, int] = {}
        self.theoretical_peptide_universe_size: int = 0
        self.total_theoretical_peptides_all_genomes: int = 0
        self.observed_matchable_peptides: int = 0
        self.observed_unique_peptide_pool_size: int = 0
        self.total_theoretical_unique_peptides_all_genomes: int = 0
        self.min_unique_for_unique_pvalue: int = 3
        self.unique_pvalue_mode: str = DEFAULT_UNIQUE_PVALUE_MODE
        self.unique_peptide_error_source: str = DEFAULT_UNIQUE_PEPTIDE_ERROR_SOURCE
        self.unique_count_power: float = DEFAULT_UNIQUE_COUNT_POWER
        self.unique_count_cap: Optional[float] = None
        self.genome_scores_df: Optional[pd.DataFrame] = None

        # Unified ranking score scales (lexicographic; unique dominates)
        self.rank_lexico_scales = {
            "U": 10**12,   # num_peptides_unique
            "UW": 10**9,   # unique_weighted_evidence
            "WE": 10**6,   # weighted_evidence
            "EP": 10**3,   # effective_peptide_count
            "MR": 10**5,   # peptide_match_ratio (0..1)
            "M": 1         # num_peptides_matched
        }

        # Degeneracy bins: 1 | 2-5 | 6-20 | 21-100 | 101-500 | >500
        self.degeneracy_bin_edges = [1, 5, 20, 100, 500]

        # Optional length bins (disabled by default for speed)
        self.use_length_strata = False
        self.peptide_length_bin_edges = [7, 10, 14, 20, 30]

        # Monte Carlo parameters
        self.knockoff_mc_iterations: int = 500
        self.knockoff_sample_block_size: int = 128
        self.knockoff_random_seed: int = 1

        # Optional: 2-stage Monte Carlo (speed/accuracy tradeoff)
        # Stage 1: run all TARGET genomes with K=self.knockoff_mc_iterations (fast screen).
        # Stage 2: recompute ONLY for genomes whose stage-1 p_presence is in given ranges, using a larger K.
        # If knockoff_stage2_mc_iterations is None, 2-stage refinement is disabled.
        self.knockoff_stage2_mc_iterations: Optional[int] = None
        self.knockoff_stage2_p_exist_ranges: List[Tuple[float, float]] = [(0.01, 0.05)]

        # Optional speed knob: compute knockoff only for top-N TARGET genomes by rank, set others p=1
        self.knockoff_top_n_targets: Optional[int] = None

        # Internal caches for knockoff
        self.peptide_degeneracy: Optional[Dict[str, int]] = None
        self.knockoff_pools_weighted_contrib: Optional[Dict[Union[int, Tuple[int, int]], np.ndarray]] = None
        self.knockoff_shared_stratum_counts_by_genome: Dict[str, Counter] = {}  # genome_id -> Counter(stratum -> count)

        # --- NEW: paper-friendly run diagnostics ---
        self.run_stats: Dict[str, object] = {}
        self.timing_stats: Dict[str, float] = {}
        self.knockoff_pool_stats: Optional[pd.DataFrame] = None
        self.peptide_error_upper_by_peptide: Dict[str, float] = {}  # peptide -> per-peptide upper bound (from error column)
        self.genome_lineage_df: Optional[pd.DataFrame] = None
        self.unit_aware_enabled: bool = False
        self.unit_presence_rule: str = "union"
        self.unit_shared_mode: str = "per-unit"
        self.unit_sample_ids: List[str] = []
        self.unit_analysis_unit_ids: List[str] = []
        self.unit_peptides: List[str] = []
        self.unit_peptide_index: Dict[str, int] = {}
        self.unit_presence_matrix = None  # sparse peptide x analysis_unit matrix
        self.unit_sample_counts: Dict[str, int] = {}
        self.sample_unit_mapping_df: Optional[pd.DataFrame] = None
        self.unit_aware_output_paths: Dict[str, str] = {}
        self.unit_aware_cohort_summary_df: Optional[pd.DataFrame] = None
        self._export_unit_derived_tables: bool = False
        self._last_unit_genome_presence_df: Optional[pd.DataFrame] = None

    def _read_genome_lineage_table(
        self,
        genome_lineage_table_path: str,
        genome_lineage_genome_id_col: str,
        genome_lineage_lineage_col: str,
    ) -> pd.DataFrame:
        """Read a genome->Lineage mapping table using explicitly provided column names."""
        path = str(genome_lineage_table_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Genome lineage table not found: {path}")

        df = pd.read_csv(path, sep="\t", dtype="string")
        if df.empty:
            raise ValueError(f"Genome lineage table is empty: {path}")

        genome_col = str(genome_lineage_genome_id_col).strip()
        lineage_col = str(genome_lineage_lineage_col).strip()
        missing_cols = [col for col in [genome_col, lineage_col] if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Genome lineage table is missing required columns: {missing_cols}. "
                f"Available columns: {list(map(str, df.columns))}"
            )

        lineage_df = df[[genome_col, lineage_col]].copy()
        lineage_df.columns = ["genome_id", "Lineage"]
        lineage_df["genome_id"] = lineage_df["genome_id"].astype("string").str.strip()
        lineage_df["Lineage"] = lineage_df["Lineage"].astype("string").str.strip()
        lineage_df = lineage_df[
            lineage_df["genome_id"].notna()
            & lineage_df["Lineage"].notna()
            & (lineage_df["genome_id"] != "")
            & (lineage_df["Lineage"] != "")
        ].copy()

        dup_count = int(lineage_df["genome_id"].duplicated().sum())
        if dup_count:
            self.logger.warning(
                f"Genome lineage table has {dup_count} duplicate genome IDs; keeping the first Lineage per genome."
            )
            lineage_df = lineage_df.drop_duplicates(subset=["genome_id"], keep="first")

        self.logger.info(f"Loaded genome lineage annotations for {len(lineage_df)} genomes from: {path}")
        return lineage_df

    def _attach_lineage_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Lineage to result tables when an annotation table is available."""
        if self.genome_lineage_df is None or df is None or len(df) == 0 or "genome_id" not in df.columns:
            return df

        out = df.copy()
        lineage_map = self.genome_lineage_df.set_index("genome_id")["Lineage"]
        lineage_values = out["genome_id"].astype(str).map(lineage_map)

        if "Lineage" in out.columns:
            existing = out["Lineage"]
            out["Lineage"] = existing.where(existing.notna() & (existing.astype(str) != ""), lineage_values)
            return out

        insert_at = out.columns.get_loc("genome_id") + 1
        out.insert(insert_at, "Lineage", lineage_values)
        return out

    def _build_theoretical_opportunity_serial(self, genome_digest_files: List[Path]) -> dict:
        """Compute genome-specific theoretical unique peptide opportunity serially."""
        peptide_owner: Dict[str, int] = {}
        genome_total_theoretical_peptides: Dict[str, int] = {}
        genome_theoretical_unique_peptides: Dict[str, int] = {}
        genome_ids_by_index: List[str] = []

        for genome_index, genome_file in enumerate(tqdm(genome_digest_files, desc="Indexing theoretical peptides")):
            genome_id = genome_file.stem
            genome_ids_by_index.append(genome_id)
            peptides = _read_unique_peptides_from_digest(genome_file)
            genome_total_theoretical_peptides[genome_id] = int(len(peptides))
            genome_theoretical_unique_peptides[genome_id] = 0
            for peptide in peptides:
                owner = peptide_owner.get(peptide)
                if owner is None:
                    peptide_owner[peptide] = genome_index
                    genome_theoretical_unique_peptides[genome_id] += 1
                elif owner != genome_index:
                    if owner >= 0:
                        previous_owner_id = genome_ids_by_index[owner]
                        genome_theoretical_unique_peptides[previous_owner_id] -= 1
                    peptide_owner[peptide] = -1

        genome_ids = sorted(genome_ids_by_index)
        digest_file_fingerprints = self._fingerprint_digest_files(genome_digest_files)

        return {
            "theoretical_peptide_universe_size": int(len(peptide_owner)),
            "genome_theoretical_unique_peptides": genome_theoretical_unique_peptides,
            "genome_total_theoretical_peptides": genome_total_theoretical_peptides,
            "target_genome_count": int(len(genome_digest_files)),
            "genome_ids": genome_ids,
            "digest_files": [str(Path(p)) for p in genome_digest_files],
            "digest_file_fingerprints": digest_file_fingerprints,
            "created_by": "MetaUmbra",
            "cache_version": THEORETICAL_OPPORTUNITY_CACHE_VERSION,
        }

    def _build_theoretical_opportunity_parallel(
        self,
        genome_digest_files: List[Path],
        num_workers_for_theoretical_opportunity: int,
        temp_parent_dir: Optional[str] = None,
    ) -> dict:
        """Compute theoretical opportunity using stable peptide-hash shards."""
        resolved_workers = _resolve_worker_count(num_workers_for_theoretical_opportunity, logger=self.logger)
        shard_count = min(
            THEORETICAL_OPPORTUNITY_MAX_SHARDS,
            max(1, resolved_workers * 2),
        )
        batches = np.array_split(
            np.asarray(genome_digest_files, dtype=object),
            max(1, resolved_workers * 4),
        )
        batch_file_lists = [list(batch) for batch in batches if len(batch) > 0]

        self.logger.info(
            "Building theoretical opportunity with %s worker(s) and %s peptide shard(s)...",
            resolved_workers,
            shard_count,
        )
        if temp_parent_dir:
            os.makedirs(temp_parent_dir, exist_ok=True)

        with tempfile.TemporaryDirectory(
            prefix="metaumbra_theoretical_opportunity_",
            dir=temp_parent_dir or None,
        ) as temp_dir:
            stage1_totals: Dict[str, int] = {}
            shard_paths_by_index: Dict[int, List[str]] = defaultdict(list)
            stage1_futures = []
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=resolved_workers)
            try:
                for batch_index, batch_file_paths in enumerate(batch_file_lists):
                    stage1_futures.append(
                        executor.submit(
                            _build_theoretical_opportunity_batch_worker,
                            int(batch_index),
                            [str(path) for path in batch_file_paths],
                            int(shard_count),
                            str(temp_dir),
                        )
                    )

                for fut in tqdm(
                    concurrent.futures.as_completed(stage1_futures),
                    total=len(stage1_futures),
                    desc="Sharding theoretical peptides",
                ):
                    batch_totals, batch_shard_paths = fut.result()
                    for genome_id, total_count in batch_totals.items():
                        stage1_totals[str(genome_id)] = int(total_count)
                    for shard_index, shard_path in batch_shard_paths.items():
                        shard_paths_by_index[int(shard_index)].append(str(shard_path))
            finally:
                try:
                    executor.shutdown(wait=True)
                except Exception:
                    pass
                try:
                    del stage1_futures
                    del executor
                except Exception:
                    pass

            shard_jobs = [
                (int(shard_index), list(shard_paths))
                for shard_index, shard_paths in sorted(shard_paths_by_index.items())
                if shard_paths
            ]
            genome_theoretical_unique_peptides: Counter = Counter()
            theoretical_peptide_universe_size = 0
            stage2_futures = []
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=resolved_workers)
            try:
                for shard_index, shard_paths in shard_jobs:
                    stage2_futures.append(
                        executor.submit(
                            _process_theoretical_opportunity_shard_worker,
                            int(shard_index),
                            [str(path) for path in shard_paths],
                        )
                    )

                for fut in tqdm(
                    concurrent.futures.as_completed(stage2_futures),
                    total=len(stage2_futures),
                    desc="Reducing theoretical peptide shards",
                ):
                    shard_universe_size, shard_unique_counts = fut.result()
                    theoretical_peptide_universe_size += int(shard_universe_size)
                    for genome_id, unique_count in shard_unique_counts.items():
                        genome_theoretical_unique_peptides[str(genome_id)] += int(unique_count)
            finally:
                try:
                    executor.shutdown(wait=True)
                except Exception:
                    pass
                try:
                    del stage2_futures
                    del executor
                except Exception:
                    pass

        genome_ids = sorted(path.stem for path in genome_digest_files)
        digest_file_fingerprints = self._fingerprint_digest_files(genome_digest_files)
        self.run_stats["theoretical_opportunity_parallelized"] = True
        self.run_stats["theoretical_opportunity_num_workers"] = int(resolved_workers)
        self.run_stats["theoretical_opportunity_shard_count"] = int(shard_count)
        self.run_stats["theoretical_opportunity_batch_count"] = int(len(batch_file_lists))

        return {
            "theoretical_peptide_universe_size": int(theoretical_peptide_universe_size),
            "genome_theoretical_unique_peptides": {
                genome_id: int(genome_theoretical_unique_peptides.get(genome_id, 0))
                for genome_id in genome_ids
            },
            "genome_total_theoretical_peptides": {
                genome_id: int(stage1_totals.get(genome_id, 0))
                for genome_id in genome_ids
            },
            "target_genome_count": int(len(genome_digest_files)),
            "genome_ids": genome_ids,
            "digest_files": [str(Path(p)) for p in genome_digest_files],
            "digest_file_fingerprints": digest_file_fingerprints,
            "created_by": "MetaUmbra",
            "cache_version": THEORETICAL_OPPORTUNITY_CACHE_VERSION,
        }

    def _build_theoretical_opportunity(
        self,
        genome_digest_files: List[Path],
        num_workers_for_theoretical_opportunity: Optional[int] = None,
        temp_parent_dir: Optional[str] = None,
    ) -> dict:
        """Compute genome-specific theoretical unique peptide opportunity."""
        requested_workers = (
            self.num_workers if num_workers_for_theoretical_opportunity is None else num_workers_for_theoretical_opportunity
        )
        resolved_workers = _resolve_worker_count(requested_workers, logger=self.logger)
        if resolved_workers <= 1 or len(genome_digest_files) <= 1:
            self.run_stats["theoretical_opportunity_parallelized"] = False
            self.run_stats["theoretical_opportunity_num_workers"] = int(resolved_workers)
            self.run_stats["theoretical_opportunity_shard_count"] = 1
            self.run_stats["theoretical_opportunity_batch_count"] = 1
            return self._build_theoretical_opportunity_serial(genome_digest_files)
        return self._build_theoretical_opportunity_parallel(
            genome_digest_files=genome_digest_files,
            num_workers_for_theoretical_opportunity=int(resolved_workers),
            temp_parent_dir=temp_parent_dir,
        )

    def _fingerprint_digest_files(self, genome_digest_files: List[Path]) -> Dict[str, dict]:
        """Return cheap cache validation metadata for digest TSV files."""
        fingerprints: Dict[str, dict] = {}
        for genome_file in genome_digest_files:
            path = Path(genome_file)
            stat = path.stat()
            fingerprints[path.stem] = {
                "path": str(path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        return fingerprints

    def _theoretical_cache_matches_digest_files(self, cached: dict, genome_digest_files: List[Path]) -> bool:
        """Check whether a theoretical opportunity cache matches selected digest files."""
        cached_version = int(cached.get("cache_version", 0) or 0)
        current_genome_ids = sorted(p.stem for p in genome_digest_files)
        cached_genome_ids = sorted(str(x) for x in cached.get("genome_ids", []))
        if cached_genome_ids != current_genome_ids:
            self.logger.warning(
                "Theoretical opportunity cache genome IDs do not match the selected genomes; rebuilding cache."
            )
            return False

        cached_fingerprints = dict(cached.get("digest_file_fingerprints", {}) or {})
        if not cached_fingerprints:
            if cached_version < THEORETICAL_OPPORTUNITY_CACHE_VERSION:
                self.logger.warning(
                    "Loaded legacy theoretical opportunity cache without digest file fingerprints; "
                    "validated genome IDs only. Rebuild the cache once to enable digest-change detection."
                )
                self.run_stats["theoretical_opportunity_cache_validation"] = "legacy_genome_ids_only"
                return True
            self.logger.info("Theoretical opportunity cache has no digest file fingerprints; rebuilding cache.")
            return False

        try:
            current_fingerprints = self._fingerprint_digest_files(genome_digest_files)
        except OSError as exc:
            self.logger.warning(f"Failed to stat digest files for theoretical opportunity cache validation: {exc}")
            return False

        for genome_id, current in current_fingerprints.items():
            cached_item = cached_fingerprints.get(genome_id)
            if not isinstance(cached_item, dict):
                self.logger.info(
                    "Theoretical opportunity cache is missing digest metadata for genome '%s'; rebuilding cache.",
                    genome_id,
                )
                return False
            if (
                str(cached_item.get("path", "")) != str(current["path"])
                or int(cached_item.get("size", -1) or -1) != int(current["size"])
                or int(cached_item.get("mtime_ns", -1) or -1) != int(current["mtime_ns"])
            ):
                self.logger.warning(
                    "Theoretical opportunity cache digest metadata changed for genome '%s'; rebuilding cache.",
                    genome_id,
                )
                return False

        self.run_stats["theoretical_opportunity_cache_validation"] = "digest_file_fingerprints"
        return True

    def _load_or_build_theoretical_opportunity(
        self,
        genome_digest_files: List[Path],
        cache_path: str,
        rebuild_cache: bool,
        num_workers_for_theoretical_opportunity: Optional[int] = None,
    ) -> Tuple[dict, bool]:
        """Load theoretical opportunity cache, or rebuild when missing/stale."""
        if cache_path and os.path.exists(cache_path) and not rebuild_cache:
            t_load0 = time.time()
            try:
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                if self._theoretical_cache_matches_digest_files(cached, genome_digest_files):
                    self.logger.info(f"Loaded theoretical opportunity cache: {cache_path}")
                    self.run_stats["theoretical_opportunity_cache_version"] = int(cached.get("cache_version", 0) or 0)
                    self.timing_stats["load_theoretical_opportunity_cache"] = float(time.time() - t_load0)
                    return cached, False
            except Exception as exc:
                self.logger.warning(f"Failed to load theoretical opportunity cache; rebuilding. Error: {exc}")
            self.timing_stats["load_theoretical_opportunity_cache"] = float(time.time() - t_load0)

        self.logger.info("Building theoretical unique peptide opportunity cache...")
        t_build0 = time.time()
        opportunity = self._build_theoretical_opportunity(
            genome_digest_files=genome_digest_files,
            num_workers_for_theoretical_opportunity=num_workers_for_theoretical_opportunity,
            temp_parent_dir=(os.path.dirname(cache_path) or ".") if cache_path else None,
        )
        self.timing_stats["build_theoretical_opportunity_cache"] = float(time.time() - t_build0)
        self.run_stats["theoretical_opportunity_cache_validation"] = "rebuilt"
        self.run_stats["theoretical_opportunity_cache_version"] = int(THEORETICAL_OPPORTUNITY_CACHE_VERSION)
        if cache_path:
            t_save0 = time.time()
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(opportunity, f)
            self.logger.info(f"Saved theoretical opportunity cache: {cache_path}")
            self.timing_stats["save_theoretical_opportunity_cache"] = float(time.time() - t_save0)
        return opportunity, True

    def _apply_theoretical_opportunity(self, opportunity: dict) -> None:
        self.theoretical_peptide_universe_size = int(opportunity.get("theoretical_peptide_universe_size", 0) or 0)
        self.genome_theoretical_unique_peptides = {
            str(k): int(v)
            for k, v in dict(opportunity.get("genome_theoretical_unique_peptides", {}) or {}).items()
        }
        self.total_theoretical_unique_peptides_all_genomes = int(sum(self.genome_theoretical_unique_peptides.values()))
        cached_totals = {
            str(k): int(v)
            for k, v in dict(opportunity.get("genome_total_theoretical_peptides", {}) or {}).items()
        }
        if cached_totals:
            self.genome_total_theoretical_peptides.update(cached_totals)

    def _hypergeom_tail_pvalue(
        self,
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

    def _unique_pvalue_stats_for_genome(self, gid: str, u_observed: int) -> dict:
        U = int(u_observed)
        mode = _normalize_unique_pvalue_mode(self.unique_pvalue_mode)
        alpha = float(min(max(self.single_peptide_error_rate_upper_bound, 1e-12), 1.0))
        p_unique = 1.0
        p_unique_depth = 1.0
        S = int(self.observed_unique_peptide_pool_size)
        expected = 0.0
        fold = 0.0
        gate_pass = False
        null_model = ""
        theoretical_unique: Optional[int] = None
        unique_effective_count = float(U)
        count_model = "raw"
        error_source_used = ""

        if mode == "hypergeometric-opportunity":
            A = int(self.genome_theoretical_unique_peptides.get(gid, 0))
            A_total = int(max(self.total_theoretical_unique_peptides_all_genomes, 1))
            theoretical_unique = int(A)
            expected = float(S) * float(A) / float(A_total)
            fold = float(U) / max(expected, 1e-12)
            null_model = "hypergeometric"
            if U >= int(self.min_unique_for_unique_pvalue) and A > 0 and S > 0 and A_total > 0:
                p_unique_depth = self._hypergeom_tail_pvalue(U, A_total, A, S)
                p_unique = p_unique_depth
                gate_pass = True
        elif mode == "alpha-upper-bound":
            matched = self.genome_matched_peptides.get(gid, set())
            uniq = sorted(p for p in matched if int((self.peptide_degeneracy or {}).get(p, 1)) == 1)
            p_unique, unique_effective_count, error_source_used = _tempered_unique_error_product_pvalue(
                unique_peptides=uniq,
                alpha=alpha,
                peptide_error_upper_by_peptide=self.peptide_error_upper_by_peptide,
                error_source=self.unique_peptide_error_source,
                unique_count_power=self.unique_count_power,
                unique_count_cap=self.unique_count_cap,
            )
            count_model = f"power:{float(self.unique_count_power):g}"
            p_unique_depth = p_unique
            gate_pass = U > 0
            null_model = error_source_used
        else:
            raise ValueError(f"Unsupported unique_pvalue_mode: {mode!r}.")

        return {
            "p_unique": _clip_pvalue(p_unique),
            "p_unique_depth": _clip_pvalue(p_unique_depth),
            "unique_observed": int(U),
            "unique_expected_null": float(expected),
            "unique_depth_fold": float(fold),
            "unique_depth_null_model": null_model,
            "unique_pvalue_mode": mode,
            "unique_peptide_error_source": error_source_used,
            "unique_gate_pass": bool(gate_pass),
            "theoretical_unique_peptides": theoretical_unique,
            "unique_effective_count": float(unique_effective_count),
            "unique_pvalue_count_model": count_model,
        }


    # =========================
    # I/O: Peptide table
    # =========================
    def read_peptide_file(
        self,
        peptide_table_path: Optional[str] = None,
        peptide_table_df: Optional[pd.DataFrame] = None,
        peptide_seq_col: str = "Base Sequence",
        peptide_score_col: Optional[str] = "Evidence",
        peptide_decoy_flag_col: Optional[str] = "Target/Decoy",
        decoy_flag_value: str = "decoy",
        peptide_table_sep: str = "\t",
        peptide_error_col: Optional[str] = None, # ["PEP", "FDR", "AUTO", None, "Q.Value", ...]
        peptide_error_cutoff: float = 0.05,
        single_peptide_error_rate_upper_bound: float = 0.3,
    ) -> bool:
        """Read a peptide table and build peptide->score dictionary."""
        if peptide_table_path is None and peptide_table_df is None:
            raise ValueError("Either peptide_table_path or peptide_table_df must be provided.")
        if peptide_table_path is not None and peptide_table_df is not None:
            raise ValueError("Provide only one of peptide_table_path or peptide_table_df, not both.")

        if peptide_table_df is not None:
            df = peptide_table_df
            self.peptide_table_dir = os.getcwd()
            available_columns = df.columns.tolist()
            
            # Ensure columns have the expected types so that boolean Parquet flags match string rules
            if peptide_seq_col in df.columns:
                df[peptide_seq_col] = df[peptide_seq_col].astype(str)
            if peptide_decoy_flag_col and peptide_decoy_flag_col in df.columns:
                df[peptide_decoy_flag_col] = df[peptide_decoy_flag_col].astype(str)
            if peptide_score_col and peptide_score_col in df.columns:
                df[peptide_score_col] = pd.to_numeric(df[peptide_score_col], errors="coerce")
            if peptide_error_col and peptide_error_col in df.columns:
                df[peptide_error_col] = pd.to_numeric(df[peptide_error_col], errors="coerce")
        else:
            if peptide_table_path is None:
                raise ValueError("peptide_table_path is None (unexpected).")
            peptide_file_path = str(peptide_table_path)
            if not os.path.exists(peptide_file_path):
                raise FileNotFoundError(f"Peptide file does not exist: {peptide_file_path}")
            self.peptide_table_dir = os.path.dirname(peptide_file_path)
            self.logger.info(f"Reading peptide file: {peptide_file_path}")

            sample_df = pd.read_csv(peptide_file_path, sep=peptide_table_sep, nrows=5)
            available_columns = sample_df.columns.tolist()
            if peptide_seq_col not in available_columns:
                raise ValueError(f"Missing peptide column '{peptide_seq_col}' in peptide file.")

            if peptide_error_col is None:
                for candidate in ("PEP", "FDR", "Q.Value"):
                    if candidate in available_columns:
                        peptide_error_col = candidate
                        self.logger.info(f"Auto-detected peptide error column: '{peptide_error_col}'.")
                        break

            cols = [peptide_seq_col]
            dtype = {peptide_seq_col: "string"}

            if peptide_score_col and peptide_score_col in available_columns:
                cols.append(peptide_score_col)
                dtype[peptide_score_col] = "float32"
                self.logger.info(f"Using peptide score column: '{peptide_score_col}'.")
            else:
                self.logger.warning(f"Score column '{peptide_score_col}' not found; setting all scores=1.")
                peptide_score_col = None

            if peptide_decoy_flag_col and peptide_decoy_flag_col in available_columns:
                cols.append(peptide_decoy_flag_col)
                dtype[peptide_decoy_flag_col] = "string"
            else:
                peptide_decoy_flag_col = None

            if peptide_error_col and peptide_error_col in available_columns:
                cols.append(peptide_error_col)
                dtype[peptide_error_col] = "float32"
            else:
                if peptide_error_col is not None:
                    self.logger.warning(
                        f"Error column '{peptide_error_col}' not found; skipping peptide-level error filtering."
                    )
                peptide_error_col = None

            df = pd.read_csv(peptide_file_path, sep=peptide_table_sep, usecols=cols, dtype=dtype, engine="c")
            self.logger.info(f"Loaded {len(df)} rows from peptide file.")

        if peptide_seq_col not in available_columns:
            raise ValueError(f"Missing peptide column '{peptide_seq_col}'.")

        if peptide_error_col is None:
            for candidate in ("PEP", "FDR", "Q.Value"):
                if candidate in df.columns:
                    peptide_error_col = candidate
                    self.logger.info(f"Auto-detected peptide error column: '{peptide_error_col}'.")
                    break

        self.peptide_error_cutoff = float(peptide_error_cutoff)
        error_filter_applied = False

        # --- NEW: run-level input stats (for paper) ---
        self.run_stats["peptide_rows_loaded"] = int(len(df))
        self.run_stats["peptide_seq_col"] = peptide_seq_col
        self.run_stats["peptide_score_col"] = peptide_score_col if peptide_score_col else None
        self.run_stats["peptide_error_col"] = peptide_error_col if peptide_error_col else None
        self.run_stats["peptide_error_cutoff"] = float(peptide_error_cutoff)
        self.run_stats["single_peptide_error_rate_upper_bound_configured"] = float(
            single_peptide_error_rate_upper_bound
        )
        self.run_stats["peptide_decoy_flag_col"] = peptide_decoy_flag_col if peptide_decoy_flag_col else None
        self.run_stats["decoy_flag_value"] = decoy_flag_value

        if peptide_decoy_flag_col and peptide_decoy_flag_col in df.columns:
            before = len(df)
            df = df[(df[peptide_decoy_flag_col] != decoy_flag_value) | (df[peptide_decoy_flag_col].isna())]
            self.logger.info(f"Peptide-level decoy filter: {before} -> {len(df)} rows.")
            self.run_stats["peptide_rows_after_decoy_filter"] = int(len(df))

        if peptide_error_col and peptide_error_col in df.columns:
            before = len(df)
            df[peptide_error_col] = pd.to_numeric(df[peptide_error_col], errors="coerce")
            df = df[df[peptide_error_col] <= float(peptide_error_cutoff)]
            self.logger.info(
                f"Peptide-level error filter on '{peptide_error_col}' (<= {peptide_error_cutoff}): {before} -> {len(df)} rows."
            )
            self.run_stats["peptide_rows_after_error_filter"] = int(len(df))
            error_filter_applied = True

        # Interpret unique-evidence bound as an upper bound on single-peptide false match probability.
        # This is intentionally separate from peptide_error_cutoff, which only filters input peptide rows.
        self.single_peptide_error_rate_upper_bound = float(
            min(max(single_peptide_error_rate_upper_bound, 1e-12), 1.0)
        )
        self.run_stats["single_peptide_error_rate_upper_bound_source"] = "single_peptide_error_rate_upper_bound"
        if not error_filter_applied:
            self.logger.warning(
                "No peptide-level error filter was applied; assuming single_peptide_error_rate_upper_bound="
                f"{self.single_peptide_error_rate_upper_bound:.4g} from single_peptide_error_rate_upper_bound."
            )
        self.run_stats["single_peptide_error_rate_upper_bound"] = float(self.single_peptide_error_rate_upper_bound)

        self.peptide_error_upper_by_peptide = {}
        if peptide_error_col and peptide_error_col in df.columns:
            # Use MAX among rows for the same peptide to stay conservative as an "upper bound".
            pep_err = (
                df.groupby(peptide_seq_col)[peptide_error_col]
                .max()
                .reset_index()
            )
            pep_err.columns = ["Peptide", "Error"]
            pep_err["Error"] = pd.to_numeric(pep_err["Error"], errors="coerce").astype(float)
            pep_err = pep_err.dropna(subset=["Error"])
            pep_err["Error"] = pep_err["Error"].clip(lower=1e-12, upper=1.0)

            self.peptide_error_upper_by_peptide = dict(
                zip(pep_err["Peptide"].astype(str), pep_err["Error"].astype(float))
            )

        self.run_stats["unique_upper_uses_per_peptide_error"] = bool(self.peptide_error_upper_by_peptide)
        self.run_stats["per_peptide_error_mapping_size"] = int(len(self.peptide_error_upper_by_peptide))

        if peptide_score_col and peptide_score_col in df.columns:
            pep_scores = df.groupby(peptide_seq_col)[peptide_score_col].max().reset_index()
            pep_scores.columns = ["Peptide", "Score"]
            min_s = float(pep_scores["Score"].min())
            max_s = float(pep_scores["Score"].max())
            if max_s > min_s:
                pep_scores["NormScore"] = (pep_scores["Score"] - min_s) / (max_s - min_s)
            else:
                pep_scores["NormScore"] = 1.0
            self.peptide_score = dict(zip(pep_scores["Peptide"].astype(str), pep_scores["NormScore"].astype(float)))
        else:
            peps = df[peptide_seq_col].astype(str).unique()
            self.peptide_score = {p: 1.0 for p in peps}

        # --- NEW: peptide score quantiles (NormScore in [0,1]) ---
        try:
            vals = np.asarray(list(self.peptide_score.values()), dtype=float)
            if vals.size > 0:
                qs = np.quantile(vals, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist()
                self.run_stats["peptide_normscore_quantiles"] = {
                    "0.05": float(qs[0]), "0.25": float(qs[1]), "0.50": float(qs[2]),
                    "0.75": float(qs[3]), "0.95": float(qs[4])
                }
        except Exception:
            pass

        self.run_stats["observed_unique_peptides"] = int(len(self.peptide_score))
        self.run_stats.setdefault("peptide_rows_after_decoy_filter", int(len(df)))
        self.run_stats.setdefault("peptide_rows_after_error_filter", int(len(df)))

        self.logger.info(f"Observed peptides: {len(self.peptide_score)} (unique)")
        return True

    def read_unit_aware_peptide_file(
        self,
        peptide_table_path: str,
        sample_id_col: str,
        peptide_seq_col: str,
        peptide_score_col: Optional[str] = "Evidence",
        peptide_decoy_flag_col: Optional[str] = "Reverse",
        decoy_flag_value: str = "+",
        intensity_col: str = "Precursor.Quantity",
        peptide_error_col: Optional[str] = "Q.Value",
        peptide_error_cutoff: float = 0.05,
        single_peptide_error_rate_upper_bound: float = 0.3,
        intensity_min_value: float = 0.0,
        intensity_min_quantile: float = 0.0,
        metadata_table_path: Optional[str] = None,
        metadata_sample_id_col: str = "sample_id",
        metadata_analysis_unit_col: str = "analysis_unit_id",
        peptide_table_sep: str = "\t",
    ) -> bool:
        """Read long-format peptide evidence and build peptide x analysis-unit presence."""
        from scipy.sparse import csr_matrix

        peptide_file_path = str(peptide_table_path)
        if not os.path.exists(peptide_file_path):
            raise FileNotFoundError(f"Peptide file does not exist: {peptide_file_path}")

        sample_col = str(sample_id_col).strip()
        seq_col = str(peptide_seq_col).strip()
        score_col = str(peptide_score_col).strip() if peptide_score_col else None
        decoy_col = str(peptide_decoy_flag_col).strip() if peptide_decoy_flag_col else None
        intensity_col = str(intensity_col).strip()
        error_col = str(peptide_error_col).strip() if peptide_error_col else None
        if not sample_col:
            raise ValueError("sample_id_col must not be empty when unit-aware scoring is enabled.")
        if not seq_col:
            raise ValueError("peptide_seq_col must not be empty when unit-aware scoring is enabled.")
        if not intensity_col:
            raise ValueError("intensity_col must not be empty when unit-aware scoring is enabled.")

        suffix = Path(peptide_file_path).suffix.lower()
        is_parquet_input = suffix in {".parquet", ".pq"}
        self.peptide_table_dir = os.path.dirname(peptide_file_path)
        self.logger.info(f"Reading unit-aware peptide table: {peptide_file_path}")

        def _norm_col_name(value: str) -> str:
            return "".join(ch.lower() for ch in str(value) if ch.isalnum())

        def _resolve_col(
            available: List[str],
            preferred: Optional[str],
            candidates: List[str],
            required_label: str,
            required: bool = True,
        ) -> Optional[str]:
            if preferred and preferred in available:
                return preferred
            lookup: Dict[str, str] = {}
            for col in available:
                key = _norm_col_name(col)
                if key and key not in lookup:
                    lookup[key] = col
            names_to_try = []
            if preferred:
                names_to_try.append(preferred)
            names_to_try.extend(candidates)
            for candidate in names_to_try:
                match = lookup.get(_norm_col_name(candidate))
                if match:
                    if preferred and match != preferred:
                        self.logger.info(
                            f"Auto-detected unit-aware {required_label} column: '{match}' "
                            f"(configured/default was '{preferred}')."
                        )
                    return match
            if required:
                raise ValueError(
                    f"Unable to locate required unit-aware {required_label} column. "
                    f"Configured/default value: {preferred!r}. Available columns: {available}"
                )
            return None

        def _infer_decoy_flag_value_from_values(values, configured_value: str) -> str:
            configured = str(configured_value)
            if configured == "":
                return configured
            value_set: Set[str] = set()
            for value in values:
                if value is None:
                    continue
                text = str(value).strip()
                if text and text != "<NA>":
                    value_set.add(text)
            if configured in value_set or configured != "+":
                return configured
            for candidate in ("True", "true", "1", "decoy", "Decoy", "DECOY", "T", "t"):
                if candidate in value_set:
                    self.logger.info(
                        f"Auto-detected unit-aware decoy marker for column '{decoy_col}': "
                        f"using '{candidate}' instead of '+'."
                    )
                    return candidate
            return configured

        if is_parquet_input:
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise RuntimeError(
                    "pyarrow is required to read parquet files. Install it with: python -m pip install pyarrow"
                ) from exc
            schema_names = list(pq.read_schema(peptide_file_path).names)
            sample_col = str(
                _resolve_col(schema_names, sample_col, ["Run", "File.Name", "Raw.File", "Sample", "Sample.Name"], "sample ID")
            )
            seq_col = str(
                _resolve_col(
                    schema_names,
                    seq_col,
                    ["Stripped.Sequence", "Base Sequence", "Sequence", "Peptide.Sequence", "PeptideSequence"],
                    "peptide sequence",
                )
            )
            score_col = _resolve_col(
                schema_names,
                score_col,
                ["Evidence", "Score", "CScore"],
                "peptide score",
                required=False,
            )
            decoy_col = _resolve_col(
                schema_names,
                decoy_col,
                ["Reverse", "Target/Decoy", "TargetDecoy", "Decoy"],
                "decoy flag",
                required=False,
            )
            intensity_col = str(
                _resolve_col(
                    schema_names,
                    intensity_col,
                    ["Precursor.Quantity", "Precursor.Normalised", "Intensity"],
                    "intensity",
                )
            )
            error_col = _resolve_col(
                schema_names,
                error_col,
                ["Q.Value", "QValue", "Qval", "QVal", "PEP", "FDR"],
                "peptide error",
                required=False,
            )
            required_cols = [sample_col, seq_col, intensity_col]
            optional_cols = [score_col, decoy_col, error_col]
            columns_to_read = list(dict.fromkeys(required_cols + [col for col in optional_cols if col in schema_names]))
            df = pq.read_table(peptide_file_path, columns=columns_to_read, use_threads=True).to_pandas()
        else:
            sep = "," if suffix == ".csv" else peptide_table_sep
            sample_df = pd.read_csv(peptide_file_path, sep=sep, nrows=5)
            available_columns = sample_df.columns.tolist()
            sample_col = str(
                _resolve_col(available_columns, sample_col, ["Run", "File.Name", "Raw.File", "Sample", "Sample.Name"], "sample ID")
            )
            seq_col = str(
                _resolve_col(
                    available_columns,
                    seq_col,
                    ["Stripped.Sequence", "Base Sequence", "Sequence", "Peptide.Sequence", "PeptideSequence"],
                    "peptide sequence",
                )
            )
            score_col = _resolve_col(
                available_columns,
                score_col,
                ["Evidence", "Score", "CScore"],
                "peptide score",
                required=False,
            )
            decoy_col = _resolve_col(
                available_columns,
                decoy_col,
                ["Reverse", "Target/Decoy", "TargetDecoy", "Decoy"],
                "decoy flag",
                required=False,
            )
            intensity_col = str(
                _resolve_col(
                    available_columns,
                    intensity_col,
                    ["Precursor.Quantity", "Precursor.Normalised", "Intensity"],
                    "intensity",
                )
            )
            error_col = _resolve_col(
                available_columns,
                error_col,
                ["Q.Value", "QValue", "Qval", "QVal", "PEP", "FDR"],
                "peptide error",
                required=False,
            )
            required_cols = [sample_col, seq_col, intensity_col]
            columns_to_read = list(dict.fromkeys(required_cols + [col for col in [score_col, decoy_col, error_col] if col]))
            dtype = {sample_col: "string", seq_col: "string"}
            df = pd.read_csv(peptide_file_path, sep=sep, usecols=columns_to_read, dtype=dtype, engine="c")

        if score_col and score_col not in df.columns:
            self.logger.warning(
                f"Score column '{score_col}' not found; setting all unit-aware peptide scores=1."
            )
            score_col = None
        if decoy_col and decoy_col not in df.columns:
            decoy_col = None
        if error_col and error_col not in df.columns:
            self.logger.warning(
                f"Error column '{error_col}' not found; skipping peptide-level error filtering for unit-aware input."
            )
            error_col = None
        if decoy_col:
            decoy_flag_value = _infer_decoy_flag_value_from_values(
                df[decoy_col].dropna().unique()[:50],
                decoy_flag_value,
            )

        self.run_stats["unit_aware"] = True
        self.run_stats["unit_aware_peptide_rows_loaded"] = int(len(df))
        self.run_stats["unit_aware_sample_id_col"] = sample_col
        self.run_stats["unit_aware_peptide_seq_col"] = seq_col
        self.run_stats["unit_aware_peptide_score_col"] = score_col if score_col else None
        self.run_stats["unit_aware_peptide_decoy_flag_col"] = decoy_col if decoy_col else None
        self.run_stats["unit_aware_decoy_flag_value"] = decoy_flag_value
        self.run_stats["unit_aware_intensity_col"] = intensity_col
        self.run_stats["unit_aware_peptide_error_col"] = error_col
        self.run_stats["unit_aware_peptide_error_cutoff"] = float(peptide_error_cutoff)
        self.run_stats["unit_aware_intensity_min_value"] = float(intensity_min_value)
        self.run_stats["unit_aware_intensity_min_quantile"] = float(intensity_min_quantile)

        self.logger.info(f"Preparing unit-aware columns for {len(df)} loaded row(s) ...")
        df = df.copy()
        df[sample_col] = df[sample_col].astype("string").str.strip()
        if is_parquet_input:
            sample_values_before = df[sample_col].copy()
            df[sample_col] = _strip_raw_suffix_from_sample_ids(df[sample_col])
            changed_rows = int((sample_values_before.notna() & (sample_values_before != df[sample_col])).sum())
            if changed_rows:
                self.logger.info(
                    f"Normalized parquet sample IDs by removing trailing '.raw' from {changed_rows} row(s)."
                )
        df[seq_col] = df[seq_col].astype("string").str.strip()
        if score_col:
            df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
        if decoy_col:
            df[decoy_col] = df[decoy_col].astype("string").str.strip()
        df[intensity_col] = pd.to_numeric(df[intensity_col], errors="coerce")
        if error_col:
            df[error_col] = pd.to_numeric(df[error_col], errors="coerce")

        if decoy_col:
            before = int(len(df))
            df = df[(df[decoy_col] != str(decoy_flag_value)) | (df[decoy_col].isna())].copy()
            self.logger.info(f"Unit-aware decoy filter on '{decoy_col}': {before} -> {len(df)} rows.")
            self.run_stats["unit_aware_peptide_rows_after_decoy_filter"] = int(len(df))

        self.logger.info("Counting total unit-aware rows per sample ...")
        total_by_sample = (
            df[df[sample_col].notna() & (df[sample_col] != "")]
            .groupby(sample_col, dropna=False)
            .size()
            .astype(int)
        )

        before = int(len(df))
        valid = df[
            df[sample_col].notna()
            & (df[sample_col] != "")
            & df[seq_col].notna()
            & (df[seq_col] != "")
            & df[intensity_col].notna()
            & (df[intensity_col] > float(intensity_min_value))
        ].copy()
        self.logger.info(
            f"Unit-aware base validity/intensity filter on '{intensity_col}' (> {intensity_min_value}): "
            f"{before} -> {len(valid)} rows."
        )
        if error_col:
            before = int(len(valid))
            valid = valid[valid[error_col].notna() & (valid[error_col] <= float(peptide_error_cutoff))].copy()
            self.logger.info(
                f"Unit-aware error filter on '{error_col}' (<= {peptide_error_cutoff}): {before} -> {len(valid)} rows."
            )

        quantile = float(intensity_min_quantile)
        if quantile < 0.0 or quantile > 1.0:
            raise ValueError("intensity_min_quantile must be between 0 and 1.")
        if quantile > 0.0 and len(valid) > 0:
            thresholds = valid.groupby(sample_col)[intensity_col].transform(lambda x: x.quantile(quantile))
            before = int(len(valid))
            valid = valid[valid[intensity_col] >= thresholds].copy()
            self.logger.info(
                f"Unit-aware within-sample intensity quantile filter (>= q{quantile:g}): {before} -> {len(valid)} rows."
            )

        if valid.empty:
            raise ValueError("No valid peptide observations remain after unit-aware intensity/error filtering.")

        self.logger.info("Resolving unit-aware sample IDs and optional metadata mapping ...")
        sample_ids = [str(x) for x in pd.unique(df.loc[df[sample_col].notna() & (df[sample_col] != ""), sample_col])]
        sample_to_unit = {sample_id: sample_id for sample_id in sample_ids}
        excluded_samples: Set[str] = set()
        metadata_fields: Optional[pd.DataFrame] = None
        metadata_path = str(metadata_table_path or "").strip()
        if metadata_path:
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Metadata table not found: {metadata_path}")
            meta_sep = "," if Path(metadata_path).suffix.lower() == ".csv" else "\t"
            metadata_df = pd.read_csv(metadata_path, sep=meta_sep, dtype="string")
            missing_meta_cols = [
                col for col in [metadata_sample_id_col, metadata_analysis_unit_col] if col not in metadata_df.columns
            ]
            if missing_meta_cols:
                raise ValueError(
                    f"Metadata table is missing required columns: {missing_meta_cols}. "
                    f"Available columns: {metadata_df.columns.tolist()}"
                )
            metadata_df = metadata_df.copy()
            metadata_df[metadata_sample_id_col] = metadata_df[metadata_sample_id_col].astype("string").str.strip()
            metadata_df[metadata_analysis_unit_col] = metadata_df[metadata_analysis_unit_col].astype("string").str.strip()
            metadata_df = metadata_df[
                metadata_df[metadata_sample_id_col].notna()
                & (metadata_df[metadata_sample_id_col] != "")
                & metadata_df[metadata_analysis_unit_col].notna()
                & (metadata_df[metadata_analysis_unit_col] != "")
            ].copy()
            duplicate_count = int(metadata_df[metadata_sample_id_col].duplicated().sum())
            if duplicate_count:
                self.logger.warning(
                    f"Metadata table has {duplicate_count} duplicate sample IDs; keeping the first mapping."
                )
            metadata_df = metadata_df.drop_duplicates(subset=[metadata_sample_id_col], keep="first")
            sample_to_unit.update(
                dict(zip(metadata_df[metadata_sample_id_col].astype(str), metadata_df[metadata_analysis_unit_col].astype(str)))
            )
            if "included" in metadata_df.columns:
                included_values = metadata_df["included"].astype("string").str.strip().str.lower()
                excluded_samples = set(
                    metadata_df.loc[
                        included_values.isin({"0", "false", "no", "n", "off"}),
                        metadata_sample_id_col,
                    ].astype(str)
                )
            missing_samples = sorted(set(sample_ids).difference(set(metadata_df[metadata_sample_id_col].astype(str))))
            if missing_samples:
                preview = ", ".join(missing_samples[:10])
                suffix_msg = " ..." if len(missing_samples) > 10 else ""
                self.logger.warning(
                    f"{len(missing_samples)} sample(s) are missing from metadata; assigning them to their own "
                    f"analysis_unit_id: {preview}{suffix_msg}"
                )
            metadata_fields = metadata_df.rename(
                columns={
                    metadata_sample_id_col: "sample_id",
                    metadata_analysis_unit_col: "analysis_unit_id",
                }
            )

        self.logger.info("Building unit-aware unique peptide-sample pairs ...")
        t_pairs0 = time.time()
        pair_source = valid[[seq_col, sample_col]]
        if is_parquet_input:
            valid_pairs = _drop_duplicate_pairs_with_pyarrow(pair_source, seq_col, sample_col, logger=self.logger)
        else:
            valid_pairs = pair_source.drop_duplicates().copy()
        valid_pairs[seq_col] = valid_pairs[seq_col].astype(str)
        valid_pairs[sample_col] = valid_pairs[sample_col].astype(str)
        valid_pair_sample_ids = set(valid_pairs[sample_col].tolist())
        valid_sample_ids = [
            sample_id
            for sample_id in sample_ids
            if sample_id in valid_pair_sample_ids and sample_id not in excluded_samples
        ]
        if not valid_sample_ids:
            raise ValueError("No sample has valid peptide observations after filtering.")
        valid_pairs_for_matrix = valid_pairs[valid_pairs[sample_col].isin(valid_sample_ids)].copy()
        pair_rows, peptide_uniques = pd.factorize(valid_pairs_for_matrix[seq_col], sort=False)
        peptide_list = [str(x) for x in peptide_uniques.tolist()]
        pair_cols = pd.Categorical(valid_pairs_for_matrix[sample_col], categories=valid_sample_ids).codes
        if (pair_cols < 0).any():
            raise RuntimeError("Failed to encode unit-aware sample IDs for sparse matrix construction.")
        analysis_unit_ids = sorted({str(sample_to_unit.get(sample_id, sample_id)) for sample_id in valid_sample_ids})
        self.logger.info(
            f"Unit-aware unique peptide-sample pairs: {len(valid_pairs_for_matrix)} pair(s), "
            f"{len(peptide_list)} peptide(s), {len(valid_sample_ids)} included sample(s), "
            f"{len(analysis_unit_ids)} analysis unit(s); built in {time.time() - t_pairs0:.2f}s."
        )
        self.timing_stats["unit_aware_unique_pairs"] = float(time.time() - t_pairs0)

        peptide_index = {peptide: i for i, peptide in enumerate(peptide_list)}
        sample_index = {sample_id: i for i, sample_id in enumerate(valid_sample_ids)}
        unit_index = {unit_id: i for i, unit_id in enumerate(analysis_unit_ids)}

        self.logger.info(
            f"Building sparse peptide x sample presence matrix "
            f"({len(peptide_list)} x {len(valid_sample_ids)}) ..."
        )
        pair_rows = pair_rows.astype(np.int64, copy=False)
        pair_cols = pair_cols.astype(np.int64, copy=False)
        data = np.ones(len(valid_pairs_for_matrix), dtype=COUNT_DTYPE)
        x_sample = csr_matrix(
            (data, (pair_rows, pair_cols)),
            shape=(len(peptide_list), len(valid_sample_ids)),
            dtype=COUNT_DTYPE,
        )

        self.logger.info(
            f"Building sparse sample x analysis-unit membership matrix "
            f"({len(valid_sample_ids)} x {len(analysis_unit_ids)}) ..."
        )
        b_rows = np.arange(len(valid_sample_ids), dtype=np.int64)
        b_cols = np.fromiter(
            (unit_index[str(sample_to_unit.get(sample_id, sample_id))] for sample_id in valid_sample_ids),
            dtype=np.int64,
            count=len(valid_sample_ids),
        )
        b_data = np.ones(len(b_rows), dtype=COUNT_DTYPE)
        b_matrix = csr_matrix(
            (b_data, (b_rows, b_cols)),
            shape=(len(valid_sample_ids), len(analysis_unit_ids)),
            dtype=COUNT_DTYPE,
        )
        self.logger.info("Combining sample-level peptide presence into unit-level presence ...")
        x_unit_counts = x_sample @ b_matrix
        x_unit = (x_unit_counts > 0).astype(COUNT_DTYPE).tocsr()

        self.logger.info("Preparing unit-aware sample-unit mapping output table ...")
        valid_counts = valid_pairs.groupby(sample_col).size().astype(int).to_dict()
        mapping_rows = []
        for sample_id in sample_ids:
            mapping_rows.append(
                {
                    "sample_id": sample_id,
                    "analysis_unit_id": str(sample_to_unit.get(sample_id, sample_id)),
                    "included": bool(sample_id in sample_index),
                    "n_valid_peptides": int(valid_counts.get(sample_id, 0)),
                    "n_total_rows": int(total_by_sample.get(sample_id, 0)),
                }
            )
        mapping_df = pd.DataFrame(mapping_rows)
        if metadata_fields is not None:
            metadata_extra = metadata_fields.drop_duplicates(subset=["sample_id"], keep="first")
            auto_cols = {
                "included",
                "n_valid_peptides",
                "n_total_rows",
                "included_x",
                "included_y",
                "n_valid_peptides_x",
                "n_valid_peptides_y",
                "n_total_rows_x",
                "n_total_rows_y",
            }
            extra_cols = [
                col
                for col in metadata_extra.columns
                if col not in {"sample_id", "analysis_unit_id"} and col not in auto_cols
            ]
            if extra_cols:
                mapping_df = mapping_df.merge(
                    metadata_extra[["sample_id", *extra_cols]],
                    on="sample_id",
                    how="left",
                )

        self.unit_aware_enabled = True
        self.unit_presence_rule = "union"
        self.unit_shared_mode = "per-unit"
        self.unit_sample_ids = valid_sample_ids
        self.unit_analysis_unit_ids = analysis_unit_ids
        self.unit_peptides = peptide_list
        self.unit_peptide_index = peptide_index
        self.unit_presence_matrix = x_unit
        self.unit_sample_counts = {
            unit_id: int(sum(1 for sample_id in valid_sample_ids if str(sample_to_unit.get(sample_id, sample_id)) == unit_id))
            for unit_id in analysis_unit_ids
        }
        self.sample_unit_mapping_df = mapping_df
        if score_col and score_col in valid.columns:
            self.logger.info("Computing unit-aware peptide scores from the configured score column ...")
            score_source = valid[
                valid[sample_col].isin(valid_sample_ids)
                & valid[seq_col].astype(str).isin(set(peptide_list))
                & valid[score_col].notna()
            ].copy()
            pep_scores = score_source.groupby(seq_col)[score_col].max().reset_index()
            pep_scores.columns = ["Peptide", "Score"]
            if not pep_scores.empty:
                min_s = float(pep_scores["Score"].min())
                max_s = float(pep_scores["Score"].max())
                if max_s > min_s:
                    pep_scores["NormScore"] = (pep_scores["Score"] - min_s) / (max_s - min_s)
                else:
                    pep_scores["NormScore"] = 1.0
                self.peptide_score = {peptide: 1.0 for peptide in peptide_list}
                self.peptide_score.update(
                    dict(zip(pep_scores["Peptide"].astype(str), pep_scores["NormScore"].astype(float)))
                )
            else:
                self.peptide_score = {peptide: 1.0 for peptide in peptide_list}
        else:
            self.peptide_score = {peptide: 1.0 for peptide in peptide_list}

        try:
            vals = np.asarray(list(self.peptide_score.values()), dtype=float)
            if vals.size > 0:
                qs = np.quantile(vals, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist()
                self.run_stats["peptide_normscore_quantiles"] = {
                    "0.05": float(qs[0]), "0.25": float(qs[1]), "0.50": float(qs[2]),
                    "0.75": float(qs[3]), "0.95": float(qs[4])
                }
        except Exception:
            pass

        self.peptide_error_cutoff = float(peptide_error_cutoff)
        self.single_peptide_error_rate_upper_bound = float(
            min(max(float(single_peptide_error_rate_upper_bound), 1e-12), 1.0)
        )
        self.run_stats["single_peptide_error_rate_upper_bound_source"] = "single_peptide_error_rate_upper_bound"
        self.run_stats["single_peptide_error_rate_upper_bound"] = float(self.single_peptide_error_rate_upper_bound)
        self.peptide_error_upper_by_peptide = {}
        if error_col and error_col in valid.columns:
            self.logger.info("Computing unit-aware per-peptide error upper bounds ...")
            pep_err = valid.groupby(seq_col)[error_col].max().reset_index()
            pep_err.columns = ["Peptide", "Error"]
            pep_err["Error"] = pd.to_numeric(pep_err["Error"], errors="coerce").clip(lower=1e-12, upper=1.0)
            pep_err = pep_err.dropna(subset=["Error"])
            self.peptide_error_upper_by_peptide = dict(
                zip(pep_err["Peptide"].astype(str), pep_err["Error"].astype(float))
            )

        self.run_stats["observed_unique_peptides"] = int(len(self.peptide_score))
        self.run_stats["unit_aware_valid_rows"] = int(len(valid))
        self.run_stats["unit_aware_samples_total"] = int(len(sample_ids))
        self.run_stats["unit_aware_samples_included"] = int(len(valid_sample_ids))
        self.run_stats["unit_aware_analysis_units"] = int(len(analysis_unit_ids))
        self.run_stats["unit_presence_rule"] = self.unit_presence_rule
        self.run_stats["unit_shared_mode"] = self.unit_shared_mode
        self.logger.info(
            f"Unit-aware observed peptides: {len(self.peptide_score)} unique across "
            f"{len(valid_sample_ids)} sample(s) and {len(analysis_unit_ids)} analysis unit(s)."
        )
        return True

    # =========================
    # Shared peptide degeneracy + weights
    # =========================
    def _compute_weight(self, d: int) -> float:
        """Shared-aware peptide weight given degeneracy d(p)=d."""
        d = int(max(d, 1))
        return 1.0 / float(d)

    def _calculate_peptide_degeneracy_and_unique_counts(
        self,
        all_matched_peptides: List[Tuple[str, Set[str], int]],
    ) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, str]]:
        """Compute peptide degeneracy d(p) and per-genome unique peptide counts."""
        self.logger.info("Computing peptide degeneracy d(p) and per-genome unique counts ...")

        genome_to_matched_peptides: Dict[str, Set[str]] = {}
        for genome_id, matched_peptides, _ in all_matched_peptides:
            genome_to_matched_peptides.setdefault(genome_id, set()).update(matched_peptides)

        num_target_genomes = len(genome_to_matched_peptides)

        peptide_genome_count = Counter()
        peptide_first_owner: Dict[str, str] = {}
        for matched_peptides in tqdm(genome_to_matched_peptides.values(), desc="Counting peptide occurrences"):
            for peptide in matched_peptides:
                peptide_genome_count[peptide] += 1
        for genome_id, matched_peptides in genome_to_matched_peptides.items():
            for peptide in matched_peptides:
                if peptide_genome_count.get(peptide, 0) == 1:
                    peptide_first_owner[peptide] = genome_id

        peptide_deg = dict(peptide_genome_count)

        genome_unique_counts: Dict[str, int] = {}
        for genome_id, matched_peptides in tqdm(genome_to_matched_peptides.items(), desc="Counting unique peptides per genome"):
            genome_unique_counts[genome_id] = sum(
                1 for peptide in matched_peptides if peptide_genome_count.get(peptide, 0) == 1
            )

        self.logger.info(
            f"d(p): {len(peptide_deg)} peptides across {num_target_genomes} genomes."
        )
        self.run_stats["num_target_genomes_for_degeneracy"] = int(num_target_genomes)
        return peptide_deg, genome_unique_counts, peptide_first_owner

    # =========================
    # Genome metrics
    # =========================
    def _calculate_genome_metrics(
        self,
        genome_data_list: List[Tuple[str, Set[str], int, int]],
        peptide_deg: Dict[str, int],
    ) -> pd.DataFrame:
        """Compute shared-aware metrics for each genome and record shared-stratum counts for knockoff."""
        self.knockoff_shared_stratum_counts_by_genome = {}

        out_rows = []
        default_score = 1.0

        for genome_id, matched_peptides, total_theoretical_peptides, unique_matched_peptides in tqdm(
            genome_data_list, desc="Computing genome metrics"
        ):
            num_peptides_matched = len(matched_peptides)

            if num_peptides_matched == 0 or total_theoretical_peptides == 0:
                out_rows.append({
                    "genome_id": genome_id,
                    "total_peptide_count": int(total_theoretical_peptides),
                    "num_peptides_matched": 0,
                    "num_peptides_unique": 0,
                    "peptide_match_ratio": 0.0,
                    "average_peptide_score": 0.0,
                    "effective_peptide_count": 0.0,
                    "weighted_evidence": 0.0,
                    "unique_weighted_evidence": 0.0,
                    "mean_degeneracy": 0.0,
                    "shared_fraction": 0.0,
                    "matched_peptide_count_shared": 0,
                    "effective_peptide_count_shared": 0.0,
                    "weighted_evidence_shared": 0.0,
                })
                self.knockoff_shared_stratum_counts_by_genome[genome_id] = Counter()
                continue

            peptide_scores: List[float] = []
            peptide_degeneracies: List[int] = []
            peptide_weights: List[float] = []
            weighted_contributions: List[float] = []
            unique_weighted_evidence = 0.0

            shared_matched_peptide_count = 0
            shared_peptide_weights: List[float] = []
            shared_weighted_contributions: List[float] = []
            strata_counter = Counter()

            for peptide in matched_peptides:
                score = float(self.peptide_score.get(peptide, default_score))
                degeneracy = int(peptide_deg.get(peptide, 1))
                weight = self._compute_weight(d=degeneracy)

                peptide_scores.append(score)
                peptide_degeneracies.append(degeneracy)
                peptide_weights.append(weight)
                weighted_contributions.append(weight * score)

                if degeneracy == 1:
                    unique_weighted_evidence += score
                else:
                    shared_matched_peptide_count += 1
                    shared_peptide_weights.append(weight)
                    shared_weighted_contributions.append(weight * score)
                    strata_counter[self._knock_stratum(d=degeneracy, pep_len=len(peptide))] += 1

            self.knockoff_shared_stratum_counts_by_genome[genome_id] = strata_counter

            average_peptide_score = float(np.mean(peptide_scores)) if peptide_scores else 0.0
            peptide_match_ratio = float(num_peptides_matched) / float(max(int(total_theoretical_peptides), 1))

            effective_peptide_count = float(np.sum(peptide_weights)) if peptide_weights else 0.0
            weighted_evidence = float(np.sum(weighted_contributions)) if weighted_contributions else 0.0

            effective_peptide_count_shared = float(np.sum(shared_peptide_weights)) if shared_peptide_weights else 0.0
            weighted_evidence_shared = float(np.sum(shared_weighted_contributions)) if shared_weighted_contributions else 0.0

            mean_degeneracy = float(np.mean(peptide_degeneracies)) if peptide_degeneracies else 0.0
            shared_fraction = (
                1.0 - (float(unique_matched_peptides) / float(num_peptides_matched))
                if num_peptides_matched > 0
                else 0.0
            )

            out_rows.append({
                "genome_id": genome_id,
                "total_peptide_count": int(total_theoretical_peptides),
                "num_peptides_matched": int(num_peptides_matched),
                "num_peptides_unique": int(unique_matched_peptides),
                "peptide_match_ratio": float(peptide_match_ratio),
                "average_peptide_score": float(average_peptide_score),
                "effective_peptide_count": float(effective_peptide_count),
                "weighted_evidence": float(weighted_evidence),
                "unique_weighted_evidence": float(unique_weighted_evidence),
                "mean_degeneracy": float(mean_degeneracy),
                "shared_fraction": float(shared_fraction),
                "matched_peptide_count_shared": int(shared_matched_peptide_count),
                "effective_peptide_count_shared": float(effective_peptide_count_shared),
                "weighted_evidence_shared": float(weighted_evidence_shared),
            })

        return pd.DataFrame(out_rows)

    # =========================
    # Lexicographic ranking
    # =========================
    def _rank_genomes(self, df_metrics: pd.DataFrame) -> pd.DataFrame:
        """
        Build a lexicographically ranked genome table.

        - This implementation enforces STRICT lexicographic ranking ("unique dominates"):
            1) num_peptides_unique (U)
            2) unique_weighted_evidence (UW)
            3) weighted_evidence (WE)
            4) effective_peptide_count (EP)
            5) peptide_match_ratio (MR)
            6) num_peptides_matched (M)
        Ties are broken deterministically by genome_id.

        The output rows are ordered from best to worst by the lexicographic rule above.
        """
        out = df_metrics.copy()

        # Mark target genomes (matched >= 1)
        if "num_peptides_matched" not in out.columns:
            raise ValueError("Missing required column: num_peptides_matched")
        out["_genomes_with_any_match"] = out["num_peptides_matched"].fillna(0).astype(int) >= 1

        # Ensure required ranking columns exist (fill missing with zeros)
        required_cols = [
            "num_peptides_unique",
            "unique_weighted_evidence",
            "weighted_evidence",
            "effective_peptide_count",
            "peptide_match_ratio",
            "num_peptides_matched",
        ]
        for c in required_cols:
            if c not in out.columns:
                out[c] = 0

        # Cast / sanitize types for stable sorting
        out["num_peptides_unique"] = pd.to_numeric(out["num_peptides_unique"], errors="coerce").fillna(0).astype(int)
        out["num_peptides_matched"] = pd.to_numeric(out["num_peptides_matched"], errors="coerce").fillna(0).astype(int)

        float_cols = [
            "unique_weighted_evidence",
            "weighted_evidence",
            "effective_peptide_count",
            "peptide_match_ratio",
        ]
        for c in float_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype(float)

        if "genome_id" not in out.columns:
            raise ValueError("Missing required column: genome_id")
        out["genome_id"] = out["genome_id"].astype(str)

        # STRICT lexicographic ordering: U > UW > WE > EP > MR > M
        sort_cols = [
            "num_peptides_unique",
            "unique_weighted_evidence",
            "weighted_evidence",
            "effective_peptide_count",
            "peptide_match_ratio",
            "num_peptides_matched",
            "genome_id",  # deterministic tie-breaker
        ]
        ascending = [False, False, False, False, False, False, True]

        # Use stable sort to ensure reproducibility across platforms
        ranked = out.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)

        return ranked


    # =========================
    # Knockoff helpers
    # =========================
    def _knock_deg_bin(self, d: int) -> int:
        """Map degeneracy d to a bin index."""
        d = int(max(d, 1))
        edges = list(self.degeneracy_bin_edges)
        if d <= edges[0]:
            return 0
        for i, e in enumerate(edges[1:], start=1):
            if d <= e:
                return i
        return len(edges)

    def _knock_len_bin(self, L: int) -> int:
        """Map peptide length to a bin index."""
        L = int(max(L, 0))
        edges = list(self.peptide_length_bin_edges)
        if L <= edges[0]:
            return 0
        for i, e in enumerate(edges[1:], start=1):
            if L <= e:
                return i
        return len(edges)

    def _knock_stratum(self, d: int, pep_len: int) -> Union[int, Tuple[int, int]]:
        """Stratum key for knockoff pools."""
        db = self._knock_deg_bin(d)
        if not self.use_length_strata:
            return int(db)
        lb = self._knock_len_bin(pep_len)
        return (int(db), int(lb))

    def _prepare_knockoff_pools(self, peptide_deg: Dict[str, int]) -> None:
        """Build stratum pools of shared peptide contributions (w*s) for observed peptides with d(p)>1."""
        pools: Dict[Union[int, Tuple[int, int]], List[float]] = {}
        for pep, s in self.peptide_score.items():
            d = int(peptide_deg.get(pep, 0))
            if d <= 1:
                continue
            w = self._compute_weight(d=d)
            key = self._knock_stratum(d=d, pep_len=len(pep))
            pools.setdefault(key, []).append(float(w * float(s)))

        self.knockoff_pools_weighted_contrib = {k: np.asarray(v, dtype=np.float32) for k, v in pools.items()}

        # --- NEW: pool summary stats (for paper diagnostics) ---
        rows = []
        for k, arr in (self.knockoff_pools_weighted_contrib or {}).items():
            if arr is None or arr.size == 0:
                rows.append({"stratum": str(k), "pool_size": 0})
                continue
            a = arr.astype(np.float64, copy=False)
            rows.append({
                "stratum": str(k),
                "pool_size": int(a.size),
                "mean": float(a.mean()),
                "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
                "p95": float(np.quantile(a, 0.95)),
                "p99": float(np.quantile(a, 0.99)),
                "min": float(a.min()),
                "max": float(a.max()),
            })
        self.knockoff_pool_stats = pd.DataFrame(rows).sort_values("pool_size", ascending=False) if rows else None

    def _build_knockoff_pools_for_peptides(
        self,
        observed_peptides: Set[str],
        peptide_deg: Dict[str, int],
    ) -> Dict[Union[int, Tuple[int, int]], np.ndarray]:
        """Build shared knockoff pools from a unit-specific observed peptide set."""
        pools: Dict[Union[int, Tuple[int, int]], List[float]] = {}
        for peptide in observed_peptides:
            d = int(peptide_deg.get(peptide, 0))
            if d <= 1:
                continue
            score = float(self.peptide_score.get(peptide, 1.0))
            weight = self._compute_weight(d=d)
            key = self._knock_stratum(d=d, pep_len=len(peptide))
            pools.setdefault(key, []).append(float(weight * score))
        return {key: np.asarray(values, dtype=np.float32) for key, values in pools.items()}

    def _mc_sum_from_pool(self, pool: Optional[np.ndarray], K: int, c: int, rng: np.random.Generator) -> np.ndarray:
        """Sample K times the sum of c draws (with replacement) from pool."""
        if c <= 0:
            return np.zeros(int(K), dtype=np.float64)
        if pool is None or pool.size == 0:
            return np.zeros(int(K), dtype=np.float64)

        K = int(K)
        c = int(c)

        # Chunked sampling to reduce peak memory
        block = int(max(1, self.knockoff_sample_block_size))
        out = np.zeros(K, dtype=np.float64)
        i = 0
        while i < K:
            j = min(K, i + block)
            # shape: (j-i, c)
            idx = rng.integers(0, int(pool.size), size=(j - i, c), endpoint=False)
            out[i:j] = pool[idx].sum(axis=1)
            i = j
        return out

    def _p_shared_knockoff_mc(
        self,
        gid: str,
        obs_shared_score: float,
        K: int,
        rng: np.random.Generator,
        return_moments: bool = False,
        pools: Optional[Dict[Union[int, Tuple[int, int]], np.ndarray]] = None,
        counts_by_genome: Optional[Dict[str, Counter]] = None,
    ) -> Union[float, Tuple[float, float, float, float, float]]:
        """Empirical p-value for shared evidence via knockoff Monte Carlo (optionally return null moments)."""
        counts_source = counts_by_genome if counts_by_genome is not None else self.knockoff_shared_stratum_counts_by_genome
        pools_source = pools if pools is not None else self.knockoff_pools_weighted_contrib
        counts = counts_source.get(gid, None)
        if not counts:
            return (1.0, 0.0, 0.0, 0.0, 0.0) if return_moments else 1.0

        null_sum = np.zeros(int(K), dtype=np.float64)
        for key, c in counts.items():
            pool = pools_source.get(key, None) if pools_source else None
            null_sum += self._mc_sum_from_pool(pool=pool, K=int(K), c=int(c), rng=rng)

        ge = float(np.sum(null_sum >= float(obs_shared_score)))
        p = (1.0 + ge) / (1.0 + float(K))

        if not return_moments:
            return float(p)

        mu = float(null_sum.mean())
        sd = float(null_sum.std(ddof=1)) if int(K) > 1 else 0.0
        p95 = float(np.quantile(null_sum, 0.95))
        p99 = float(np.quantile(null_sum, 0.99))
        return float(p), mu, sd, p95, p99

    @staticmethod
    def _fisher_p_2(p1: float, p2: float) -> float:
        """Fisher combine two p-values (df=4) using chi-square survival approximation."""
        p1 = float(min(max(p1, 1e-300), 1.0))
        p2 = float(min(max(p2, 1e-300), 1.0))
        stat = -2.0 * (np.log(p1) + np.log(p2))  # chi-square with df=4
        # sf for df=4 has closed form: exp(-x/2) * (1 + x/2)
        x = float(stat)
        return float(np.exp(-x / 2.0) * (1.0 + x / 2.0))

    @staticmethod
    def _bh_qvalues(pvals: np.ndarray) -> np.ndarray:
        """Benjamini-Hochberg q-values for a 1D array of p-values."""
        p = np.asarray(pvals, dtype=float)
        n = int(p.size)
        if n == 0:
            return p

        order = np.argsort(p)
        ranked = p[order]
        q = ranked * float(n) / (np.arange(1, n + 1, dtype=float))
        # enforce monotonicity
        q = np.minimum.accumulate(q[::-1])[::-1]
        q = np.clip(q, 0.0, 1.0)
        out = np.empty_like(q)
        out[order] = q
        return out

    def _unit_unique_pvalue_stats_for_genome(
        self,
        gid: str,
        matched_peptides: Set[str],
        observed_unique: int,
        observed_unique_pool_size: int,
        min_unique_for_unique_pvalue: int,
        mode: str,
        peptide_deg: Dict[str, int],
    ) -> dict:
        """Unique-evidence p-value stats for one genome in one analysis unit."""
        U = int(observed_unique)
        mode = _normalize_unique_pvalue_mode(mode)

        alpha = float(min(max(self.single_peptide_error_rate_upper_bound, 1e-12), 1.0))
        S = int(observed_unique_pool_size)
        A = int(self.genome_theoretical_unique_peptides.get(gid, 0))
        A_total = int(max(self.total_theoretical_unique_peptides_all_genomes, 1))
        theoretical_unique: object = pd.NA
        p_unique = 1.0
        p_unique_depth = 1.0
        expected = 0.0
        fold = 0.0
        gate_pass = False
        null_model = ""
        unique_effective_count = float(U)
        count_model = "raw"
        error_source_used = ""

        if mode == "hypergeometric-opportunity":
            theoretical_unique = int(A)
            expected = float(S) * float(A) / float(A_total) if A_total > 0 else 0.0
            fold = float(U) / max(expected, 1e-12)
            gate_pass = bool(U >= int(min_unique_for_unique_pvalue))
            null_model = "hypergeometric"
            if gate_pass and S > 0 and A > 0 and A_total > 0:
                p_unique_depth = self._hypergeom_tail_pvalue(
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
                peptide_error_upper_by_peptide=self.peptide_error_upper_by_peptide,
                error_source=self.unique_peptide_error_source,
                unique_count_power=self.unique_count_power,
                unique_count_cap=self.unique_count_cap,
            )
            count_model = f"power:{float(self.unique_count_power):g}"
            p_unique_depth = p_unique
            gate_pass = U > 0
            null_model = error_source_used

        return {
            "p_unique": _clip_pvalue(p_unique),
            "p_unique_depth": _clip_pvalue(p_unique_depth),
            "unique_observed": int(U),
            "unique_expected_null": float(expected),
            "unique_depth_fold": float(fold),
            "unique_depth_null_model": null_model,
            "unique_pvalue_mode": mode,
            "unique_peptide_error_source": error_source_used,
            "unique_gate_pass": bool(gate_pass),
            "theoretical_unique_peptides": theoretical_unique,
            "unique_effective_count": float(unique_effective_count),
            "unique_pvalue_count_model": count_model,
        }

    def _unit_shared_metrics_for_genome(
        self,
        genome_id: str,
        matched_peptides: Set[str],
        peptide_deg: Dict[str, int],
    ) -> dict:
        """Compute shared knockoff metrics for one genome in one analysis unit."""
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

        for peptide in matched_peptides:
            d = int(peptide_deg.get(peptide, 1))
            score = float(self.peptide_score.get(peptide, 1.0))
            weight = float(self._compute_weight(d=d))
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
                strata_counter[self._knock_stratum(d=d, pep_len=len(peptide))] += 1

        total_theoretical = int(self.genome_total_theoretical_peptides.get(genome_id, 0))
        effective_peptide_count = float(np.sum(peptide_weights)) if peptide_weights else 0.0
        weighted_evidence = float(np.sum(weighted_contributions)) if weighted_contributions else 0.0
        effective_shared = float(np.sum(shared_weights)) if shared_weights else 0.0
        weighted_shared = float(np.sum(shared_contributions)) if shared_contributions else 0.0

        return {
            "num_peptides_matched": int(total_matched),
            "num_peptides_unique": int(unique_count),
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


    def _add_knockoff_existence_stats(
        self,
        df_scored: pd.DataFrame,
        unique_pvalue_mode: str = DEFAULT_UNIQUE_PVALUE_MODE,
        unique_peptide_error_source: str = DEFAULT_UNIQUE_PEPTIDE_ERROR_SOURCE,
        min_unique_for_unique_pvalue: int = 3,
        unique_count_power: float = DEFAULT_UNIQUE_COUNT_POWER,
        unique_count_cap: Optional[float] = None,
    ) -> pd.DataFrame:
        """Add per-genome knockoff existence p/q-values."""
        unique_count_power = float(unique_count_power)
        if not np.isfinite(unique_count_power) or not (0 < unique_count_power <= 1):
            raise ValueError("unique_count_power must be in the interval (0, 1].")
        if unique_count_cap is not None:
            cap_value = float(unique_count_cap)
            if not np.isfinite(cap_value) or cap_value <= 0:
                raise ValueError("unique_count_cap must be positive or None.")
        unique_peptide_error_source = _normalize_unique_peptide_error_source(unique_peptide_error_source)

        out = df_scored.copy()
        # Conservative defaults for genomes with no match / skipped inference.
        out["p_shared_knock"] = 1.0
        out["p_unique"] = 1.0
        out["p_unique_depth"] = 1.0
        out["p_presence"] = 1.0
        out["q_presence"] = 1.0
        out["presence_score"] = 0.0
        out["unique_observed"] = 0
        out["unique_expected_null"] = 0.0
        out["unique_depth_fold"] = 0.0
        out["unique_gate_pass"] = False
        out["unique_depth_null_model"] = ""
        out["theoretical_unique_peptides"] = pd.NA
        out["unique_pvalue_mode"] = unique_pvalue_mode
        out["unique_peptide_error_source"] = unique_peptide_error_source
        out["unique_effective_count"] = 0.0
        out["unique_pvalue_count_model"] = ""

        # --- NEW: knockoff null diagnostics ---
        out["null_mean_shared"] = 0.0
        out["null_sd_shared"] = 0.0
        out["null_p95_shared"] = 0.0
        out["null_p99_shared"] = 0.0
        out["z_shared"] = 0.0

        if self.peptide_degeneracy is None:
            raise RuntimeError("Knockoff requires peptide_deg to be set.")

        if self.knockoff_pools_weighted_contrib is None:
            self._prepare_knockoff_pools(self.peptide_degeneracy)

        # Use independent RNG streams for stage-1 and stage-2 so refinement is reproducible
        # regardless of how many candidates enter stage-2.
        seed = int(self.knockoff_random_seed)
        ss = np.random.SeedSequence(seed)
        children = ss.spawn(2)
        rng_stage1 = np.random.default_rng(children[0])
        rng_stage2 = np.random.default_rng(children[1])

        K1 = int(max(50, self.knockoff_mc_iterations))
        K2 = None
        if self.knockoff_stage2_mc_iterations is not None:
            K2 = int(max(50, self.knockoff_stage2_mc_iterations))
        peptide_error_upper = float(min(max(self.single_peptide_error_rate_upper_bound, 1e-12), 1.0))
        mode = _normalize_unique_pvalue_mode(unique_pvalue_mode)
        if int(min_unique_for_unique_pvalue) < 0:
            raise ValueError("min_unique_for_unique_pvalue must be >= 0.")
        self.unique_pvalue_mode = mode
        self.unique_peptide_error_source = unique_peptide_error_source
        self.min_unique_for_unique_pvalue = int(min_unique_for_unique_pvalue)
        self.unique_count_power = float(unique_count_power)
        self.unique_count_cap = unique_count_cap
        error_col = self.run_stats.get("peptide_error_col", None)
        use_per_peptide_error = bool(mode == "alpha-upper-bound" and unique_peptide_error_source == "peptide-error-column")
        uses_pep_column = bool(use_per_peptide_error and isinstance(error_col, str) and error_col.upper() == "PEP")
        source_col_display = str(error_col) if error_col is not None else "none"
        self.run_stats["unique_pvalue_mode"] = mode
        self.run_stats["unique_peptide_error_source"] = unique_peptide_error_source
        self.run_stats["min_unique_for_unique_pvalue"] = int(min_unique_for_unique_pvalue)
        self.run_stats["unique_count_power"] = float(unique_count_power)
        self.run_stats["unique_count_cap"] = (
            float(unique_count_cap) if unique_count_cap is not None else None
        )
        self.run_stats["unique_pvalue_count_model"] = (
            f"power:{unique_count_power:g}" if mode == "alpha-upper-bound" else "raw"
        )
        self.run_stats["unique_pvalue_uses_per_peptide_error"] = bool(use_per_peptide_error)
        self.run_stats["unique_pvalue_error_source_col"] = str(error_col) if use_per_peptide_error and error_col is not None else None
        self.run_stats["unique_pvalue_uses_pep_column"] = bool(uses_pep_column)
        if mode == "hypergeometric-opportunity":
            self.logger.info(
                "Unique p-value mode: [hypergeometric-opportunity] "
                f"null_model=hypergeometric, min_unique={int(min_unique_for_unique_pvalue)}, "
                f"observed_unique_pool={int(self.observed_unique_peptide_pool_size)}, "
                f"theoretical_unique_universe={int(self.total_theoretical_unique_peptides_all_genomes)}"
            )
        elif use_per_peptide_error:
            self.logger.info(
                f"Unique p-value mode: [alpha-upper-bound], error_source={'PEP' if uses_pep_column else 'peptide-error-column'}, "
                f"count_model={self.run_stats['unique_pvalue_count_model']}, "
                f"source_col='{source_col_display}', "
                f"per_peptide_error_n={len(self.peptide_error_upper_by_peptide)}, "
                f"alpha={peptide_error_upper:.4g}"
            )
        else:
            reason = "global_alpha_error_source" if mode == "alpha-upper-bound" else "not_alpha_upper_bound_mode"
            formula = "exp(sum(log(epsilon_i)) * U_eff / U_raw)"
            self.logger.info(
                f"Unique p-value mode: [{mode}] alpha={peptide_error_upper:.4g}, "
                f"error_source={unique_peptide_error_source}, "
                f"formula={formula}, count_model={self.run_stats['unique_pvalue_count_model']}, "
                f"reason={reason}"
            )

        target_mask = out["_genomes_with_any_match"].astype(bool)
        target_df = out.loc[target_mask].copy()

        if len(target_df) == 0:
            out["pass_q_0_01"] = False
            out["pass_q_0_05"] = False
            return out

        if self.knockoff_top_n_targets is not None:
            topN = int(self.knockoff_top_n_targets)
            target_df = target_df.sort_values("evidence_rank", ascending=True, kind="mergesort").head(topN)

        # -----------------
        # Stage 1 (fast screen)
        # -----------------
        for idx, row in tqdm(
            target_df.iterrows(), total=len(target_df), desc=f"Knockoff p-values stage1 (K={K1})"
        ):
            genome_id = row["genome_id"]
            observed_shared_evidence = float(row.get("weighted_evidence_shared", 0.0))
            result = self._p_shared_knockoff_mc(
                gid=genome_id,
                obs_shared_score=observed_shared_evidence,
                K=K1,
                rng=rng_stage1,
                return_moments=True,
            )
            p_shared, mu, sd, p95, p99 = result if isinstance(result, tuple) else (result, 0.0, 0.0, 0.0, 0.0)

            unique_stats = self._unique_pvalue_stats_for_genome(
                gid=genome_id,
                u_observed=int(row.get("num_peptides_unique", 0)),
            )
            p_unique = float(unique_stats["p_unique"])
            p_existence = self._fisher_p_2(p1=p_shared, p2=p_unique)

            out.at[idx, "p_shared_knock"] = p_shared
            for key, value in unique_stats.items():
                out.at[idx, key] = value
            out.at[idx, "p_presence"] = p_existence

            out.at[idx, "null_mean_shared"] = mu
            out.at[idx, "null_sd_shared"] = sd
            out.at[idx, "null_p95_shared"] = p95
            out.at[idx, "null_p99_shared"] = p99
            out.at[idx, "z_shared"] = (observed_shared_evidence - mu) / (sd + 1e-12)

        # -----------------
        # Stage 2 (refine near-threshold genomes)
        # -----------------
        if K2 is not None:
            ranges = list(self.knockoff_stage2_p_exist_ranges or [])

            def _in_any_range(x: float) -> bool:
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return False
                xv = float(x)
                for a, b in ranges:
                    lo = float(min(a, b))
                    hi = float(max(a, b))
                    if lo <= xv <= hi:
                        return True
                return False

            if ranges:
                # Only consider genomes we actually computed in stage-1 (i.e., those in target_df)
                stage1_idx = target_df.index.to_numpy()
                p_stage1 = out.loc[stage1_idx, "p_presence"].to_numpy(dtype=float)
                cand_mask = np.asarray([_in_any_range(x) for x in p_stage1], dtype=bool)
                cand_idx = stage1_idx[cand_mask]

                if cand_idx.size > 0:
                    for idx in tqdm(cand_idx, total=int(cand_idx.size), desc=f"Knockoff p-values stage2 (K={K2})"):
                        row = out.loc[idx]
                        genome_id = row["genome_id"]
                        observed_shared_evidence = float(row.get("weighted_evidence_shared", 0.0))
                        result = self._p_shared_knockoff_mc(
                            gid=genome_id,
                            obs_shared_score=observed_shared_evidence,
                            K=K2,
                            rng=rng_stage2,
                            return_moments=True,
                        )
                        p_shared, mu, sd, p95, p99 = result if isinstance(result, tuple) else (result, 0.0, 0.0, 0.0, 0.0)

                        unique_stats = self._unique_pvalue_stats_for_genome(
                            gid=genome_id,
                            u_observed=int(row.get("num_peptides_unique", 0)),
                        )
                        p_unique = float(unique_stats["p_unique"])
                        p_existence = self._fisher_p_2(p1=p_shared, p2=p_unique)

                        out.at[idx, "p_shared_knock"] = p_shared
                        for key, value in unique_stats.items():
                            out.at[idx, key] = value
                        out.at[idx, "p_presence"] = p_existence

                        out.at[idx, "null_mean_shared"] = mu
                        out.at[idx, "null_sd_shared"] = sd
                        out.at[idx, "null_p95_shared"] = p95
                        out.at[idx, "null_p99_shared"] = p99
                        out.at[idx, "z_shared"] = (observed_shared_evidence - mu) / (sd + 1e-12)

        all_p = out.loc[target_mask, "p_presence"].to_numpy(dtype=float)
        out.loc[target_mask, "q_presence"] = self._bh_qvalues(all_p)
        qvals = pd.to_numeric(out["q_presence"], errors="coerce")
        valid = qvals.notna()
        out.loc[valid, "presence_score"] = -np.log10(np.clip(qvals.loc[valid].to_numpy(dtype=float), 1e-300, 1.0))

        out["pass_q_0_01"] = (out["q_presence"] <= 0.01) & (out["_genomes_with_any_match"])
        out["pass_q_0_05"] = (out["q_presence"] <= 0.05) & (out["_genomes_with_any_match"])

        return out

    # =========================
    # Coverage (reference only; not used for final calling)
    # =========================
    def _add_coverage_stats(
        self,
        df_scored: pd.DataFrame,
        order_col: str = "presence_rank",
    ) -> pd.DataFrame:
        """
        Add coverage statistics as a *human reference* (not used in q-value computation).

        Coverage is computed along the existing ranking order (default: presence_rank),
        and reports how many unique, matchable observed peptides are cumulatively explained.

        Columns added:
        - peptides_added_in_ranking
        - cumulative_covered_peptides
        - cumulative_coverage_percent
        - peptide_coverage_rank

        Notes:
        - Denominator (total_matchable_peptides) is the number of observed peptides that map to ≥1 genome.
                    This equals len(self.peptide_degeneracy).
        """
        out = df_scored.copy()

        # Initialize columns (use nullable integer dtype for integer counts so assigned ints remain ints)
        out["peptides_added_in_ranking"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="Int64")
        out["cumulative_covered_peptides"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="Int64")
        out["cumulative_coverage_percent"] = np.nan
        out["peptide_coverage_rank"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="Int64")

        if order_col not in out.columns:
            # Fallback: keep current row order
            order_idx = out.index.to_numpy()
        else:
            tmp = out.sort_values(order_col, ascending=True, kind="mergesort").reset_index()
            order_idx = tmp["index"].to_numpy()

        # Determine denominator: total matchable peptides
        if self.peptide_degeneracy is not None:
            total_matchable = int(len(self.peptide_degeneracy))
        else:
            # Robust fallback: union over all genomes
            seen = set()
            for _, ps in self.genome_matched_peptides.items():
                for p in ps:
                    if p not in seen:
                        seen.add(p)
            total_matchable = int(len(seen))

        if total_matchable <= 0:
            self.logger.warning("Coverage skipped: total_matchable_peptides=0.")
            return out

        # Compute cumulative union along the ranking
        covered = set()
        cov_rank = 0
        for ridx in order_idx:
            row = out.loc[ridx]
            gid = row["genome_id"]
            if not bool(row.get("_genomes_with_any_match", False)):
                continue

            ps = self.genome_matched_peptides.get(gid, set())
            if not ps:
                continue

            before = len(covered)
            covered |= ps
            added = len(covered) - before

            cov_rank += 1
            out.at[ridx, "peptides_added_in_ranking"] = int(added)
            out.at[ridx, "cumulative_covered_peptides"] = int(len(covered))
            out.at[ridx, "cumulative_coverage_percent"] = float(len(covered) * 100.0 / total_matchable)
            out.at[ridx, "peptide_coverage_rank"] = int(cov_rank)

        return out

    # =========================
    # Top-level pipeline
    # =========================

    def _export_temp_artifacts(
        self,
        out_dir: str,
        stem: str,
        df_scored: pd.DataFrame,
        export_peptide_contrib_topN: int = 0,
    ) -> None:
        """Export additional statistics for paper figures into out_dir/<stem>_artifacts/."""
        temp_dir = os.path.join(out_dir, f"{stem}_artifacts")
        os.makedirs(temp_dir, exist_ok=True)

        # --------------- run summary JSON ---------------
        meta = dict(self.run_stats) if isinstance(self.run_stats, dict) else {}
        meta["use_length_strata"] = bool(self.use_length_strata)
        meta["degeneracy_bin_edges"] = list(self.degeneracy_bin_edges)
        meta["peptide_length_bin_edges"] = list(self.peptide_length_bin_edges)
        meta["knockoff_mc_iterations"] = int(self.knockoff_mc_iterations)
        meta["knockoff_stage2_mc_iterations"] = int(self.knockoff_stage2_mc_iterations) if self.knockoff_stage2_mc_iterations is not None else None
        meta["knockoff_stage2_p_exist_ranges"] = list(self.knockoff_stage2_p_exist_ranges or [])
        meta["knockoff_top_n_targets"] = int(self.knockoff_top_n_targets) if self.knockoff_top_n_targets is not None else None
        meta["knockoff_random_seed"] = int(self.knockoff_random_seed)
        meta["python_version"] = sys.version
        meta["platform"] = platform.platform()
        meta["num_workers"] = int(self.num_workers)

        try:
            meta["genomes_total"] = int(len(df_scored))
            meta["genomes_with_any_match"] = int(df_scored.get("_genomes_with_any_match", False).sum())
            if "pass_q_0_01" in df_scored.columns:
                meta["genomes_q_le_0p01"] = int(df_scored["pass_q_0_01"].fillna(False).sum())
            if "pass_q_0_05" in df_scored.columns:
                meta["genomes_q_le_0p05"] = int(df_scored["pass_q_0_05"].fillna(False).sum())
        except Exception:
            pass

        meta["timing_seconds"] = {k: float(v) for k, v in (self.timing_stats or {}).items()}

        with open(os.path.join(temp_dir, "run_summary.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # --------------- full tables ---------------
        df_scored.to_csv(os.path.join(temp_dir, "full_internal_metrics.tsv"), sep="\t", index=False)

        # --------------- knockoff pools stats ---------------
        if self.knockoff_pool_stats is not None and isinstance(self.knockoff_pool_stats, pd.DataFrame) and len(self.knockoff_pool_stats) > 0:
            self.knockoff_pool_stats.to_csv(os.path.join(temp_dir, "knockoff_pools.tsv"), sep="\t", index=False)

        # --------------- degeneracy histogram (peptide-space) ---------------
        if self.peptide_degeneracy is not None and isinstance(self.peptide_degeneracy, dict) and len(self.peptide_degeneracy) > 0:
            ds = pd.Series(list(self.peptide_degeneracy.values()), dtype=int)
            bins = [-np.inf, 1, 5, 20, 100, 500, np.inf]
            labels = ["1", "2-5", "6-20", "21-100", "101-500", ">500"]
            h = pd.cut(ds, bins=bins, labels=labels).value_counts().reindex(labels).fillna(0).astype(int)
            frac = (h / max(int(h.sum()), 1)).astype(float)
            deg_df = pd.DataFrame({"deg_bin": labels, "count": h.values, "fraction": frac.values})
            deg_df.to_csv(os.path.join(temp_dir, "degeneracy_hist.tsv"), sep="\t", index=False)

        # --------------- p_shared histogram (diagnostic) ---------------
        if "p_shared_knock" in df_scored.columns and "_genomes_with_any_match" in df_scored.columns:
            def _hist(series: pd.Series, tag: str) -> pd.DataFrame:
                s = pd.to_numeric(series, errors="coerce").dropna()
                if len(s) == 0:
                    return pd.DataFrame(columns=["set", "bin", "count", "fraction"])
                edges = [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0]
                bins = pd.cut(s, bins=edges, include_lowest=True)
                h = bins.value_counts().sort_index()
                frac = (h / float(h.sum())).astype(float)
                return pd.DataFrame({"set": tag, "bin": h.index.astype(str), "count": h.values.astype(int), "fraction": frac.values})

            target = df_scored.loc[df_scored["_genomes_with_any_match"]].copy()
            h1 = _hist(target["p_shared_knock"], "all_targets")
            h2 = pd.DataFrame()
            if "num_peptides_unique" in target.columns:
                h2 = _hist(target.loc[target["num_peptides_unique"].fillna(0).astype(int) == 0, "p_shared_knock"], "unique0_targets")
            hs = pd.concat([h1, h2], axis=0, ignore_index=True)
            hs.to_csv(os.path.join(temp_dir, "p_shared_hist.tsv"), sep="\t", index=False)

        # --------------- q calling curve ---------------
        if "q_presence" in df_scored.columns and "_genomes_with_any_match" in df_scored.columns:
            target = df_scored.loc[df_scored["_genomes_with_any_match"]].copy()
            q = pd.to_numeric(target["q_presence"], errors="coerce")
            thresholds = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
            rows = []
            for t in thresholds:
                rows.append({"q_threshold": float(t), "n_called": int((q <= t).sum())})
            pd.DataFrame(rows).to_csv(os.path.join(temp_dir, "q_calling_curve.tsv"), sep="\t", index=False)


        # --------------- shared stratum counts (sparse long table) ---------------
        if self.knockoff_shared_stratum_counts_by_genome:
            rows = []
            for gid, ctr in self.knockoff_shared_stratum_counts_by_genome.items():
                if not ctr:
                    continue
                for k, c in ctr.items():
                    rows.append({"genome_id": gid, "stratum": str(k), "count": int(c)})
            if rows:
                pd.DataFrame(rows).to_csv(os.path.join(temp_dir, "shared_stratum_counts.tsv"), sep="\t", index=False)

        # --------------- top-N peptide contribution table (optional) ---------------
        topN = int(export_peptide_contrib_topN) if export_peptide_contrib_topN is not None else 0
        if topN > 0 and self.peptide_degeneracy is not None:
            if "_genomes_with_any_match" in df_scored.columns:
                target = df_scored.loc[df_scored["_genomes_with_any_match"]].copy()
            else:
                target = df_scored.copy()
            if "evidence_rank" in target.columns:
                target = target.sort_values("evidence_rank", ascending=True, kind="mergesort")
            target = target.head(topN)
            out_rows = []
            for _, r in target.iterrows():
                gid = str(r["genome_id"])
                peps = self.genome_matched_peptides.get(gid, set())
                if not peps:
                    continue
                for pep in peps:
                    d = int(self.peptide_degeneracy.get(pep, 1))
                    w = float(self._compute_weight(d=d))
                    s = float(self.peptide_score.get(pep, 1.0))
                    is_unique = (d == 1)
                    out_rows.append({
                        "genome_id": gid,
                        "peptide": pep,
                        "pep_len": int(len(pep)),
                        "degeneracy": int(d),
                        "stratum": str(self._knock_stratum(d=d, pep_len=len(pep))),
                        "score": float(s),
                        "weight": float(w),
                        "contribution": float(w * s),
                        "is_unique": bool(is_unique),
                    })

            if out_rows:
                pd.DataFrame(out_rows).to_csv(
                    os.path.join(temp_dir, f"top{topN}_peptide_contrib.tsv"),
                    sep="\t",
                    index=False
                )

    def _compute_unit_aware_presence(
        self,
        df_scored: pd.DataFrame,
        peptide_deg: Dict[str, int],
        peptide_unique_owner: Dict[str, str],
        mode: str,
        min_unique_for_unique_pvalue: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute per-analysis-unit genome presence using already scanned genome matches."""
        from scipy.sparse import csr_matrix

        if not self.unit_aware_enabled or self.unit_presence_matrix is None:
            raise RuntimeError("Unit-aware peptide presence is not available. Call read_unit_aware_peptide_file first.")
        self.unit_presence_rule = "union"
        self.unit_shared_mode = "per-unit"

        genome_ids = [str(x) for x in df_scored["genome_id"].astype(str).tolist()]
        genome_index = {genome_id: i for i, genome_id in enumerate(genome_ids)}
        n_genomes = len(genome_ids)
        n_peptides = len(self.unit_peptides)
        n_units = len(self.unit_analysis_unit_ids)

        if n_genomes == 0 or n_peptides == 0 or n_units == 0:
            empty_unit = pd.DataFrame()
            empty_summary = pd.DataFrame()
            return empty_unit, empty_summary

        self.logger.info(
            f"Preparing unit-aware sparse genome matrices for {n_genomes} genome(s), "
            f"{n_peptides} peptide(s), and {n_units} analysis unit(s) ..."
        )
        g_rows: List[int] = []
        g_cols: List[int] = []
        for genome_id in genome_ids:
            row_idx = genome_index[genome_id]
            for peptide in self.genome_matched_peptides.get(genome_id, set()):
                col_idx = self.unit_peptide_index.get(peptide)
                if col_idx is not None:
                    g_rows.append(row_idx)
                    g_cols.append(col_idx)
        g_data = np.ones(len(g_rows), dtype=COUNT_DTYPE)
        genome_peptide = csr_matrix(
            (g_data, (g_rows, g_cols)),
            shape=(n_genomes, n_peptides),
            dtype=COUNT_DTYPE,
        )
        self.logger.info(f"Genome x unit-aware peptide matrix built with {len(g_rows)} non-zero entry/entries.")

        self.logger.info("Preparing unit-aware genome-unique peptide matrix ...")
        gu_rows: List[int] = []
        gu_cols: List[int] = []
        for peptide, owner in peptide_unique_owner.items():
            row_idx = genome_index.get(str(owner))
            col_idx = self.unit_peptide_index.get(str(peptide))
            if row_idx is not None and col_idx is not None:
                gu_rows.append(row_idx)
                gu_cols.append(col_idx)
        gu_data = np.ones(len(gu_rows), dtype=COUNT_DTYPE)
        genome_unique_peptide = csr_matrix(
            (gu_data, (gu_rows, gu_cols)),
            shape=(n_genomes, n_peptides),
            dtype=COUNT_DTYPE,
        )
        self.logger.info(f"Genome x unique-peptide matrix built with {len(gu_rows)} non-zero entry/entries.")

        self.logger.info("Multiplying genome peptide matrices by unit peptide presence matrix ...")
        x_unit = self.unit_presence_matrix.astype(COUNT_DTYPE)
        matched_matrix = (genome_peptide @ x_unit).astype(COUNT_DTYPE)
        unique_matrix = (genome_unique_peptide @ x_unit).astype(COUNT_DTYPE)
        if matched_matrix.min() < 0 or unique_matrix.min() < 0:
            raise RuntimeError("Negative peptide counts detected; check sparse matrix dtype.")
        self.logger.info(
            f"Unit-aware count matrices ready: matched nnz={matched_matrix.nnz}, unique nnz={unique_matrix.nnz}."
        )

        peptide_by_index = list(self.unit_peptides)
        lineage_map: Dict[str, object] = {}
        if "Lineage" in df_scored.columns:
            lineage_map = dict(zip(df_scored["genome_id"].astype(str), df_scored["Lineage"]))

        mode = _normalize_unique_pvalue_mode(mode)
        a_total = int(self.total_theoretical_unique_peptides_all_genomes)
        if mode == "hypergeometric-opportunity" and a_total <= 0:
            raise ValueError(
                "Unit-aware hypergeometric-opportunity p-values require theoretical unique peptide opportunity."
            )

        gate_min = int(min_unique_for_unique_pvalue)
        K1 = int(max(50, self.knockoff_mc_iterations))
        K2 = int(max(50, self.knockoff_stage2_mc_iterations)) if self.knockoff_stage2_mc_iterations is not None else None
        ranges = list(self.knockoff_stage2_p_exist_ranges or [])

        unit_log_interval = max(1, n_units // 10)
        self.logger.info(
            f"Computing unit-aware per-unit shared knockoff and {mode} unique p/q values "
            f"for {n_units} unit(s) x {n_genomes} genome(s) ..."
        )
        resolved_workers = _resolve_worker_count(self.num_workers, logger=self.logger)
        parallel_unit_workers = min(int(resolved_workers), int(n_units))
        use_parallel = bool(self.num_workers > 1 and n_units > 1 and parallel_unit_workers > 1)
        effective_unit_workers = parallel_unit_workers if use_parallel else 1
        self.run_stats["unit_aware_parallelized"] = bool(use_parallel)
        self.run_stats["unit_aware_num_workers"] = int(effective_unit_workers)
        self.run_stats["unit_aware_parallel_unit_count"] = int(n_units if use_parallel else 0)
        if use_parallel:
            self.logger.info(
                f"Unit-aware scoring running in parallel with {effective_unit_workers} worker(s) "
                f"across {n_units} analysis unit(s)."
            )
        else:
            self.logger.info(
                f"Unit-aware scoring running serially with 1 worker across {n_units} analysis unit(s)."
            )

        worker_context = {
            "genome_ids": genome_ids,
            "genome_matched_peptides": self.genome_matched_peptides,
            "peptide_deg": peptide_deg,
            "peptide_score": self.peptide_score,
            "genome_total_theoretical_peptides": self.genome_total_theoretical_peptides,
            "genome_theoretical_unique_peptides": self.genome_theoretical_unique_peptides,
            "total_theoretical_unique_peptides_all_genomes": int(
                self.total_theoretical_unique_peptides_all_genomes
            ),
            "single_peptide_error_rate_upper_bound": float(self.single_peptide_error_rate_upper_bound),
            "peptide_error_upper_by_peptide": self.peptide_error_upper_by_peptide,
            "unique_peptide_error_source": str(self.unique_peptide_error_source),
            "unique_count_power": float(self.unique_count_power),
            "unique_count_cap": self.unique_count_cap,
            "lineage_map": lineage_map,
            "mode": mode,
            "min_unique_for_unique_pvalue": int(gate_min),
            "knockoff_mc_iterations": int(K1),
            "knockoff_stage2_mc_iterations": K2,
            "knockoff_stage2_p_exist_ranges": ranges,
            "knockoff_sample_block_size": int(self.knockoff_sample_block_size),
            "knockoff_random_seed": int(self.knockoff_random_seed),
            "use_length_strata": bool(self.use_length_strata),
            "degeneracy_bin_edges": list(self.degeneracy_bin_edges),
            "peptide_length_bin_edges": list(self.peptide_length_bin_edges),
        }
        worker_args = []
        for unit_idx, unit_id in enumerate(self.unit_analysis_unit_ids):
            matched_counts = np.asarray(matched_matrix[:, unit_idx].todense()).ravel().astype(int)
            unit_peptide_indices = self.unit_presence_matrix[:, unit_idx].nonzero()[0]
            unit_observed_peptides = {peptide_by_index[int(i)] for i in unit_peptide_indices}
            args_for_unit = {
                "unit_idx": int(unit_idx),
                "analysis_unit_id": str(unit_id),
                "matched_counts": matched_counts,
                "unit_observed_peptides": tuple(sorted(unit_observed_peptides)),
                "n_samples_in_unit": int(self.unit_sample_counts.get(unit_id, 0)),
            }
            worker_args.append(args_for_unit)

        unit_results = []
        if use_parallel:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=effective_unit_workers,
                initializer=_init_unit_aware_worker,
                initargs=(worker_context,),
            ) as executor:
                future_to_unit_idx = {
                    executor.submit(_compute_unit_aware_single_unit_worker, args): int(args["unit_idx"])
                    for args in worker_args
                }
                completed = 0
                for future in concurrent.futures.as_completed(future_to_unit_idx):
                    unit_results.append(future.result())
                    completed += 1
                    if completed == 1 or completed % unit_log_interval == 0 or completed == n_units:
                        self.logger.info(
                            f"Unit-aware p/q progress: {completed}/{n_units} analysis unit(s) processed."
                        )
        else:
            serial_context = dict(worker_context)
            serial_context["shared_knockoff_func"] = self._p_shared_knockoff_mc
            _init_unit_aware_worker(serial_context)
            for args in worker_args:
                unit_results.append(_compute_unit_aware_single_unit_worker(args))
                completed = int(args["unit_idx"]) + 1
                if completed == 1 or completed % unit_log_interval == 0 or completed == n_units:
                    self.logger.info(
                        f"Unit-aware p/q progress: {completed}/{n_units} analysis unit(s) processed."
                    )

        rows = []
        for result in sorted(unit_results, key=lambda item: int(item["unit_idx"])):
            rows.extend(result["rows"])

        unit_level_df = pd.DataFrame(rows)
        if "Lineage" in unit_level_df.columns and unit_level_df["Lineage"].isna().all():
            unit_level_df = unit_level_df.drop(columns=["Lineage"])

        self.logger.info("Building unit-aware cohort summary table ...")
        summary_rows = []
        grouped = unit_level_df.groupby("genome_id", sort=False)
        for genome_id, group in grouped:
            qvals = pd.to_numeric(group["qvalue"], errors="coerce")
            ranks = pd.to_numeric(group["presence_rank"], errors="coerce")
            unique_counts = pd.to_numeric(group["num_peptides_unique"], errors="coerce").fillna(0).astype(int)
            matched_counts = pd.to_numeric(group["num_peptides_matched"], errors="coerce").fillna(0).astype(int)
            n_units = int(len(group))
            summary_rows.append(
                {
                    "genome_id": genome_id,
                    "Lineage": lineage_map.get(str(genome_id), pd.NA),
                    "n_units_tested": n_units,
                    "n_units_matched_ge_1": int((matched_counts >= 1).sum()),
                    "n_units_unique_ge_3": int((unique_counts >= 3).sum()),
                    "n_units_q_le_0_05": int((qvals <= 0.05).sum()),
                    "n_units_q_le_0_01": int((qvals <= 0.01).sum()),
                    "best_qvalue": float(qvals.min()) if len(qvals) else 1.0,
                    "median_qvalue": float(qvals.median()) if len(qvals) else 1.0,
                    "best_presence_rank": int(ranks.min()) if len(ranks.dropna()) else 0,
                    "median_presence_rank": float(ranks.median()) if len(ranks.dropna()) else 0.0,
                    "total_unique_peptides_across_units": int(unique_counts.sum()),
                    "max_unique_peptides_in_one_unit": int(unique_counts.max()) if len(unique_counts) else 0,
                    "total_matched_peptides_across_units": int(matched_counts.sum()),
                    "max_matched_peptides_in_one_unit": int(matched_counts.max()) if len(matched_counts) else 0,
                    "fraction_units_q_le_0_05": float((qvals <= 0.05).sum()) / float(max(n_units, 1)),
                    "fraction_units_q_le_0_01": float((qvals <= 0.01).sum()) / float(max(n_units, 1)),
                }
            )
        cohort_summary_df = pd.DataFrame(summary_rows)
        if "Lineage" in cohort_summary_df.columns and cohort_summary_df["Lineage"].isna().all():
            cohort_summary_df = cohort_summary_df.drop(columns=["Lineage"])
        if not cohort_summary_df.empty:
            cohort_summary_df = cohort_summary_df.sort_values(
                ["n_units_q_le_0_05", "best_qvalue", "total_unique_peptides_across_units", "genome_id"],
                ascending=[False, True, False, True],
                kind="mergesort",
            ).reset_index(drop=True)

        self.run_stats["unit_aware_output_rows"] = int(len(unit_level_df))
        self.run_stats["unit_aware_cohort_summary_rows"] = int(len(cohort_summary_df))
        self.run_stats["unit_aware_unique_pvalue_mode"] = mode
        self.run_stats["unit_aware_presence_rule"] = "union"
        self.run_stats["unit_aware_shared_mode"] = "per-unit"
        return unit_level_df, cohort_summary_df

    def _prepare_unit_aware_output_tables(
        self,
        unit_level_df: pd.DataFrame,
        cohort_summary_df: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """Validate and prepare primary and derived unit-aware output tables."""
        def _ordered_columns(df: pd.DataFrame, preferred: List[str]) -> pd.DataFrame:
            ordered = [col for col in preferred if col in df.columns]
            ordered.extend([col for col in df.columns if col not in ordered])
            return df.loc[:, ordered].copy()

        def _coerce_bool_like(series: pd.Series, column_name: str) -> pd.Series:
            if pd.api.types.is_bool_dtype(series):
                return series.fillna(False).astype(bool)
            normalized = series.map(
                lambda value: (
                    pd.NA
                    if pd.isna(value)
                    else str(value).strip().lower()
                )
            )
            truthy = {"true", "1", "1.0", "yes", "y", "t"}
            falsy = {"false", "0", "0.0", "no", "n", "f"}
            valid = normalized.isna() | normalized.isin(truthy.union(falsy))
            if not bool(valid.all()):
                bad_values = sorted({str(v) for v in series.loc[~valid].head(5).tolist()})
                raise RuntimeError(f"Unit-aware sanity check failed: {column_name} is not boolean-like: {bad_values}")
            return normalized.isin(truthy).fillna(False).astype(bool)

        def _require_columns(df: pd.DataFrame, required: List[str], table_name: str) -> None:
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise RuntimeError(f"Unit-aware sanity check failed: {table_name} is missing required columns: {missing}")

        def _sort_existing(df: pd.DataFrame, columns: List[str], ascending: List[bool]) -> pd.DataFrame:
            sort_cols = [col for col in columns if col in df.columns]
            if not sort_cols or df.empty:
                return df.reset_index(drop=True)
            sort_asc = [ascending[columns.index(col)] for col in sort_cols]
            return df.sort_values(sort_cols, ascending=sort_asc, kind="mergesort").reset_index(drop=True)

        def _build_unit_call_counts(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return pd.DataFrame(
                    columns=[
                        "analysis_unit_id",
                        "n_samples_in_unit",
                        "n_genomes_q_le_0_01",
                        "n_genomes_q_le_0_05",
                        "n_genomes_matched_ge_1",
                        "n_genomes_unique_ge_3",
                        "median_qvalue",
                        "best_qvalue",
                    ]
                )
            rows = []
            for unit_id, group in df.groupby("analysis_unit_id", sort=False):
                qvals = pd.to_numeric(group["qvalue"], errors="coerce")
                matched = pd.to_numeric(group["num_peptides_matched"], errors="coerce").fillna(0)
                unique = pd.to_numeric(group["num_peptides_unique"], errors="coerce").fillna(0)
                n_samples = (
                    pd.to_numeric(group.get("n_samples_in_unit", pd.Series(dtype=float)), errors="coerce")
                    .dropna()
                    .astype(int)
                )
                rows.append(
                    {
                        "analysis_unit_id": unit_id,
                        "n_samples_in_unit": int(n_samples.max()) if len(n_samples) else 0,
                        "n_genomes_q_le_0_01": int(group["pass_q_0_01"].sum()),
                        "n_genomes_q_le_0_05": int(group["pass_q_0_05"].sum()),
                        "n_genomes_matched_ge_1": int((matched >= 1).sum()),
                        "n_genomes_unique_ge_3": int((unique >= 3).sum()),
                        "median_qvalue": float(qvals.median()) if len(qvals.dropna()) else 1.0,
                        "best_qvalue": float(qvals.min()) if len(qvals.dropna()) else 1.0,
                    }
                )
            return (
                pd.DataFrame(rows)
                .sort_values("analysis_unit_id", key=lambda s: s.astype(str), kind="mergesort")
                .reset_index(drop=True)
            )

        def _build_presence_matrix(
            df: pd.DataFrame,
            genome_order_df: pd.DataFrame,
            value_col: str,
            unit_ids: List[str],
            fill_value: object,
        ) -> pd.DataFrame:
            lineage_cols = ["Lineage"] if "Lineage" in genome_order_df.columns else []
            base = genome_order_df[["genome_id", *lineage_cols]].drop_duplicates("genome_id").copy()
            if base.empty:
                return pd.DataFrame(columns=["genome_id", *lineage_cols, *unit_ids])
            pivot_source = df[["genome_id", "analysis_unit_id", value_col]].copy()
            pivot_source["analysis_unit_id"] = pivot_source["analysis_unit_id"].astype(str)
            pivot = pivot_source.pivot_table(
                index="genome_id",
                columns="analysis_unit_id",
                values=value_col,
                aggfunc="max",
            )
            pivot = pivot.reindex(index=base["genome_id"].astype(str), columns=unit_ids)
            pivot = pivot.fillna(fill_value)
            matrix = base.reset_index(drop=True).join(pivot.reset_index(drop=True))
            return matrix

        _require_columns(
            unit_level_df,
            [
                "analysis_unit_id",
                "genome_id",
                "qvalue",
                "pass_q_0_01",
                "pass_q_0_05",
                "num_peptides_matched",
                "num_peptides_unique",
            ],
            "unit_genome_presence",
        )
        _require_columns(
            cohort_summary_df,
            ["genome_id", "n_units_q_le_0_01", "n_units_q_le_0_05"],
            "cohort_genome_summary",
        )

        unit_level_df = unit_level_df.copy()
        cohort_summary_df = cohort_summary_df.copy()
        matched = pd.to_numeric(unit_level_df["num_peptides_matched"], errors="coerce")
        unique = pd.to_numeric(unit_level_df["num_peptides_unique"], errors="coerce")
        if bool((matched < 0).fillna(False).any()) or bool((unique < 0).fillna(False).any()):
            raise RuntimeError("Unit-aware sanity check failed: negative peptide counts detected.")
        if bool((matched < unique).fillna(False).any()):
            raise RuntimeError("Unit-aware sanity check failed: num_peptides_matched is smaller than num_peptides_unique.")
        unit_level_df["pass_q_0_01"] = _coerce_bool_like(unit_level_df["pass_q_0_01"], "pass_q_0_01")
        unit_level_df["pass_q_0_05"] = _coerce_bool_like(unit_level_df["pass_q_0_05"], "pass_q_0_05")
        q001_units = pd.to_numeric(cohort_summary_df["n_units_q_le_0_01"], errors="coerce").fillna(0).astype(int)
        q005_units = pd.to_numeric(cohort_summary_df["n_units_q_le_0_05"], errors="coerce").fillna(0).astype(int)
        if bool((q001_units > q005_units).any()):
            raise RuntimeError("Unit-aware sanity check failed: n_units_q_le_0_01 exceeds n_units_q_le_0_05.")

        unit_level_columns = [
            "analysis_unit_id",
            "genome_id",
            "Lineage",
            "presence_rank",
            "qvalue",
            "pvalue",
            "pass_q_0_01",
            "pass_q_0_05",
            "num_peptides_unique",
            "unique_effective_count",
            "num_peptides_matched",
            "matched_peptide_count_shared",
            "weighted_evidence_shared",
            "effective_peptide_count_shared",
            "expected_unique_null",
            "unique_depth_fold",
            "theoretical_unique_peptides",
            "observed_unique_peptide_pool_size",
            "pvalue_unique",
            "pvalue_unique_depth",
            "pvalue_shared",
            "presence_score",
            "n_samples_in_unit",
            "unit_presence_rule",
            "unit_shared_mode",
        ]
        cohort_columns = [
            "genome_id",
            "Lineage",
            "n_units_tested",
            "n_units_q_le_0_05",
            "fraction_units_q_le_0_05",
            "n_units_q_le_0_01",
            "fraction_units_q_le_0_01",
            "n_units_matched_ge_1",
            "n_units_unique_ge_3",
            "best_qvalue",
            "median_qvalue",
            "best_presence_rank",
            "median_presence_rank",
            "max_unique_peptides_in_one_unit",
            "total_unique_peptides_across_units",
            "max_matched_peptides_in_one_unit",
            "total_matched_peptides_across_units",
        ]
        significant_columns = [
            "analysis_unit_id",
            "genome_id",
            "Lineage",
            "presence_rank",
            "qvalue",
            "pvalue",
            "num_peptides_unique",
            "unique_effective_count",
            "num_peptides_matched",
            "expected_unique_null",
            "unique_depth_fold",
            "theoretical_unique_peptides",
            "n_samples_in_unit",
        ]
        union_columns = [
            "genome_id",
            "Lineage",
            "n_units_tested",
            "n_units_q_le_0_05",
            "fraction_units_q_le_0_05",
            "n_units_q_le_0_01",
            "fraction_units_q_le_0_01",
            "best_qvalue",
            "median_qvalue",
            "best_presence_rank",
            "median_presence_rank",
            "max_unique_peptides_in_one_unit",
            "total_unique_peptides_across_units",
            "max_matched_peptides_in_one_unit",
            "total_matched_peptides_across_units",
        ]

        unit_level_out = _ordered_columns(unit_level_df, unit_level_columns)
        cohort_summary_out = _ordered_columns(cohort_summary_df, cohort_columns)
        if not self._export_unit_derived_tables:
            return {
                "unit_genome_presence": unit_level_out,
                "cohort_genome_summary": cohort_summary_out,
            }
        unit_call_counts = _build_unit_call_counts(unit_level_df)
        unit_q001 = _sort_existing(
            _ordered_columns(unit_level_df.loc[unit_level_df["pass_q_0_01"]], significant_columns),
            ["analysis_unit_id", "presence_rank", "qvalue"],
            [True, True, True],
        )
        unit_q005 = _sort_existing(
            _ordered_columns(unit_level_df.loc[unit_level_df["pass_q_0_05"]], significant_columns),
            ["analysis_unit_id", "presence_rank", "qvalue"],
            [True, True, True],
        )
        genome_union_q001 = _sort_existing(
            _ordered_columns(cohort_summary_df.loc[q001_units >= 1], union_columns),
            ["n_units_q_le_0_01", "best_qvalue", "total_unique_peptides_across_units", "genome_id"],
            [False, True, False, True],
        )
        genome_union_q005 = _sort_existing(
            _ordered_columns(cohort_summary_df.loc[q005_units >= 1], union_columns),
            ["n_units_q_le_0_05", "best_qvalue", "total_unique_peptides_across_units", "genome_id"],
            [False, True, False, True],
        )
        if not set(genome_union_q001["genome_id"].astype(str)).issubset(set(genome_union_q005["genome_id"].astype(str))):
            raise RuntimeError("Unit-aware sanity check failed: genome_union_q001 is not a subset of genome_union_q005.")

        matrix_df = unit_level_df.copy()
        matrix_df["analysis_unit_id"] = matrix_df["analysis_unit_id"].astype(str)
        matrix_df["genome_id"] = matrix_df["genome_id"].astype(str)
        unit_ids = sorted(matrix_df["analysis_unit_id"].dropna().unique().tolist())
        matrix_q001 = _build_presence_matrix(matrix_df, genome_union_q001, "pass_q_0_01", unit_ids, 0)
        matrix_q005 = _build_presence_matrix(matrix_df, genome_union_q005, "pass_q_0_05", unit_ids, 0)
        matrix_qvalue = _build_presence_matrix(matrix_df, cohort_summary_out, "qvalue", unit_ids, np.nan)
        for matrix in (matrix_q001, matrix_q005):
            for unit_id in unit_ids:
                if unit_id in matrix.columns:
                    matrix[unit_id] = pd.to_numeric(matrix[unit_id], errors="coerce").fillna(0).astype(int)

        return {
            "unit_genome_presence": unit_level_out,
            "cohort_genome_summary": cohort_summary_out,
            "unit_call_counts": unit_call_counts,
            "unit_q001_genomes": unit_q001,
            "unit_q005_genomes": unit_q005,
            "genome_union_q001": genome_union_q001,
            "genome_union_q005": genome_union_q005,
            "genome_by_unit_q001_matrix": matrix_q001,
            "genome_by_unit_q005_matrix": matrix_q005,
            "genome_by_unit_qvalue_matrix": matrix_qvalue,
        }

    def _export_unit_aware_primary_outputs(
        self,
        out_dir: str,
        stem: str,
        requested_output_path: str,
        pooled_df: pd.DataFrame,
        tables: Dict[str, pd.DataFrame],
        mapping_df: pd.DataFrame,
        export_pooled_result: bool,
    ) -> None:
        """Write the primary unit-aware outputs."""
        os.makedirs(out_dir or ".", exist_ok=True)
        artifact_dir = os.path.join(out_dir, f"{stem}_artifacts")
        unit_level_path = str(requested_output_path)
        cohort_path = os.path.join(out_dir, f"{stem}_cohort_genome_summary.tsv")
        mapping_path = os.path.join(out_dir, f"{stem}_sample_unit_mapping.tsv")
        pooled_path = os.path.join(artifact_dir, "pooled_genome_presence.tsv")

        self.unit_aware_output_paths = {
            "unit_genome_presence": unit_level_path,
            "cohort_genome_summary": cohort_path,
            "sample_unit_mapping": mapping_path,
        }
        unit_level_out = tables["unit_genome_presence"]
        cohort_summary_out = tables["cohort_genome_summary"]
        self.unit_aware_cohort_summary_df = cohort_summary_out.copy()
        self._last_unit_genome_presence_df = unit_level_out.copy()

        unit_level_out.to_csv(unit_level_path, sep="\t", index=False)
        cohort_summary_out.to_csv(cohort_path, sep="\t", index=False)
        mapping_df.to_csv(mapping_path, sep="\t", index=False)

        if export_pooled_result:
            os.makedirs(artifact_dir, exist_ok=True)
            pooled_df.to_csv(pooled_path, sep="\t", index=False)
            self.unit_aware_output_paths["pooled_genome_presence"] = pooled_path
            self.logger.info(f"Saved pooled peptide-set genome presence table: {pooled_path}")

        self.run_stats["unit_aware_derived_tables_exported"] = False
        self.run_stats["unit_aware_unit_call_count_rows"] = 0
        self.run_stats["unit_aware_genome_union_q001_rows"] = 0
        self.run_stats["unit_aware_genome_union_q005_rows"] = 0
        self.run_stats["unit_aware_output_paths"] = dict(self.unit_aware_output_paths)
        self.logger.info(f"Saved unit-aware genome presence table: {unit_level_path}")
        self.logger.info(f"Saved unit-aware cohort summary: {cohort_path}")
        self.logger.info(f"Saved sample-unit mapping: {mapping_path}")

    def _export_unit_aware_derived_outputs(
        self,
        out_dir: str,
        stem: str,
        tables: Dict[str, pd.DataFrame],
    ) -> None:
        """Write optional derived unit-aware tables under the artifacts directory."""
        derived_dir = os.path.join(out_dir, f"{stem}_artifacts", "unit_aware")
        os.makedirs(derived_dir, exist_ok=True)
        derived_paths = {
            "unit_call_counts": os.path.join(derived_dir, "unit_call_counts.tsv"),
            "unit_q001_genomes": os.path.join(derived_dir, "unit_q001_genomes.tsv"),
            "unit_q005_genomes": os.path.join(derived_dir, "unit_q005_genomes.tsv"),
            "genome_union_q001": os.path.join(derived_dir, "genome_union_q001.tsv"),
            "genome_union_q005": os.path.join(derived_dir, "genome_union_q005.tsv"),
            "genome_by_unit_q001_matrix": os.path.join(derived_dir, "genome_by_unit_q001_matrix.tsv"),
            "genome_by_unit_q005_matrix": os.path.join(derived_dir, "genome_by_unit_q005_matrix.tsv"),
            "genome_by_unit_qvalue_matrix": os.path.join(derived_dir, "genome_by_unit_qvalue_matrix.tsv"),
        }
        missing = [key for key in derived_paths if key not in tables]
        if missing:
            raise RuntimeError(
                "Derived unit-aware tables were requested but not prepared: " + ", ".join(missing)
            )
        for key, path in derived_paths.items():
            tables[key].to_csv(path, sep="\t", index=False)
            self.unit_aware_output_paths[key] = path
        self.unit_aware_output_paths["derived_unit_aware_tables_dir"] = derived_dir
        self.run_stats["unit_aware_output_paths"] = dict(self.unit_aware_output_paths)
        self.run_stats["unit_aware_derived_tables_exported"] = True
        self.run_stats["unit_aware_unit_call_count_rows"] = int(len(tables["unit_call_counts"]))
        self.run_stats["unit_aware_genome_union_q001_rows"] = int(len(tables["genome_union_q001"]))
        self.run_stats["unit_aware_genome_union_q005_rows"] = int(len(tables["genome_union_q005"]))
        self.logger.info(f"Saved derived unit-aware tables: {derived_dir}")

    def analyze_genomes(
        self,
        genome_digest_dirs: Union[str, List[str]],
        output_tsv_path: Optional[str] = None,
        genome_lineage_table_path: Optional[str] = None,
        genome_lineage_genome_id_col: Optional[str] = None,
        genome_lineage_lineage_col: Optional[str] = None,
        genome_list: Optional[List[str]] = None,
        exclude_genome_ids: Optional[List[str]] = None,
        test_genomes_num: Optional[int] = None,
        all_matched_peptides: Optional[List[Tuple[str, Set[str], int]]] = None,
        save_matched_peptides_cache: bool = True,
        matched_peptides_cache_path: Optional[str] = None,
        compute_coverage: bool = True,
        export_temp: bool = True,
        export_peptide_contrib_topN: int = 0,
        use_cache_if_exists: bool = True,
        unique_pvalue_mode: str = DEFAULT_UNIQUE_PVALUE_MODE,
        unique_peptide_error_source: str = DEFAULT_UNIQUE_PEPTIDE_ERROR_SOURCE,
        min_unique_for_unique_pvalue: int = 3,
        unique_count_power: float = DEFAULT_UNIQUE_COUNT_POWER,
        unique_count_cap: Optional[float] = None,
        theoretical_opportunity_cache_path: Optional[str] = None,
        rebuild_theoretical_opportunity_cache: bool = False,
        num_workers_for_theoretical_opportunity: Optional[int] = None,
        return_full_table: bool = False,
        unit_aware: bool = False,
        export_unit_derived_tables: bool = False,
    ) -> pd.DataFrame:
        """End-to-end analysis producing a genome-level q-value (q_presence)."""
        mode = _normalize_unique_pvalue_mode(unique_pvalue_mode)
        unique_peptide_error_source = _normalize_unique_peptide_error_source(unique_peptide_error_source)
        if int(min_unique_for_unique_pvalue) < 0:
            raise ValueError("min_unique_for_unique_pvalue must be >= 0.")
        unique_count_power = float(unique_count_power)
        if not np.isfinite(unique_count_power) or not (0 < unique_count_power <= 1):
            raise ValueError("unique_count_power must be in the interval (0, 1].")
        if unique_count_cap is not None:
            cap_value = float(unique_count_cap)
            if not np.isfinite(cap_value) or cap_value <= 0:
                raise ValueError("unique_count_cap must be positive or None.")
        if unit_aware:
            if not self.unit_aware_enabled:
                raise ValueError("unit_aware=True requires read_unit_aware_peptide_file() before analyze_genomes().")
            self.unit_presence_rule = "union"
            self.unit_shared_mode = "per-unit"
        self._export_unit_derived_tables = bool(export_unit_derived_tables)
        self._last_unit_genome_presence_df = None
        self.unique_pvalue_mode = mode
        self.unique_peptide_error_source = unique_peptide_error_source
        self.min_unique_for_unique_pvalue = int(min_unique_for_unique_pvalue)
        self.unique_count_power = float(unique_count_power)
        self.unique_count_cap = unique_count_cap
        theoretical_opportunity_workers = (
            self.num_workers
            if num_workers_for_theoretical_opportunity is None
            else int(num_workers_for_theoretical_opportunity)
        )
        self.run_stats["theoretical_opportunity_requested_num_workers"] = (
            None if num_workers_for_theoretical_opportunity is None else int(num_workers_for_theoretical_opportunity)
        )
        self.run_stats["theoretical_opportunity_effective_num_workers"] = int(
            _resolve_worker_count(theoretical_opportunity_workers, logger=self.logger)
        )

        if output_tsv_path is None:
            out_dir = self.peptide_table_dir if self.peptide_table_dir else os.getcwd()
            output_tsv_path = os.path.join(out_dir, "genome_presence.tsv")
            self.logger.info(f"Output file not specified. Using: {output_tsv_path}")
        else:
            out_dir = os.path.dirname(output_tsv_path) or "."
            os.makedirs(out_dir, exist_ok=True)

        stem = Path(output_tsv_path).stem
        default_cache_dir = os.path.join(out_dir, f"{stem}_artifacts") if export_temp else out_dir
        default_cache_pkl_path = os.path.join(default_cache_dir, "matched_peptides.pkl")
        default_theoretical_cache_path = os.path.join(default_cache_dir, "theoretical_opportunity_cache.pkl")
        theoretical_cache_path = str(theoretical_opportunity_cache_path) if theoretical_opportunity_cache_path else default_theoretical_cache_path

        if genome_lineage_table_path:
            if not genome_lineage_genome_id_col or not genome_lineage_lineage_col:
                raise ValueError(
                    "When genome_lineage_table_path is provided, you must also provide "
                    "genome_lineage_genome_id_col and genome_lineage_lineage_col."
                )
            self.genome_lineage_df = self._read_genome_lineage_table(
                genome_lineage_table_path=genome_lineage_table_path,
                genome_lineage_genome_id_col=genome_lineage_genome_id_col,
                genome_lineage_lineage_col=genome_lineage_lineage_col,
            )
            self.run_stats["genome_lineage_table_path"] = str(genome_lineage_table_path)
            self.run_stats["genome_lineage_genome_id_col"] = str(genome_lineage_genome_id_col)
            self.run_stats["genome_lineage_lineage_col"] = str(genome_lineage_lineage_col)
            self.run_stats["genome_lineage_annotations_loaded"] = int(len(self.genome_lineage_df))
        else:
            self.genome_lineage_df = None
            self.run_stats["genome_lineage_table_path"] = None
            self.run_stats["genome_lineage_genome_id_col"] = None
            self.run_stats["genome_lineage_lineage_col"] = None
            self.run_stats["genome_lineage_annotations_loaded"] = 0

        t_all0 = time.time()
        self.timing_stats = {}
        self.unit_aware_output_paths = {}
        self.unit_aware_cohort_summary_df = None

        # Normalize cache path (if provided); otherwise use the default.
        cache_pkl_path: Optional[str] = None
        if matched_peptides_cache_path:
            cache_path = str(matched_peptides_cache_path)
            cache_pkl_path = cache_path if cache_path.lower().endswith(".pkl") else f"{cache_path}.pkl"
        else:
            cache_pkl_path = default_cache_pkl_path

        self.run_stats["matched_peptides_cache_path"] = str(cache_pkl_path) if cache_pkl_path else None
        self.run_stats["matched_peptides_cache_is_default"] = bool(not matched_peptides_cache_path)

        # Prefer using existing matched-peptides cache if allowed and available.
        if use_cache_if_exists and all_matched_peptides is None and cache_pkl_path and os.path.exists(cache_pkl_path):
            self.logger.info(f"Loading matched peptides cache: {cache_pkl_path}")
            try:
                with open(cache_pkl_path, "rb") as f:
                    cached = pickle.load(f)

                # Minimal validation + normalization:
                # Expect List[Tuple[str, Iterable[str], int]]
                normalized: List[Tuple[str, Set[str], int]] = []
                if not isinstance(cached, list):
                    raise TypeError(f"Cache must be a list, got {type(cached)}")

                for i, item in enumerate(cached):
                    if not isinstance(item, (tuple, list)) or len(item) != 3:
                        raise TypeError(f"Cache item #{i} must be a 3-tuple, got {type(item)} len={len(item) if hasattr(item, '__len__') else 'NA'}")
                    genome_id, matched_peps, total_cnt = item
                    if not isinstance(genome_id, str) or not genome_id.strip():
                        raise TypeError(f"Cache item #{i} genome_id must be non-empty str, got {type(genome_id)}")

                    # matched_peps may be set/list/tuple/np array; normalize to Set[str]
                    if matched_peps is None:
                        matched_set: Set[str] = set()
                    else:
                        try:
                            matched_set = {str(x) for x in matched_peps if x is not None and str(x) != ""}
                        except TypeError as e:
                            raise TypeError(f"Cache item #{i} matched_peptides is not iterable: {type(matched_peps)}") from e

                    try:
                        total_int = int(total_cnt)
                    except Exception as e:
                        raise TypeError(f"Cache item #{i} total_cnt must be int-like, got {type(total_cnt)}") from e
                    if total_int < 0:
                        raise ValueError(f"Cache item #{i} total_cnt must be >=0, got {total_int}")

                    normalized.append((genome_id.strip(), matched_set, total_int))

                all_matched_peptides = normalized
                self.logger.info(f"Loaded matched peptides cache OK: {len(all_matched_peptides)} genomes")
            except Exception as e:
                self.logger.warning(
                    f"Failed to load/validate matched peptides cache ({cache_pkl_path}); recomputing. Error: {e}"
                )
                all_matched_peptides = None
        elif not use_cache_if_exists and all_matched_peptides is None and cache_pkl_path and os.path.exists(cache_pkl_path):
            self.logger.info(
                f"Cache exists but use_cache_if_exists=False; recomputing matched peptides: {cache_pkl_path}"
            )

        t_scan0 = time.time()

        if all_matched_peptides is None:
            folders = [genome_digest_dirs] if isinstance(genome_digest_dirs, str) else list(genome_digest_dirs)
            valid_folders = [f for f in folders if f and os.path.exists(f)]
            if not valid_folders:
                raise ValueError("No valid genome folders found.")

            all_genome_files: List[Path] = []
            for folder in valid_folders:
                files = list(Path(folder).glob("*.tsv"))
                self.logger.info(f"Found {len(files)} genome peptide files in: {folder}")
                all_genome_files.extend(files)

            if not all_genome_files:
                raise ValueError("No genome peptide TSV files found.")

            if genome_list:
                genome_set = {g.strip() for g in genome_list if isinstance(g, str) and g.strip()}
                selected_genome_files = [p for p in all_genome_files if p.stem in genome_set]
                missing_genomes = sorted(genome_set.difference({p.stem for p in selected_genome_files}))
                self.logger.info(
                    f"Selected {len(selected_genome_files)} genomes from include list ({len(genome_set)} requested)."
                )
                if missing_genomes:
                    preview = ", ".join(missing_genomes[:10])
                    suffix = " ..." if len(missing_genomes) > 10 else ""
                    self.logger.warning(
                        f"{len(missing_genomes)} requested genome IDs were not found in the digest folders: "
                        f"{preview}{suffix}"
                    )
            elif test_genomes_num and test_genomes_num < len(all_genome_files):
                selected_genome_files = random.sample(all_genome_files, int(test_genomes_num))
            else:
                selected_genome_files = all_genome_files

            if exclude_genome_ids:
                ex = {g.strip() for g in exclude_genome_ids if isinstance(g, str) and g.strip()}
                before = len(selected_genome_files)
                selected_genome_files = [p for p in selected_genome_files if p.stem not in ex]
                self.logger.info(f"Excluded {before - len(selected_genome_files)} genomes by exclude list.")

            if not selected_genome_files:
                if genome_list:
                    raise ValueError(
                        "No genome peptide TSV files matched the Only Run Genome IDs list. "
                        "Please check that the IDs match the TSV filenames."
                    )
                raise ValueError("No genome peptide TSV files remained after filtering.")

            self.logger.info(f"Processing {len(selected_genome_files)} genome files with num_workers={self.num_workers} ...")

            batches = np.array_split(
                np.asarray(selected_genome_files, dtype=object),
                max(1, self.num_workers * 4),
            )
            # Submit jobs to a ProcessPoolExecutor. Use explicit executor and
            # explicit shutdown in a finally block, and clear references to
            # futures/executor to avoid weakref callbacks during interpreter
            # shutdown which can raise harmless-but-noisy exceptions.
            raw_worker_results: List[Tuple[str, Set[str], int, Optional[str]]] = []
            futures = []
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.num_workers,
                initializer=_init_genome_batch_worker,
                initargs=(frozenset(self.peptide_score.keys()),),
            )
            try:
                for b in batches:
                    if len(b) == 0:
                        continue
                    futures.append(executor.submit(_process_genome_batch_worker, list(b)))

                for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Scanning genomes"):
                    raw_worker_results.extend(fut.result())
            finally:
                try:
                    executor.shutdown(wait=True)
                except Exception:
                    # Best-effort shutdown; swallow exceptions during interpreter
                    # teardown to avoid noisy output.
                    pass
                # remove references that may keep callbacks alive during module
                # teardown
                try:
                    del futures
                    del executor
                except Exception:
                    pass

            read_errors: List[Tuple[str, str]] = []
            all_matched_peptides = []
            for genome_id, matched_peptides, total_cnt, err in raw_worker_results:
                if err:
                    read_errors.append((genome_id, err))
                    continue
                all_matched_peptides.append((genome_id, matched_peptides, total_cnt))

            self.run_stats["genome_scan_read_error_count"] = int(len(read_errors))
            if read_errors:
                preview = [{"genome_id": gid, "error": msg} for gid, msg in read_errors[:20]]
                self.run_stats["genome_scan_read_error_preview"] = preview
                examples = "; ".join(f"{gid}: {msg}" for gid, msg in read_errors[:5])
                raise RuntimeError(
                    f"Failed to read {len(read_errors)} genome peptide files. "
                    f"Examples: {examples}"
                )

            if save_matched_peptides_cache:
                pkl_path = cache_pkl_path or os.path.join(out_dir, "matched_peptides.pkl")
                os.makedirs(os.path.dirname(pkl_path) or ".", exist_ok=True)
                with open(pkl_path, "wb") as f:
                    pickle.dump(all_matched_peptides, f)
                self.logger.info(f"Saved matched peptides cache: {pkl_path}")

        self.timing_stats["scan_genomes"] = float(time.time() - t_scan0)

        selected_genome_set = {
            g.strip() for g in (genome_list or []) if isinstance(g, str) and g.strip()
        }
        excluded_genome_set = {
            g.strip() for g in (exclude_genome_ids or []) if isinstance(g, str) and g.strip()
        }
        if selected_genome_set or excluded_genome_set:
            before = len(all_matched_peptides)
            available_genome_ids = {genome_id for genome_id, _, _ in all_matched_peptides}
            filtered_matched_peptides: List[Tuple[str, Set[str], int]] = []
            for genome_id, matched_peptides, total_cnt in all_matched_peptides:
                if selected_genome_set and genome_id not in selected_genome_set:
                    continue
                if excluded_genome_set and genome_id in excluded_genome_set:
                    continue
                filtered_matched_peptides.append((genome_id, matched_peptides, total_cnt))
            all_matched_peptides = filtered_matched_peptides
            self.logger.info(f"Retained {len(all_matched_peptides)} genomes after applying cache-safe filters.")
            if not all_matched_peptides:
                if selected_genome_set:
                    raise ValueError(
                        "No genomes remained after applying the Only Run Genome IDs list to the matched-peptide data."
                    )
                raise ValueError("No genomes remained after applying genome filters.")
            if selected_genome_set:
                missing_from_data = sorted(selected_genome_set.difference(available_genome_ids))
                if missing_from_data:
                    preview = ", ".join(missing_from_data[:10])
                    suffix = " ..." if len(missing_from_data) > 10 else ""
                    self.logger.warning(
                        f"{len(missing_from_data)} Only Run Genome IDs were not present in the matched-peptide data: "
                        f"{preview}{suffix}"
                    )
            self.run_stats["genome_filter_input_count"] = int(before)
            self.run_stats["genome_filter_output_count"] = int(len(all_matched_peptides))

        self.genome_matched_peptides = {}
        self.genome_total_theoretical_peptides = {}
        obs_set = set(self.peptide_score.keys())

        for genome_id, matched_peptides, total_cnt in all_matched_peptides:
            matched_peptides = set(matched_peptides).intersection(obs_set)
            self.genome_matched_peptides.setdefault(genome_id, set()).update(matched_peptides)
            prev = self.genome_total_theoretical_peptides.get(genome_id, 0)
            self.genome_total_theoretical_peptides[genome_id] = max(prev, int(total_cnt))
        self.total_theoretical_peptides_all_genomes = int(sum(self.genome_total_theoretical_peptides.values()))
        self.run_stats["total_theoretical_peptides_all_genomes"] = int(self.total_theoretical_peptides_all_genomes)

        opportunity_rebuilt = False
        if mode == "hypergeometric-opportunity":
            matched_genome_ids = set(self.genome_matched_peptides.keys())
            folders = [genome_digest_dirs] if isinstance(genome_digest_dirs, str) else list(genome_digest_dirs)
            genome_files_by_id: Dict[str, Path] = {}
            for folder in [f for f in folders if f and os.path.exists(f)]:
                for path in Path(folder).glob("*.tsv"):
                    if path.stem in matched_genome_ids:
                        genome_files_by_id.setdefault(path.stem, path)
            genome_files_for_opportunity = [genome_files_by_id[gid] for gid in sorted(genome_files_by_id)]
            if len(genome_files_for_opportunity) != len(matched_genome_ids):
                missing = sorted(matched_genome_ids.difference({p.stem for p in genome_files_for_opportunity}))
                preview = ", ".join(missing[:10])
                suffix = " ..." if len(missing) > 10 else ""
                raise ValueError(
                    "Hypergeometric-opportunity unique p-values require digest TSV files for all selected genomes. "
                    f"Missing digest files for {len(missing)} genomes: {preview}{suffix}"
                )
            opportunity, opportunity_rebuilt = self._load_or_build_theoretical_opportunity(
                genome_digest_files=genome_files_for_opportunity,
                cache_path=theoretical_cache_path,
                rebuild_cache=bool(rebuild_theoretical_opportunity_cache),
                num_workers_for_theoretical_opportunity=theoretical_opportunity_workers,
            )
            self._apply_theoretical_opportunity(opportunity)
            vals = pd.Series(list(self.genome_theoretical_unique_peptides.values()), dtype=float)
            self.run_stats["unique_depth_null_model"] = "hypergeometric"
            self.run_stats["theoretical_peptide_universe_size"] = int(self.theoretical_peptide_universe_size)
            self.run_stats["total_theoretical_unique_peptides_all_genomes"] = int(self.total_theoretical_unique_peptides_all_genomes)
            self.run_stats["theoretical_opportunity_cache_path"] = str(theoretical_cache_path) if theoretical_cache_path else None
            self.run_stats["theoretical_opportunity_cache_rebuilt"] = bool(opportunity_rebuilt)
            self.run_stats["genome_theoretical_unique_peptides_quantiles"] = (
                {
                    "q0": float(vals.quantile(0.0)),
                    "q25": float(vals.quantile(0.25)),
                    "q50": float(vals.quantile(0.5)),
                    "q75": float(vals.quantile(0.75)),
                    "q100": float(vals.quantile(1.0)),
                }
                if len(vals) > 0
                else {}
            )
        else:
            self.run_stats["unique_depth_null_model"] = ""
            self.run_stats["theoretical_opportunity_cache_path"] = None
            self.run_stats["theoretical_opportunity_cache_rebuilt"] = False

        t_deg0 = time.time()
        peptide_deg, genome_unique_counts, peptide_unique_owner = self._calculate_peptide_degeneracy_and_unique_counts(
            all_matched_peptides
        )
        self.timing_stats["compute_degeneracy"] = float(time.time() - t_deg0)
        self.peptide_degeneracy = peptide_deg
        self.observed_matchable_peptides = int(len(peptide_deg))
        self.observed_unique_peptide_pool_size = int(sum(int(v) for v in genome_unique_counts.values()))
        self.run_stats["observed_matchable_peptides"] = int(self.observed_matchable_peptides)
        self.run_stats["observed_unique_peptide_pool_size"] = int(self.observed_unique_peptide_pool_size)
        if mode == "hypergeometric-opportunity":
            self.run_stats.setdefault("theoretical_peptide_universe_size", int(self.theoretical_peptide_universe_size))

        genome_data_list = [
            (
                genome_id,
                matched_peptides,
                self.genome_total_theoretical_peptides.get(genome_id, 0),
                genome_unique_counts.get(genome_id, 0),
            )
            for genome_id, matched_peptides in self.genome_matched_peptides.items()
        ]

        self.logger.info(f"Computing metrics for {len(genome_data_list)} genomes ...")
        t_metrics0 = time.time()
        df_metrics = self._calculate_genome_metrics(genome_data_list, peptide_deg)
        self.timing_stats["compute_metrics"] = float(time.time() - t_metrics0)

        self.logger.info("Ranking genomes ...")
        t_score0 = time.time()
        df_scored = self._rank_genomes(df_metrics)
        self.timing_stats["rank_genomes"] = float(time.time() - t_score0)
        df_scored["evidence_rank"] = np.arange(1, len(df_scored) + 1, dtype=int)

        t_knock0 = time.time()
        self.logger.info("Computing per-genome existence q-values via knockoff...")
        df_scored = self._add_knockoff_existence_stats(
            df_scored,
            unique_pvalue_mode=mode,
            unique_peptide_error_source=unique_peptide_error_source,
            min_unique_for_unique_pvalue=int(min_unique_for_unique_pvalue),
            unique_count_power=float(unique_count_power),
            unique_count_cap=unique_count_cap,
        )
        self.timing_stats["knockoff_pvalues"] = float(time.time() - t_knock0)

        # Re-sort by presence_score (descending) before coverage computation
        self.logger.info("Re-sorting by presence_score...")
        df_scored = df_scored.sort_values("presence_score", ascending=False, kind="mergesort").reset_index(drop=True)
        df_scored["presence_rank"] = np.arange(1, len(df_scored) + 1, dtype=int)

        if compute_coverage:
            self.logger.info("Computing coverage statistics (reference only; not used for final calling) ...")
            t_cov0 = time.time()
            df_scored = self._add_coverage_stats(df_scored, order_col="presence_rank")
            self.timing_stats["coverage_stats"] = float(time.time() - t_cov0)

        df_scored = self._attach_lineage_column(df_scored)
        self.genome_scores_df = df_scored

        if unit_aware:
            t_unit0 = time.time()
            self.logger.info("Computing unit-aware genome presence outputs...")
            unit_level_df, cohort_summary_df = self._compute_unit_aware_presence(
                df_scored=df_scored,
                peptide_deg=peptide_deg,
                peptide_unique_owner=peptide_unique_owner,
                mode=mode,
                min_unique_for_unique_pvalue=int(min_unique_for_unique_pvalue),
            )
            mapping_df = self.sample_unit_mapping_df if self.sample_unit_mapping_df is not None else pd.DataFrame()
            self.timing_stats["unit_aware_outputs"] = float(time.time() - t_unit0)
        else:
            unit_level_df = None
            cohort_summary_df = None
            mapping_df = None

        source_cols = [
            "genome_id",
        ]
        if "Lineage" in df_scored.columns:
            source_cols.append("Lineage")
        source_cols.extend([
            "evidence_rank",
            "presence_rank",
            "num_peptides_matched",
            "num_peptides_unique",
            "unique_effective_count",
        ])
        if mode == "hypergeometric-opportunity":
            source_cols.append("theoretical_unique_peptides")
        source_cols.extend([
            "unique_expected_null",
            "unique_depth_fold",
            "unique_gate_pass",
            "p_shared_knock",
            "p_unique",
            "p_unique_depth",
            "p_presence",
            "q_presence",
            "presence_score",
            "pass_q_0_01",
            "pass_q_0_05",
        ])
        if "cumulative_coverage_percent" in df_scored.columns:
            source_cols.append("cumulative_coverage_percent")
        rename_map = {
            "p_shared_knock": "pvalue_shared",
            "p_unique": "pvalue_unique",
            "p_unique_depth": "pvalue_unique_depth",
            "p_presence": "pvalue",
            "q_presence": "qvalue",
            "unique_expected_null": "expected_unique_null",
        }
        missing = [c for c in source_cols if c not in df_scored.columns]
        if missing:
            raise ValueError(f"Missing required columns for main result table: {missing}")

        df_main = df_scored[source_cols].copy().rename(columns=rename_map)
        
        df_out = df_scored if return_full_table else df_main
        t_save0 = time.time()
        if unit_aware:
            self.logger.info(f"Saving unit-aware primary results to: {output_tsv_path}")
            if unit_level_df is None or cohort_summary_df is None or mapping_df is None:
                raise RuntimeError("Unit-aware output tables were not computed.")
            unit_tables = self._prepare_unit_aware_output_tables(
                unit_level_df=unit_level_df,
                cohort_summary_df=cohort_summary_df,
            )
            self._export_unit_aware_primary_outputs(
                out_dir=out_dir,
                stem=stem,
                requested_output_path=output_tsv_path,
                pooled_df=df_out,
                tables=unit_tables,
                mapping_df=mapping_df,
                export_pooled_result=bool(export_temp),
            )
            if export_unit_derived_tables:
                self._export_unit_aware_derived_outputs(out_dir=out_dir, stem=stem, tables=unit_tables)
        else:
            self.logger.info(f"Saving results to: {output_tsv_path}")
            df_out.to_csv(output_tsv_path, sep="\t", index=False)

        self.timing_stats["save_tsv"] = float(time.time() - t_save0)
        self.timing_stats["total_runtime_before_export"] = float(time.time() - t_all0)

        # --- NEW: export extra artifacts for paper figures ---
        if export_temp:
            try:
                t_export0 = time.time()
                self._export_temp_artifacts(
                    out_dir=out_dir,
                    stem=stem,
                    df_scored=df_scored,
                    export_peptide_contrib_topN=export_peptide_contrib_topN,
                )
                self.timing_stats["export_temp"] = float(time.time() - t_export0)
            except Exception as e:
                self.logger.warning(f"Failed to export temp artifacts: {e}")

        self._print_summary()
        result = df_scored if return_full_table else df_main
        if unit_aware and isinstance(self._last_unit_genome_presence_df, pd.DataFrame):
            return self._last_unit_genome_presence_df.copy()
        return result

    # =========================
    # Summary
    # =========================
    def _print_summary(self) -> None:
        if self.genome_scores_df is None or len(self.genome_scores_df) == 0:
            return

        df = self.genome_scores_df
        target = df.loc[df["_genomes_with_any_match"]].copy()

        def _format_qvalue(value: object) -> str:
            return f"{float(value):.3g}" if pd.notna(value) else "NA"

        def _print_row(rank_label: object, row: pd.Series) -> None:
            print(
                f"{rank_label}. {row['genome_id']} | Unique_Pep={int(row['num_peptides_unique'])}, "
                f"Matched_Pep={int(row['num_peptides_matched'])}, "
                f"Qvalue={_format_qvalue(row.get('q_presence', np.nan))}, "
                f"Coverage={float(row.get('cumulative_coverage_percent', 0.0)):.1f}%"
            )

        def _get_threshold_window_positions(threshold: float, window_size: int = 2) -> List[int]:
            if "q_presence" not in target.columns or target.empty:
                return []

            qvals = pd.to_numeric(target["q_presence"], errors="coerce")
            passing_positions = np.flatnonzero((qvals <= threshold).fillna(False).to_numpy(dtype=bool))
            failing_positions = np.flatnonzero((qvals > threshold).fillna(False).to_numpy(dtype=bool))

            if len(passing_positions) == 0 and len(failing_positions) == 0:
                return []

            positions: List[int] = []
            if len(passing_positions) > 0:
                start = max(0, len(passing_positions) - window_size)
                positions.extend(int(pos) for pos in passing_positions[start:])
            if len(failing_positions) > 0:
                positions.extend(int(pos) for pos in failing_positions[:window_size])
            return sorted(set(positions))

        def _print_threshold_window(threshold: float, positions: List[int], window_size: int = 2) -> None:
            if not positions:
                return

            qvals = pd.to_numeric(target["q_presence"], errors="coerce")
            passing_positions = np.flatnonzero((qvals <= threshold).fillna(False).to_numpy(dtype=bool))
            failing_positions = np.flatnonzero((qvals > threshold).fillna(False).to_numpy(dtype=bool))
            print(f"\nAround q<={threshold:.2f}:")
            printed_any = False

            if len(passing_positions) > 0:
                start = max(0, len(passing_positions) - window_size)
                for pos in passing_positions[start:]:
                    row = target.iloc[int(pos)]
                    rank_label = int(row["presence_rank"]) if "presence_rank" in row.index else int(pos) + 1
                    _print_row(rank_label, row)
                    printed_any = True

            if len(passing_positions) > 0 and len(failing_positions) > 0:
                print("...")

            if len(failing_positions) > 0:
                for pos in failing_positions[:window_size]:
                    row = target.iloc[int(pos)]
                    rank_label = int(row["presence_rank"]) if "presence_rank" in row.index else int(pos) + 1
                    _print_row(rank_label, row)
                    printed_any = True

            if not printed_any:
                print("(no genomes around this threshold)")

        print("\n======= MetaUmbra scoring summary =======")
        print(f"Mode: {'unit-aware' if self.unit_aware_output_paths else 'pooled peptide-set scoring'}")
        print(f"Genomes analyzed: {len(df)}")
        print(f"Genomes with matched>=1: {int(df['_genomes_with_any_match'].sum())}")

        if "q_presence" in df.columns:
            keep01 = int(df["pass_q_0_01"].fillna(False).sum())
            keep05 = int(df["pass_q_0_05"].fillna(False).sum())
            if self.unit_aware_output_paths:
                print("\nPooled peptide-set result:")
                print(f"Pooled genomes q<=0.01: {keep01}")
                print(f"Pooled genomes q<=0.05: {keep05}")
            else:
                print(f"Genomes q<=0.01: {keep01}")
                print(f"Genomes q<=0.05: {keep05}")

        if self.unit_aware_output_paths:
            cohort = self.unit_aware_cohort_summary_df
            if cohort is not None and not cohort.empty:
                n_units = int(pd.to_numeric(cohort["n_units_tested"], errors="coerce").fillna(0).max())
                q01_units = pd.to_numeric(cohort["n_units_q_le_0_01"], errors="coerce").fillna(0).astype(int)
                q05_units = pd.to_numeric(cohort["n_units_q_le_0_05"], errors="coerce").fillna(0).astype(int)
                tested_units = pd.to_numeric(cohort["n_units_tested"], errors="coerce").fillna(0).astype(int)
                q05_any = int((q05_units >= 1).sum())
                q05_all = int(((tested_units > 0) & (q05_units == tested_units)).sum())
                q01_any = int((q01_units >= 1).sum())
                q01_all = int(((tested_units > 0) & (q01_units == tested_units)).sum())
            else:
                n_units = 0
                q05_any = 0
                q05_all = 0
                q01_any = 0
                q01_all = 0

            print("\nUnit-level result:")
            print(f"Analysis units: {n_units}")
            print(f"Genomes q<=0.05 in >=1 unit: {q05_any}")
            print(f"Genomes q<=0.05 in all units: {q05_all}")
            print(f"Genomes q<=0.01 in >=1 unit: {q01_any}")
            print(f"Genomes q<=0.01 in all units: {q01_all}")

            print("\nPrimary unit-aware outputs:")
            primary_labels = [
                ("Unit genome presence", "unit_genome_presence"),
                ("Cohort genome summary", "cohort_genome_summary"),
                ("Sample-unit mapping", "sample_unit_mapping"),
            ]
            for label, key in primary_labels:
                path = self.unit_aware_output_paths.get(key)
                if path:
                    print(f"  {label}: {path}")

            print("\nSupplementary outputs:")
            supplementary_labels = [
                ("Pooled genome presence", "pooled_genome_presence"),
                ("Derived unit-aware tables", "derived_unit_aware_tables_dir"),
            ]
            for label, key in supplementary_labels:
                path = self.unit_aware_output_paths.get(key)
                if path:
                    print(f"  {label}: {path}")

        top = target.head(10)
        print("\nTop 10 pooled target genomes by rank:")
        for i, (_, r) in enumerate(top.iterrows(), 1):
            _print_row(i, r)
        if len(target) > 10:
            print("...")

        if "q_presence" in target.columns:
            window_001 = _get_threshold_window_positions(0.01)
            window_005 = _get_threshold_window_positions(0.05)
            _print_threshold_window(0.01, window_001)
            if window_001 and window_005 and (max(window_001) + 1 < min(window_005)):
                print("...")
            _print_threshold_window(0.05, window_005)


if __name__ == "__main__":
    if __package__ in {None, ""}:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from metaumbra.cli import main as cli_main
    else:
        from .cli import main as cli_main
    raise SystemExit(cli_main(["score", *sys.argv[1:]]))

