"""Empirical-background unique-evidence calibration helpers.

The production empirical statistic retains the established alpha-excess
conversion after estimating a threshold within a theoretical
panel-unique-opportunity stratum.  The finite-sample direct empirical tail is
also reported and can be selected explicitly for diagnostic sensitivity
analysis; it is not the production default.
"""

from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd

from .stats import _clip_pvalue

DEFAULT_UNIQUE_EMPIRICAL_BACKGROUND_THRESHOLD_QUANTILE = 0.95
DEFAULT_EMPIRICAL_MIN_COMPARABLE_BACKGROUND = 100
DEFAULT_EMPIRICAL_MIN_EXPECTED_UPPER_TAIL = 5.0
DEFAULT_EMPIRICAL_MIN_ADEQUATE_FRACTION = 0.90
DEFAULT_EMPIRICAL_MAX_OPPORTUNITY_BINS = 8
DEFAULT_EMPIRICAL_MIN_BIN_SIZE = 50
DEFAULT_UNIQUE_EMPIRICAL_PVALUE_METHOD = "alpha-excess"
UNIQUE_EMPIRICAL_PVALUE_METHODS = ("empirical-tail", "alpha-excess")
EMPIRICAL_BACKGROUND_OUTPUT_COLUMNS = (
    "p_unique_empirical_formal",
    "p_unique_empirical_background_excess",
    "p_unique_empirical_tail",
    "unique_empirical_background_bin",
    "unique_empirical_background_size",
    "unique_empirical_background_threshold",
    "unique_empirical_excess_count",
    "unique_alpha_excess_index",
    "unique_empirical_q95_threshold",
    "empirical_tail_percentile",
    "empirical_background_size",
    "minimum_attainable_empirical_p",
    "unique_excess_count",
    "expected_unique_null",
    "unique_depth_fold",
)


def _normalize_unique_empirical_pvalue_method(method: str) -> str:
    """Normalize the formal empirical p-value method.

    ``alpha-excess`` is the established production default.  ``empirical-tail``
    remains an explicit finite-sample diagnostic and sensitivity option.
    """
    normalized = str(method or DEFAULT_UNIQUE_EMPIRICAL_PVALUE_METHOD).strip().lower()
    aliases = {
        "tail": "empirical-tail",
        "direct-tail": "empirical-tail",
        "empirical_upper_tail": "empirical-tail",
        "alpha": "alpha-excess",
        "alpha_power": "alpha-excess",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in UNIQUE_EMPIRICAL_PVALUE_METHODS:
        raise ValueError(
            "unique empirical p-value method must be one of "
            f"{', '.join(UNIQUE_EMPIRICAL_PVALUE_METHODS)}; got {method!r}"
        )
    return normalized


def _evaluate_empirical_background_suitability(
    iteration_trace,
    *,
    final_exclude_fraction: float,
    max_exclude_fraction: float,
    min_adequate_fraction: float = DEFAULT_EMPIRICAL_MIN_ADEQUATE_FRACTION,
    final_bin_diagnostics=None,
) -> dict:
    """Summarize iterative empirical-background behavior for diagnostics.

    These outcome-derived diagnostics do not select a formal unique-evidence
    mode. Auto mode is resolved exclusively from structural eligibility.
    """
    trace = list(iteration_trace or [])
    cap = float(np.clip(max_exclude_fraction, 0.0, 1.0))
    final_fraction = float(np.clip(final_exclude_fraction, 0.0, 1.0))
    min_fraction = float(np.clip(min_adequate_fraction, 0.0, 1.0))
    cap_reached = bool(final_fraction >= cap - 1e-12)
    last = trace[-1] if trace and isinstance(trace[-1], dict) else {}
    next_requested = float(
        last.get("final_candidate_fraction", last.get("candidate_fraction", final_fraction))
    )
    update_above_cap = bool(next_requested > cap + 1e-12)
    converged = False
    if trace:
        converged = bool(
            abs(
                float(last.get("new_exclude_fraction", 0.0))
                - float(last.get("exclude_fraction", 0.0))
            )
            < 0.01
        )
    bin_df = final_bin_diagnostics if isinstance(final_bin_diagnostics, pd.DataFrame) else pd.DataFrame()
    if not bin_df.empty and "empirical_background_cap_pressure" in bin_df.columns:
        active_total = float(pd.to_numeric(bin_df.get("active_genome_count", 0), errors="coerce").fillna(0).sum())
        pressured_total = float(
            pd.to_numeric(
                bin_df.loc[bin_df["empirical_background_cap_pressure"].astype(bool), "active_genome_count"],
                errors="coerce",
            ).fillna(0).sum()
        )
        affected_fraction = pressured_total / max(active_total, 1.0)
    else:
        affected_fraction = 0.0
    broad_stratum_pressure = bool(affected_fraction + 1e-12 >= min_fraction)
    cap_pressure = bool(
        cap_reached
        and (update_above_cap or not converged)
        and broad_stratum_pressure
    )
    suitable = not cap_pressure
    if suitable:
        if not cap_reached:
            reason = "final exclusion remained below the exclusion cap"
        elif not (update_above_cap or not converged):
            reason = "iteration converged at the exclusion cap without requesting a larger exclusion"
        else:
            reason = (
                "exclusion-cap pressure affected only "
                f"{affected_fraction:.3f} of active candidates, below the "
                f"{min_fraction:.3f} run-level threshold"
            )
    else:
        reason = (
            "exclusion-cap pressure or nonconvergence affected "
            f"{affected_fraction:.3f} of active candidates across opportunity strata"
        )
    return {
        "suitable": suitable,
        "reason": reason,
        "cap_reached": cap_reached,
        "update_above_cap": update_above_cap,
        "converged": converged,
        "cap_pressure": cap_pressure,
        "affected_candidate_fraction": float(affected_fraction),
        "adequate_bin_fraction": float(1.0 - affected_fraction),
        "min_adequate_fraction": min_fraction,
        "next_requested_exclusion_fraction": float(next_requested),
    }


def _evaluate_empirical_background_eligibility(
    theoretical_opportunity_counts,
    *,
    min_comparable_background: int = DEFAULT_EMPIRICAL_MIN_COMPARABLE_BACKGROUND,
    threshold_quantile: float = DEFAULT_UNIQUE_EMPIRICAL_BACKGROUND_THRESHOLD_QUANTILE,
    min_expected_upper_tail: float = DEFAULT_EMPIRICAL_MIN_EXPECTED_UPPER_TAIL,
    min_adequate_fraction: float = DEFAULT_EMPIRICAL_MIN_ADEQUATE_FRACTION,
    max_bins: int = DEFAULT_EMPIRICAL_MAX_OPPORTUNITY_BINS,
    min_bin_size: int = DEFAULT_EMPIRICAL_MIN_BIN_SIZE,
) -> Tuple[pd.DataFrame, dict]:
    """Evaluate empirical-background adequacy as a pre-scoring structural operational heuristic.

    Eligibility relies on structural properties available before evaluating
    candidate significance (e.g. theoretical peptide opportunity counts).
    Candidates are grouped into theoretical opportunity bins. A candidate is
    considered adequate when its bin contains at least `min_comparable_background`
    other genomes. (Note: at threshold_quantile=0.95, min_comparable_background=100
    naturally implies an expected upper tail count of 100 * 0.05 = 5.0).
    """
    min_comparable_background = int(max(1, min_comparable_background))
    min_expected_upper_tail = float(max(0.0, min_expected_upper_tail))
    min_adequate_fraction = float(np.clip(min_adequate_fraction, 0.0, 1.0))
    max_bins = int(max(1, max_bins))
    min_bin_size = int(max(1, min_bin_size))
    threshold_quantile = float(np.clip(threshold_quantile, 0.0, 1.0))

    opportunity = pd.to_numeric(
        pd.Series(list(theoretical_opportunity_counts), dtype="float64"),
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)
    n_candidates = int(len(opportunity))
    required_for_tail = int(
        np.ceil(min_expected_upper_tail / max(1.0 - threshold_quantile, 1e-12))
    )
    target_bin_size = max(
        min_bin_size,
        min_comparable_background + 1,
        required_for_tail + 1,
    )
    effective_bins = max(1, min(max_bins, n_candidates // target_bin_size))

    if n_candidates == 0:
        details = pd.DataFrame(
            columns=[
                "candidate_index",
                "theoretical_opportunity",
                "opportunity_bin",
                "comparable_background_count",
                "expected_upper_tail_count",
                "empirical_background_adequate",
            ]
        )
        return details, {
            "eligible": False,
            "reason": "no evaluated candidate genomes",
            "candidate_count": 0,
            "effective_bins": 0,
            "adequate_candidate_count": 0,
            "adequate_candidate_fraction": 0.0,
            "min_comparable_background": min_comparable_background,
            "min_expected_upper_tail": min_expected_upper_tail,
            "min_adequate_fraction": min_adequate_fraction,
            "threshold_quantile": threshold_quantile,
            "target_bin_size": target_bin_size,
        }

    bin_labels = pd.Series("bin_0", index=opportunity.index, dtype="string")
    if effective_bins > 1 and int(opportunity.nunique()) > 1:
        try:
            binned = pd.qcut(opportunity, q=effective_bins, duplicates="drop")
            codes = binned.cat.codes.astype(int)
            bin_labels = codes.map(lambda value: f"bin_{int(value)}").astype("string")
        except (ValueError, TypeError):
            effective_bins = 1

    bin_counts = bin_labels.value_counts().to_dict()
    comparable = bin_labels.map(lambda label: max(int(bin_counts[str(label)]) - 1, 0)).astype(int)
    expected_tail = comparable.astype(float) * float(1.0 - threshold_quantile)
    adequate = (
        (comparable >= min_comparable_background)
        & (expected_tail + 1e-12 >= min_expected_upper_tail)
        & ((comparable + 1) >= min_bin_size)
    )
    adequate_count = int(adequate.sum())
    adequate_fraction = float(adequate_count) / float(max(n_candidates, 1))
    panel_eligible = bool(adequate_fraction + 1e-12 >= min_adequate_fraction)
    reason = (
        "structural empirical-background adequacy criterion satisfied"
        if panel_eligible
        else (
            f"only {adequate_count}/{n_candidates} candidates ({adequate_fraction:.3f}) "
            f"had >= {min_comparable_background} comparable backgrounds and >= "
            f"{min_expected_upper_tail:g} expected observations above q{threshold_quantile:g}"
        )
    )
    details = pd.DataFrame(
        {
            "candidate_index": np.arange(n_candidates, dtype=int),
            "theoretical_opportunity": opportunity.to_numpy(dtype=float),
            "opportunity_bin": bin_labels.astype(str).to_numpy(),
            "comparable_background_count": comparable.to_numpy(dtype=int),
            "expected_upper_tail_count": expected_tail.to_numpy(dtype=float),
            "empirical_background_adequate": adequate.to_numpy(dtype=bool),
        }
    )
    return details, {
        "eligible": panel_eligible,
        "reason": reason,
        "candidate_count": n_candidates,
        "effective_bins": int(details["opportunity_bin"].nunique()),
        "adequate_candidate_count": adequate_count,
        "adequate_candidate_fraction": adequate_fraction,
        "min_comparable_background": min_comparable_background,
        "min_expected_upper_tail": min_expected_upper_tail,
        "min_adequate_fraction": min_adequate_fraction,
        "threshold_quantile": threshold_quantile,
        "target_bin_size": target_bin_size,
        "min_comparable_observed": int(comparable.min()),
        "median_comparable_observed": float(comparable.median()),
        "max_comparable_observed": int(comparable.max()),
    }


def _compute_empirical_background_stats_for_table(
    df_scored: pd.DataFrame,
    *,
    alpha: float,
    top_exclude_fraction: float,
    threshold_quantile: float = DEFAULT_UNIQUE_EMPIRICAL_BACKGROUND_THRESHOLD_QUANTILE,
    n_bins: int = 8,
    min_bin_size: int = 50,
    pvalue_method: str = DEFAULT_UNIQUE_EMPIRICAL_PVALUE_METHOD,
) -> Tuple[pd.DataFrame, dict]:
    """Compute empirical-background unique evidence diagnostics for one scored table.

    The input must contain ``theoretical_panel_unique_peptide_opportunity``.
    No fallback to total theoretical peptide count is permitted because doing
    so changes the estimand and can contaminate the empirical null.
    """
    pvalue_method = _normalize_unique_empirical_pvalue_method(pvalue_method)
    required_opportunity_column = "theoretical_panel_unique_peptide_opportunity"
    if df_scored is not None and not df_scored.empty and required_opportunity_column not in df_scored.columns:
        raise ValueError(
            "Empirical-background scoring requires the corrected theoretical panel-unique opportunity "
            f"column {required_opportunity_column!r}; no total-peptide fallback is allowed."
        )
    defaults: Dict[str, object] = {
        "p_unique_empirical_formal": 1.0,
        "p_unique_empirical_background_excess": 1.0,
        "p_unique_empirical_tail": 1.0,
        "unique_empirical_background_bin": "",
        "unique_empirical_background_size": 0,
        "unique_empirical_background_threshold": 0.0,
        "unique_empirical_excess_count": 0.0,
        "unique_alpha_excess_index": 1.0,
        "unique_empirical_q95_threshold": 0.0,
        "empirical_tail_percentile": 0.0,
        "empirical_background_size": 0,
        "minimum_attainable_empirical_p": 1.0,
        "unique_excess_count": 0.0,
        "expected_unique_null": 0.0,
        "unique_depth_fold": 0.0,
    }
    meta = {
        "unique_empirical_background_size": 0,
        "unique_empirical_background_active_genomes": 0,
        "unique_empirical_background_excluded_genomes": 0,
        "unique_empirical_background_requested_exclude_fraction": float(np.clip(top_exclude_fraction, 0.0, 1.0)),
        "unique_empirical_background_excluded_fraction": 0.0,
        "unique_empirical_background_bin_count": 0,
        "unique_empirical_background_min_bin_size": int(max(1, min_bin_size)),
        "unique_empirical_background_opportunity_source": "theoretical_panel_unique_peptide_opportunity",
        "unique_empirical_background_threshold_quantile": float(np.clip(threshold_quantile, 0.0, 1.0)),
        "unique_empirical_background_alpha": float(min(max(alpha, 1e-12), 1.0)),
    }
    out = df_scored.copy() if df_scored is not None else pd.DataFrame()
    for column, value in defaults.items():
        if column not in out.columns:
            out[column] = value
        else:
            out[column] = out[column].fillna(value)

    if out.empty:
        return out, meta

    if "_genomes_with_any_match" in out.columns:
        active_mask = out["_genomes_with_any_match"].astype(bool)
    else:
        active_mask = pd.Series(True, index=out.index)
        out["_genomes_with_any_match"] = True
    active = out.loc[active_mask].copy()
    if active.empty:
        return out, meta

    active["genome_id"] = active["genome_id"].astype(str)
    active["_unique_empirical_U"] = (
        pd.to_numeric(active.get("num_peptides_unique", 0), errors="coerce")
        .fillna(0)
        .clip(lower=0)
        .astype(int)
    )
    active["_unique_empirical_A"] = (
        pd.to_numeric(active[required_opportunity_column], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .astype(float)
    )

    exclude_fraction = float(np.clip(float(top_exclude_fraction), 0.0, 1.0))
    n_active = int(len(active))
    n_exclude = int(np.ceil(float(n_active) * exclude_fraction)) if exclude_fraction > 0 else 0
    n_exclude = min(max(n_exclude, 0), max(n_active - 1, 0))
    if n_exclude > 0:
        for metric in ("unique_weighted_evidence", "weighted_evidence"):
            if metric not in active.columns:
                active[metric] = 0.0
            active[metric] = pd.to_numeric(active[metric], errors="coerce").fillna(0.0)
        top_idx = (
            active.sort_values(
                ["_unique_empirical_U", "unique_weighted_evidence", "weighted_evidence", "genome_id"],
                ascending=[False, False, False, True],
                kind="mergesort",
            )
            .head(n_exclude)
            .index
        )
        excluded_idx: Set[object] = set(top_idx)
    else:
        excluded_idx = set()

    active["_unique_empirical_excluded_from_background"] = active.index.isin(excluded_idx)
    background = active.loc[~active["_unique_empirical_excluded_from_background"]].copy()
    if background.empty:
        background = active.copy()
        active["_unique_empirical_excluded_from_background"] = False

    requested_bins = int(max(1, n_bins))
    min_bin_size = int(max(1, min_bin_size))
    effective_bins = max(1, min(requested_bins, int(len(background) // min_bin_size) or 1))
    effective_bins = min(effective_bins, int(background["_unique_empirical_A"].nunique()) or 1)

    if effective_bins <= 1 or len(background) <= 1:
        active["_unique_empirical_bin"] = "bin_0"
    else:
        try:
            _, bin_edges = pd.qcut(
                background["_unique_empirical_A"],
                q=effective_bins,
                retbins=True,
                duplicates="drop",
            )
            bin_edges = np.asarray(bin_edges, dtype=float)
            if bin_edges.size <= 2:
                active["_unique_empirical_bin"] = "bin_0"
            else:
                bin_edges[0] = -np.inf
                bin_edges[-1] = np.inf
                labels = [f"bin_{i}" for i in range(bin_edges.size - 1)]
                binned = pd.cut(
                    active["_unique_empirical_A"],
                    bins=bin_edges,
                    labels=labels,
                    include_lowest=True,
                )
                active["_unique_empirical_bin"] = binned.astype("string").fillna("bin_0").astype(str)
        except Exception:
            active["_unique_empirical_bin"] = "bin_0"

    background = active.loc[~active["_unique_empirical_excluded_from_background"]].copy()
    all_background_u = pd.to_numeric(background["_unique_empirical_U"], errors="coerce").fillna(0).astype(int)
    alpha = float(min(max(alpha, 1e-12), 1.0))
    threshold_quantile = float(np.clip(float(threshold_quantile), 0.0, 1.0))
    for idx, row in active.iterrows():
        bin_id = str(row["_unique_empirical_bin"])
        same_bin = background.loc[background["_unique_empirical_bin"].astype(str) == bin_id]
        # Leave the candidate out of its own null whenever it is part of the
        # retained background.  Excluded signal candidates are already absent.
        if idx in same_bin.index:
            same_bin = same_bin.drop(index=idx)
        if same_bin.empty:
            same_bin_u = all_background_u.drop(index=idx, errors="ignore")
        else:
            same_bin_u = pd.to_numeric(same_bin["_unique_empirical_U"], errors="coerce").fillna(0).astype(int)
        bg_size = int(len(same_bin_u))
        U = int(row["_unique_empirical_U"])
        ge = int((same_bin_u >= U).sum()) if bg_size > 0 else 0
        p_tail = (1.0 + float(ge)) / (1.0 + float(bg_size))
        percentile = (
            float((same_bin_u <= U).sum()) / float(bg_size)
            if bg_size > 0
            else 0.0
        )
        minimum_attainable_p = 1.0 / (1.0 + float(bg_size))
        threshold = float(np.quantile(same_bin_u.to_numpy(dtype=float), threshold_quantile)) if bg_size > 0 else 0.0
        excess = float(max(0.0, float(U) - threshold))
        alpha_excess_index = float(alpha ** excess) if excess > 0 else 1.0
        p_value = p_tail if pvalue_method == "empirical-tail" else alpha_excess_index
        expected = float(same_bin_u.mean()) if bg_size > 0 else 0.0

        active.at[idx, "p_unique_empirical_formal"] = _clip_pvalue(p_value)
        active.at[idx, "p_unique_empirical_background_excess"] = _clip_pvalue(alpha_excess_index)
        active.at[idx, "p_unique_empirical_tail"] = _clip_pvalue(p_tail)
        active.at[idx, "unique_alpha_excess_index"] = _clip_pvalue(alpha_excess_index)
        active.at[idx, "expected_unique_null"] = float(expected)
        active.at[idx, "unique_depth_fold"] = float(U) / max(float(expected), 1e-12)
        active.at[idx, "unique_empirical_background_bin"] = bin_id
        active.at[idx, "unique_empirical_background_size"] = int(bg_size)
        active.at[idx, "unique_empirical_background_threshold"] = float(threshold)
        active.at[idx, "unique_empirical_excess_count"] = float(excess)
        active.at[idx, "unique_empirical_q95_threshold"] = float(threshold)
        active.at[idx, "empirical_tail_percentile"] = float(percentile)
        active.at[idx, "empirical_background_size"] = int(bg_size)
        active.at[idx, "minimum_attainable_empirical_p"] = float(
            minimum_attainable_p
        )
        active.at[idx, "unique_excess_count"] = float(excess)

    for column in EMPIRICAL_BACKGROUND_OUTPUT_COLUMNS:
        out.loc[active.index, column] = active[column]

    excluded_genomes = int(active["_unique_empirical_excluded_from_background"].sum())
    global_quantiles = (
        all_background_u.quantile([0.50, 0.90, 0.95, 0.99])
        if len(all_background_u) > 0
        else pd.Series({0.50: 0.0, 0.90: 0.0, 0.95: 0.0, 0.99: 0.0})
    )
    active_background_sizes = pd.to_numeric(
        active["unique_empirical_background_size"], errors="coerce"
    ).fillna(0.0)
    meta.update(
        {
            "unique_empirical_background_size": int(len(background)),
            "unique_empirical_background_active_genomes": int(n_active),
            "unique_empirical_background_excluded_genomes": int(excluded_genomes),
            "unique_empirical_background_requested_exclude_fraction": float(exclude_fraction),
            "unique_empirical_background_excluded_fraction": float(excluded_genomes) / float(max(n_active, 1)),
            "unique_empirical_background_bin_count": int(active["unique_empirical_background_bin"].nunique()),
            "unique_empirical_background_min_bin_size": int(min_bin_size),
            "unique_empirical_background_opportunity_source": "theoretical_panel_unique_peptide_opportunity",
            "unique_empirical_background_pvalue_method": pvalue_method,
            "unique_empirical_background_threshold_quantile": float(threshold_quantile),
            "unique_empirical_background_alpha": float(alpha),
            "background_unique_count_q50": float(global_quantiles.loc[0.50]),
            "background_unique_count_q90": float(global_quantiles.loc[0.90]),
            "background_unique_count_q95": float(global_quantiles.loc[0.95]),
            "background_unique_count_q99": float(global_quantiles.loc[0.99]),
            "minimum_empirical_p_resolution": float(
                (1.0 / (1.0 + active_background_sizes)).min()
            ) if len(active_background_sizes) else 1.0,
        }
    )
    bin_rows = []
    for bin_id, group in active.groupby("_unique_empirical_bin", sort=True):
        background_group = background.loc[background["_unique_empirical_bin"] == bin_id]
        bg_n = int(len(background_group))
        background_u = pd.to_numeric(
            background_group["_unique_empirical_U"], errors="coerce"
        ).fillna(0.0)
        group_background_sizes = pd.to_numeric(
            group["unique_empirical_background_size"], errors="coerce"
        ).fillna(0.0)
        quantiles = (
            background_u.quantile([0.50, 0.90, 0.95, 0.99])
            if bg_n > 0
            else pd.Series({0.50: 0.0, 0.90: 0.0, 0.95: 0.0, 0.99: 0.0})
        )
        min_resolution = float(
            (1.0 / (1.0 + group_background_sizes)).min()
        ) if len(group_background_sizes) else 1.0
        bin_rows.append(
            {
                "unique_empirical_background_bin": str(bin_id),
                "active_genome_count": int(len(group)),
                "background_genome_count": bg_n,
                "background_unique_count_q50": float(quantiles.loc[0.50]),
                "background_unique_count_q90": float(quantiles.loc[0.90]),
                "background_unique_count_q95": float(quantiles.loc[0.95]),
                "background_unique_count_q99": float(quantiles.loc[0.99]),
                "minimum_empirical_p_resolution": min_resolution,
                "min_observed_unique": int(group["_unique_empirical_U"].min()) if len(group) else 0,
                "max_observed_unique": int(group["_unique_empirical_U"].max()) if len(group) else 0,
            }
        )
    meta["unique_empirical_background_bin_diagnostics"] = pd.DataFrame(bin_rows)
    return out, meta
