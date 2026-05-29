"""Empirical-background unique-evidence calibration helpers."""

from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd

from .stats import _clip_pvalue

DEFAULT_UNIQUE_EMPIRICAL_BACKGROUND_THRESHOLD_QUANTILE = 0.95
EMPIRICAL_BACKGROUND_OUTPUT_COLUMNS = (
    "p_unique_empirical_background_excess",
    "p_unique_empirical_tail",
    "unique_empirical_background_bin",
    "unique_empirical_background_size",
    "unique_empirical_background_threshold",
    "unique_empirical_excess_count",
    "expected_unique_null",
    "unique_depth_fold",
)


def _compute_empirical_background_stats_for_table(
    df_scored: pd.DataFrame,
    *,
    alpha: float,
    top_exclude_fraction: float,
    threshold_quantile: float = DEFAULT_UNIQUE_EMPIRICAL_BACKGROUND_THRESHOLD_QUANTILE,
    n_bins: int = 8,
    min_bin_size: int = 50,
) -> Tuple[pd.DataFrame, dict]:
    """Compute empirical-background unique evidence diagnostics for one scored table."""
    defaults: Dict[str, object] = {
        "p_unique_empirical_background_excess": 1.0,
        "p_unique_empirical_tail": 1.0,
        "unique_empirical_background_bin": "",
        "unique_empirical_background_size": 0,
        "unique_empirical_background_threshold": 0.0,
        "unique_empirical_excess_count": 0.0,
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
        "unique_empirical_background_opportunity_source": "total_peptide_count",
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
        pd.to_numeric(active.get("total_peptide_count", 0), errors="coerce")
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
        if same_bin.empty:
            same_bin_u = all_background_u
        else:
            same_bin_u = pd.to_numeric(same_bin["_unique_empirical_U"], errors="coerce").fillna(0).astype(int)
        bg_size = int(len(same_bin_u))
        U = int(row["_unique_empirical_U"])
        ge = int((same_bin_u >= U).sum()) if bg_size > 0 else 0
        p_tail = (1.0 + float(ge)) / (1.0 + float(bg_size))
        threshold = float(np.quantile(same_bin_u.to_numpy(dtype=float), threshold_quantile)) if bg_size > 0 else 0.0
        excess = float(max(0.0, float(U) - threshold))
        p_value = float(alpha ** excess) if excess > 0 else 1.0
        expected = float(same_bin_u.mean()) if bg_size > 0 else 0.0

        active.at[idx, "p_unique_empirical_background_excess"] = _clip_pvalue(p_value)
        active.at[idx, "p_unique_empirical_tail"] = _clip_pvalue(p_tail)
        active.at[idx, "expected_unique_null"] = float(expected)
        active.at[idx, "unique_depth_fold"] = float(U) / max(float(expected), 1e-12)
        active.at[idx, "unique_empirical_background_bin"] = bin_id
        active.at[idx, "unique_empirical_background_size"] = int(bg_size)
        active.at[idx, "unique_empirical_background_threshold"] = float(threshold)
        active.at[idx, "unique_empirical_excess_count"] = float(excess)

    for column in EMPIRICAL_BACKGROUND_OUTPUT_COLUMNS:
        out.loc[active.index, column] = active[column]

    excluded_genomes = int(active["_unique_empirical_excluded_from_background"].sum())
    meta.update(
        {
            "unique_empirical_background_size": int(len(background)),
            "unique_empirical_background_active_genomes": int(n_active),
            "unique_empirical_background_excluded_genomes": int(excluded_genomes),
            "unique_empirical_background_requested_exclude_fraction": float(exclude_fraction),
            "unique_empirical_background_excluded_fraction": float(excluded_genomes) / float(max(n_active, 1)),
            "unique_empirical_background_bin_count": int(active["unique_empirical_background_bin"].nunique()),
            "unique_empirical_background_min_bin_size": int(min_bin_size),
            "unique_empirical_background_opportunity_source": "total_peptide_count",
            "unique_empirical_background_threshold_quantile": float(threshold_quantile),
            "unique_empirical_background_alpha": float(alpha),
        }
    )
    return out, meta
