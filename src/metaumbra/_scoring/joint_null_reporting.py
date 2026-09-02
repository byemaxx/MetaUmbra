"""Machine-readable comparisons for the experimental joint-null pilot."""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd

from .ranking import bh_qvalues


METHOD_COLUMNS = {
    "current_bonferroni": "pvalue_combined_bonferroni",
    "unique_only": "pvalue_unique",
    "standard_fisher_independence_unvalidated": "pvalue_combined_fisher",
    "empirical_joint_null_fisher": "pvalue_combined_joint_null_fisher",
}


def build_joint_null_method_comparison(
    candidates: pd.DataFrame,
    *,
    family_id: str,
    expected_truth_by_genome: Optional[Mapping[str, bool]] = None,
) -> pd.DataFrame:
    """Return one candidate/method row with raw p, BH q, rank, and truth fields."""
    required = {
        "analysis_unit_id",
        "genome_id",
        "num_peptides_matched",
        "num_peptides_unique",
        *METHOD_COLUMNS.values(),
    }
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError("Candidate table is missing required columns: " + ", ".join(missing))

    base = candidates.copy()
    base["analysis_unit_id"] = base["analysis_unit_id"].astype(str)
    base["genome_id"] = base["genome_id"].astype(str)
    truth_available = expected_truth_by_genome is not None
    truth_map = dict(expected_truth_by_genome or {})
    rows = []
    for unit_id, unit in base.groupby("analysis_unit_id", sort=False):
        active = pd.to_numeric(unit["num_peptides_matched"], errors="coerce").fillna(0).to_numpy() >= 1
        unique_positive = (
            pd.to_numeric(unit["num_peptides_unique"], errors="coerce").fillna(0).to_numpy() > 0
        )
        methods = {
            method: np.clip(
                pd.to_numeric(unit[column], errors="coerce").fillna(1.0).to_numpy(dtype=float),
                1e-300,
                1.0,
            )
            for method, column in METHOD_COLUMNS.items()
        }
        methods["zero_unique_gated_standard_fisher"] = np.where(
            unique_positive,
            methods["standard_fisher_independence_unvalidated"],
            1.0,
        )
        for method, pvalues in methods.items():
            qvalues = np.ones(len(unit), dtype=float)
            if bool(np.any(active)):
                qvalues[active] = bh_qvalues(pvalues[active])
            order = np.lexsort((unit["genome_id"].to_numpy(dtype=object), pvalues, qvalues))
            ranks = np.empty(len(unit), dtype=int)
            ranks[order] = np.arange(1, len(unit) + 1, dtype=int)
            for position, (_, candidate) in enumerate(unit.iterrows()):
                genome_id = str(candidate["genome_id"])
                expected = truth_map.get(genome_id, False) if truth_available else pd.NA
                rows.append(
                    {
                        "family_id": str(family_id),
                        "analysis_unit_id": str(unit_id),
                        "genome_id": genome_id,
                        "method": method,
                        "pvalue_raw": float(pvalues[position]),
                        "qvalue_bh": float(qvalues[position]),
                        "rank": int(ranks[position]),
                        "num_peptides_matched": int(candidate["num_peptides_matched"]),
                        "num_peptides_unique": int(candidate["num_peptides_unique"]),
                        "benchmark_truth_available": bool(truth_available),
                        "expected_genome": expected,
                    }
                )
    result = pd.DataFrame(rows)
    result["expected_genome"] = result["expected_genome"].astype("boolean")
    return result


def summarize_joint_null_calls(comparison: pd.DataFrame) -> pd.DataFrame:
    """Summarize call counts, denominators, and benchmark recovery at two q thresholds."""
    required = {
        "family_id",
        "analysis_unit_id",
        "genome_id",
        "method",
        "qvalue_bh",
        "num_peptides_matched",
        "benchmark_truth_available",
        "expected_genome",
    }
    missing = sorted(required.difference(comparison.columns))
    if missing:
        raise ValueError("Comparison table is missing required columns: " + ", ".join(missing))

    rows = []
    group_cols = ["family_id", "analysis_unit_id", "method"]
    for keys, group in comparison.groupby(group_cols, sort=False):
        truth_available = bool(group["benchmark_truth_available"].all())
        expected = group["expected_genome"].fillna(False).astype(bool)
        active = pd.to_numeric(group["num_peptides_matched"], errors="coerce").fillna(0) >= 1
        for threshold in (0.01, 0.05):
            called = pd.to_numeric(group["qvalue_bh"], errors="coerce").fillna(1.0) <= threshold
            rows.append(
                {
                    "family_id": keys[0],
                    "analysis_unit_id": keys[1],
                    "method": keys[2],
                    "q_threshold": float(threshold),
                    "call_count": int(called.sum()),
                    "candidate_denominator": int(len(group)),
                    "active_candidate_denominator": int(active.sum()),
                    "benchmark_truth_available": truth_available,
                    "expected_genome_denominator": int(expected.sum()) if truth_available else pd.NA,
                    "expected_recovery_count": int((called & expected).sum()) if truth_available else pd.NA,
                    "additional_call_count": int((called & ~expected).sum()) if truth_available else pd.NA,
                }
            )
    return pd.DataFrame(rows)


def compare_calls_to_current(comparison: pd.DataFrame) -> pd.DataFrame:
    """Return candidate-level gains and losses relative to current Bonferroni."""
    rows = []
    key_columns = ["family_id", "analysis_unit_id", "genome_id"]
    for threshold in (0.01, 0.05):
        calls = comparison.assign(
            called=pd.to_numeric(comparison["qvalue_bh"], errors="coerce").fillna(1.0)
            <= threshold
        ).pivot_table(
            index=key_columns,
            columns="method",
            values="called",
            aggfunc="first",
            fill_value=False,
        )
        if "current_bonferroni" not in calls.columns:
            raise ValueError("Comparison table has no current_bonferroni rows.")
        current = calls["current_bonferroni"].astype(bool)
        for method in calls.columns:
            if method == "current_bonferroni":
                continue
            selected = calls[method].astype(bool)
            for relation, mask in (
                ("gained", selected & ~current),
                ("lost", current & ~selected),
            ):
                for family_id, unit_id, genome_id in calls.index[mask]:
                    rows.append(
                        {
                            "family_id": family_id,
                            "analysis_unit_id": unit_id,
                            "genome_id": genome_id,
                            "method": method,
                            "q_threshold": float(threshold),
                            "change_relative_to_current": relation,
                        }
                    )
    return pd.DataFrame(rows)


def build_joint_null_shared_impact(
    candidates: pd.DataFrame,
    *,
    family_id: str,
    expected_truth_by_genome: Optional[Mapping[str, bool]] = None,
    material_rank_change: int = 10,
) -> pd.DataFrame:
    """Compare the experimental combined result with its own unique component."""
    required = {
        "analysis_unit_id",
        "genome_id",
        "num_peptides_matched",
        "num_peptides_unique",
        "pvalue_joint_null_unique_component",
        "pvalue_combined_joint_null_fisher",
    }
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError("Candidate table is missing required columns: " + ", ".join(missing))
    if int(material_rank_change) < 1:
        raise ValueError("material_rank_change must be positive.")

    truth_available = expected_truth_by_genome is not None
    truth_map = dict(expected_truth_by_genome or {})
    rows = []
    for unit_id, unit in candidates.groupby("analysis_unit_id", sort=False):
        unit = unit.copy()
        active = pd.to_numeric(unit["num_peptides_matched"], errors="coerce").fillna(0).to_numpy() >= 1
        unique_p = np.clip(
            pd.to_numeric(unit["pvalue_joint_null_unique_component"], errors="coerce")
            .fillna(1.0)
            .to_numpy(dtype=float),
            1e-300,
            1.0,
        )
        combined_p = np.clip(
            pd.to_numeric(unit["pvalue_combined_joint_null_fisher"], errors="coerce")
            .fillna(1.0)
            .to_numpy(dtype=float),
            1e-300,
            1.0,
        )
        unique_q = np.ones(len(unit), dtype=float)
        combined_q = np.ones(len(unit), dtype=float)
        if bool(np.any(active)):
            unique_q[active] = bh_qvalues(unique_p[active])
            combined_q[active] = bh_qvalues(combined_p[active])

        def ranks_for(pvalues: np.ndarray, qvalues: np.ndarray) -> np.ndarray:
            order = np.lexsort(
                (unit["genome_id"].astype(str).to_numpy(dtype=object), pvalues, qvalues)
            )
            ranks = np.empty(len(unit), dtype=int)
            ranks[order] = np.arange(1, len(unit) + 1, dtype=int)
            return ranks

        unique_rank = ranks_for(unique_p, unique_q)
        combined_rank = ranks_for(combined_p, combined_q)
        for position, (_, candidate) in enumerate(unit.iterrows()):
            genome_id = str(candidate["genome_id"])
            expected = truth_map.get(genome_id, False) if truth_available else pd.NA
            rank_improvement = int(unique_rank[position] - combined_rank[position])
            rows.append(
                {
                    "family_id": str(family_id),
                    "analysis_unit_id": str(unit_id),
                    "genome_id": genome_id,
                    "num_peptides_unique": int(candidate["num_peptides_unique"]),
                    "pvalue_joint_null_unique_component": float(unique_p[position]),
                    "qvalue_joint_null_unique_component": float(unique_q[position]),
                    "rank_joint_null_unique_component": int(unique_rank[position]),
                    "pvalue_combined_joint_null_fisher": float(combined_p[position]),
                    "qvalue_combined_joint_null_fisher": float(combined_q[position]),
                    "rank_combined_joint_null_fisher": int(combined_rank[position]),
                    "raw_p_improved_with_shared": bool(combined_p[position] < unique_p[position]),
                    "q_improved_with_shared": bool(combined_q[position] < unique_q[position]),
                    "rank_improvement_with_shared": rank_improvement,
                    "material_rank_improvement_with_shared": bool(
                        rank_improvement >= int(material_rank_change)
                    ),
                    "gained_q_0_01_with_shared": bool(
                        combined_q[position] <= 0.01 < unique_q[position]
                    ),
                    "gained_q_0_05_with_shared": bool(
                        combined_q[position] <= 0.05 < unique_q[position]
                    ),
                    "benchmark_truth_available": bool(truth_available),
                    "expected_genome": expected,
                }
            )
    result = pd.DataFrame(rows)
    result["expected_genome"] = result["expected_genome"].astype("boolean")
    return result
