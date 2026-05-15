"""Unit-aware output cleanup for MetaUmbra scoring.

This module keeps the public unit-aware output layout focused on the core result
tables and moves derived convenience tables behind an explicit export option.
"""

from __future__ import annotations

import os
from functools import wraps
from typing import Dict, List

import numpy as np
import pandas as pd


_PATCH_FLAG = "_metaumbra_unit_output_cleanup_applied"
_ORIGINAL_ANALYZE_ATTR = "_metaumbra_original_analyze_genomes"


def _ordered_columns(df: pd.DataFrame, preferred: List[str]) -> pd.DataFrame:
    ordered = [col for col in preferred if col in df.columns]
    ordered.extend([col for col in df.columns if col not in ordered])
    return df.loc[:, ordered].copy()


def _coerce_bool_like(series: pd.Series, column_name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.map(
        lambda value: pd.NA if pd.isna(value) else str(value).strip().lower()
    )
    truthy = {"true", "1", "1.0", "yes", "y", "t"}
    falsy = {"false", "0", "0.0", "no", "n", "f"}
    valid = normalized.isna() | normalized.isin(truthy.union(falsy))
    if not bool(valid.all()):
        bad_values = sorted({str(v) for v in series.loc[~valid].head(5).tolist()})
        raise RuntimeError(
            f"Unit-aware sanity check failed: {column_name} is not boolean-like: {bad_values}"
        )
    return normalized.isin(truthy).fillna(False).astype(bool)


def _require_columns(df: pd.DataFrame, required: List[str], table_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(
            f"Unit-aware sanity check failed: {table_name} is missing required columns: {missing}"
        )


def _sort_existing(df: pd.DataFrame, columns: List[str], ascending: List[bool]) -> pd.DataFrame:
    sort_cols = [col for col in columns if col in df.columns]
    if not sort_cols or df.empty:
        return df.reset_index(drop=True)
    sort_asc = [ascending[columns.index(col)] for col in sort_cols]
    return df.sort_values(sort_cols, ascending=sort_asc, kind="mergesort").reset_index(drop=True)


def _build_unit_call_counts(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "analysis_unit_id",
        "n_samples_in_unit",
        "n_genomes_q_le_0_01",
        "n_genomes_q_le_0_05",
        "n_genomes_matched_ge_1",
        "n_genomes_unique_ge_3",
        "median_qvalue",
        "best_qvalue",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

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
        pd.DataFrame(rows, columns=columns)
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
    return base.reset_index(drop=True).join(pivot.reset_index(drop=True))


def _prepare_unit_aware_output_tables(
    self,
    unit_level_df: pd.DataFrame,
    cohort_summary_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Validate and prepare unit-aware tables.

    Derived convenience tables are only materialized when explicitly requested,
    which avoids unnecessary work and avoids treating filtered/pivoted views as
    separate primary results.
    """
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
        raise RuntimeError(
            "Unit-aware sanity check failed: num_peptides_matched is smaller than num_peptides_unique."
        )

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
        "num_peptides_matched",
        "expected_unique_null",
        "unique_depth_fold",
        "theoretical_unique_peptides",
        "observed_unique_peptide_pool_size",
        "pvalue_unique",
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

    tables: Dict[str, pd.DataFrame] = {
        "unit_genome_presence": _ordered_columns(unit_level_df, unit_level_columns),
        "cohort_genome_summary": _ordered_columns(cohort_summary_df, cohort_columns),
    }

    if not bool(getattr(self, "_metaumbra_export_unit_derived_tables", False)):
        return tables

    significant_columns = [
        "analysis_unit_id",
        "genome_id",
        "Lineage",
        "presence_rank",
        "qvalue",
        "pvalue",
        "num_peptides_unique",
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
    if not set(genome_union_q001["genome_id"].astype(str)).issubset(
        set(genome_union_q005["genome_id"].astype(str))
    ):
        raise RuntimeError("Unit-aware sanity check failed: genome_union_q001 is not a subset of genome_union_q005.")

    matrix_df = unit_level_df.copy()
    matrix_df["analysis_unit_id"] = matrix_df["analysis_unit_id"].astype(str)
    matrix_df["genome_id"] = matrix_df["genome_id"].astype(str)
    unit_ids = sorted(matrix_df["analysis_unit_id"].dropna().unique().tolist())
    matrix_q001 = _build_presence_matrix(matrix_df, genome_union_q001, "pass_q_0_01", unit_ids, 0)
    matrix_q005 = _build_presence_matrix(matrix_df, genome_union_q005, "pass_q_0_05", unit_ids, 0)
    matrix_qvalue = _build_presence_matrix(matrix_df, tables["cohort_genome_summary"], "qvalue", unit_ids, np.nan)
    for matrix in (matrix_q001, matrix_q005):
        for unit_id in unit_ids:
            if unit_id in matrix.columns:
                matrix[unit_id] = pd.to_numeric(matrix[unit_id], errors="coerce").fillna(0).astype(int)

    tables.update(
        {
            "unit_call_counts": unit_call_counts,
            "unit_q001_genomes": unit_q001,
            "unit_q005_genomes": unit_q005,
            "genome_union_q001": genome_union_q001,
            "genome_union_q005": genome_union_q005,
            "genome_by_unit_q001_matrix": matrix_q001,
            "genome_by_unit_q005_matrix": matrix_q005,
            "genome_by_unit_qvalue_matrix": matrix_qvalue,
        }
    )
    return tables


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
    """Write only the primary unit-aware outputs by default."""
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
    self._metaumbra_last_unit_genome_presence_df = unit_level_out.copy()

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


def apply_patch() -> None:
    """Apply unit-aware output cleanup to GenomePresenceScorer."""
    from . import scoring

    cls = scoring.GenomePresenceScorer
    if getattr(cls, _PATCH_FLAG, False):
        return

    original_analyze = cls.analyze_genomes
    setattr(cls, _ORIGINAL_ANALYZE_ATTR, original_analyze)

    @wraps(original_analyze)
    def analyze_genomes(self, *args, **kwargs):
        unit_aware_requested = bool(kwargs.get("unit_aware", False))
        self._metaumbra_export_unit_derived_tables = bool(
            kwargs.get("export_unit_derived_tables", False)
        )
        self._metaumbra_last_unit_genome_presence_df = None
        result = original_analyze(self, *args, **kwargs)
        if unit_aware_requested and isinstance(
            getattr(self, "_metaumbra_last_unit_genome_presence_df", None), pd.DataFrame
        ):
            return self._metaumbra_last_unit_genome_presence_df.copy()
        return result

    cls._prepare_unit_aware_output_tables = _prepare_unit_aware_output_tables
    cls._export_unit_aware_primary_outputs = _export_unit_aware_primary_outputs
    cls._export_unit_aware_derived_outputs = _export_unit_aware_derived_outputs
    cls.analyze_genomes = analyze_genomes
    setattr(cls, _PATCH_FLAG, True)
