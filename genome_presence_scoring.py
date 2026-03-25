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
#    - Unique evidence p-value upper bound (conservative): p_unique_upper = (peptide_error_cutoff) ** U.
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
import multiprocessing as mp
import concurrent.futures
from pathlib import Path
from typing import Optional, List, Dict, Set, Tuple, Union, FrozenSet
from collections import Counter
import json
import sys
import platform

import numpy as np
import pandas as pd
from tqdm import tqdm


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
            # Chunked read to reduce peak memory on large genome TSV files.
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
                # Fallback for files without "Peptide": read only the first column by index.
                fallback_to_first_col = True
                try:
                    fallback_available_columns = pd.read_csv(
                        genome_peptides_path,
                        sep="\t",
                        nrows=0,
                    ).columns.tolist()
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
            matched_peptides: Set[str] = set()
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
                if not chunk_unique:
                    continue
                new_peptides = chunk_unique.difference(seen_theoretical)
                if not new_peptides:
                    continue
                seen_theoretical.update(new_peptides)
                matched_peptides.update(new_peptides.intersection(_OBS_PEPTIDES_WORKER))

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
        self.num_workers = num_workers if num_workers is not None else max(1, (mp.cpu_count() or 1) - 1)
        self.logger = setup_logger("GenomePresenceScorer", log_file)

        # Core states
        self.peptide_score: Dict[str, float] = {}  # peptide -> normalized score in [0,1] (or 1.0)
        self.peptide_error_cutoff: float = 0.05    # stored from read_peptide_file()
        # Upper bound on per-peptide false match probability used for unique-evidence p-value bound.
        self.single_peptide_error_rate_upper_bound: float = 1.0
        self.peptide_table_dir: Optional[str] = None

        self.genome_matched_peptides: Dict[str, Set[str]] = {}  # genome -> matched peptides (observed ∩ theoretical)
        self.genome_total_theoretical_peptides: Dict[str, int] = {}  # genome -> total theoretical peptides count
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
        self.knockoff_stage2_p_exist_ranges: List[Tuple[float, float]] = [(0.005, 0.02), (0.02, 0.08)]

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


    # =========================
    # I/O: Peptide table
    # =========================
    def read_peptide_file(
        self,
        peptide_table_path: Optional[str] = None,
        peptide_table_df: Optional[pd.DataFrame] = None,
        peptide_seq_col: str = "Base Sequence",
        peptide_score_col: Optional[str] = "Score",
        peptide_decoy_flag_col: Optional[str] = "Target/Decoy",
        decoy_flag_value: str = "decoy",
        peptide_table_sep: str = "\t",
        peptide_error_col: Optional[str] = None, # ["PEP", "FDR", "AUTO", None, "Q.Value", ...]
        peptide_error_cutoff: float = 0.05,
    ) -> bool:
        """Read a peptide table and build peptide->score dictionary."""
        if peptide_table_path is None and peptide_table_df is None:
            raise ValueError("Either peptide_table_path or peptide_table_df must be provided.")
        if peptide_table_path is not None and peptide_table_df is not None:
            raise ValueError("Provide only one of peptide_table_path or peptide_table_df, not both.")

        if peptide_table_df is not None:
            df = peptide_table_df.copy()
            self.peptide_table_dir = os.getcwd()
            available_columns = df.columns.tolist()
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
        # Keep backward-compatible behavior: if peptide-level error filtering was not applied,
        # still use peptide_error_cutoff as the assumed per-peptide upper bound.
        if error_filter_applied:
            self.single_peptide_error_rate_upper_bound = float(min(max(peptide_error_cutoff, 1e-12), 1.0))
            self.run_stats["single_peptide_error_rate_upper_bound_source"] = "peptide_error_cutoff"
        else:
            self.single_peptide_error_rate_upper_bound = float(min(max(peptide_error_cutoff, 1e-12), 1.0))
            self.run_stats["single_peptide_error_rate_upper_bound_source"] = "assumed_from_peptide_error_cutoff_without_filter"
            self.logger.warning(
                "No peptide-level error filter was applied; assuming single_peptide_error_rate_upper_bound="
                f"{self.single_peptide_error_rate_upper_bound:.4g} from peptide_error_cutoff."
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
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Compute peptide degeneracy d(p) and per-genome unique peptide counts."""
        self.logger.info("Computing peptide degeneracy d(p) and per-genome unique counts ...")

        genome_to_matched_peptides: Dict[str, Set[str]] = {}
        for genome_id, matched_peptides, _ in all_matched_peptides:
            genome_to_matched_peptides.setdefault(genome_id, set()).update(matched_peptides)

        num_target_genomes = len(genome_to_matched_peptides)

        peptide_genome_count = Counter()
        for matched_peptides in tqdm(genome_to_matched_peptides.values(), desc="Counting peptide occurrences"):
            for peptide in matched_peptides:
                peptide_genome_count[peptide] += 1

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
        return peptide_deg, genome_unique_counts

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
    ) -> Union[float, Tuple[float, float, float, float, float]]:
        """Empirical p-value for shared evidence via knockoff Monte Carlo (optionally return null moments)."""
        counts = self.knockoff_shared_stratum_counts_by_genome.get(gid, None)
        if not counts:
            return (1.0, 0.0, 0.0, 0.0, 0.0) if return_moments else 1.0

        null_sum = np.zeros(int(K), dtype=np.float64)
        for key, c in counts.items():
            pool = self.knockoff_pools_weighted_contrib.get(key, None) if self.knockoff_pools_weighted_contrib else None
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


    def _add_knockoff_existence_stats(
        self,
        df_scored: pd.DataFrame,
        use_peptide_error_for_unique_pvalue: bool = True,
    ) -> pd.DataFrame:
        """Add per-genome knockoff existence p/q-values."""
        def _p_unique_upper_for_genome(gid: str) -> float:
            matched = self.genome_matched_peptides.get(gid, set())
            if not matched:
                return 1.0

            # Unique peptides are those with degeneracy == 1
            uniq = [p for p in matched if int(self.peptide_degeneracy.get(p, 1)) == 1]
            if not uniq:
                return 1.0

            # Use global alpha^U if per-peptide mode is disabled or unavailable.
            if not use_per_peptide_error:
                return float(peptide_error_upper ** len(uniq))

            # Product of per-peptide upper bounds; fallback to peptide_error_upper if missing
            errs = [self.peptide_error_upper_by_peptide.get(p, peptide_error_upper) for p in uniq]
            return float(np.prod(np.clip(errs, 1e-12, 0.5)))
        
        out = df_scored.copy()
        # Conservative defaults for genomes with no match / skipped inference.
        out["p_shared_knock"] = 1.0
        out["p_unique_upper"] = 1.0
        out["p_presence"] = 1.0
        out["q_presence"] = 1.0
        out["presence_score"] = 0.0

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
        peptide_error_upper = float(min(max(self.single_peptide_error_rate_upper_bound, 1e-12), 0.5))
        error_col = self.run_stats.get("peptide_error_col", None)
        has_per_peptide_error = bool(self.peptide_error_upper_by_peptide)
        use_per_peptide_error = bool(use_peptide_error_for_unique_pvalue and has_per_peptide_error)
        uses_pep_column = bool(use_per_peptide_error and isinstance(error_col, str) and error_col.upper() == "PEP")
        source_col_display = str(error_col) if error_col is not None else "none"
        self.run_stats["unique_pvalue_use_peptide_error_switch"] = bool(use_peptide_error_for_unique_pvalue)
        self.run_stats["unique_pvalue_use_peptide_error_for_unique_pvalue"] = bool(use_peptide_error_for_unique_pvalue)
        self.run_stats["unique_pvalue_uses_per_peptide_error"] = bool(use_per_peptide_error)
        self.run_stats["unique_pvalue_error_source_col"] = str(error_col) if use_per_peptide_error and error_col is not None else None
        self.run_stats["unique_pvalue_uses_pep_column"] = bool(uses_pep_column)
        if use_per_peptide_error:
            self.logger.info(
                f"Unique p-value mode: {'[PEP]' if uses_pep_column else '[per-peptide error column]'} "
                f"source_col='{source_col_display}', "
                f"per_peptide_error_n={len(self.peptide_error_upper_by_peptide)}, "
                f"global_fallback={peptide_error_upper:.4g}"
            )
        else:
            reason = "disabled_by_switch" if not bool(use_peptide_error_for_unique_pvalue) else "per_peptide_error_not_available"
            self.logger.info(
                f"Unique p-value mode: [(alpha={peptide_error_upper:.4g})^U] "
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

            p_unique_upper = _p_unique_upper_for_genome(genome_id)


            p_existence = self._fisher_p_2(p1=p_shared, p2=p_unique_upper)

            out.at[idx, "p_shared_knock"] = p_shared
            out.at[idx, "p_unique_upper"] = p_unique_upper
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

                        p_unique_upper = _p_unique_upper_for_genome(genome_id)

                        p_existence = self._fisher_p_2(p1=p_shared, p2=p_unique_upper)

                        out.at[idx, "p_shared_knock"] = p_shared
                        out.at[idx, "p_unique_upper"] = p_unique_upper
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
                    return pd.DataFrame({"set": [tag], "bin": [], "count": [], "fraction": []})
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
        use_peptide_error_for_unique_pvalue: bool = False,
        return_full_table: bool = False,
    ) -> pd.DataFrame:
        """End-to-end analysis producing a genome-level q-value (q_presence)."""
        if output_tsv_path is None:
            out_dir = self.peptide_table_dir if self.peptide_table_dir else os.getcwd()
            output_tsv_path = os.path.join(out_dir, "genome_presence.tsv")
            self.logger.info(f"Output file not specified. Using: {output_tsv_path}")
        else:
            out_dir = os.path.dirname(output_tsv_path) or "."
            os.makedirs(out_dir, exist_ok=True)

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

        # Normalize cache path (if provided)
        cache_pkl_path: Optional[str] = None
        if matched_peptides_cache_path:
            cache_path = str(matched_peptides_cache_path)
            cache_pkl_path = cache_path if cache_path.lower().endswith(".pkl") else f"{cache_path}.pkl"

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
            elif test_genomes_num and test_genomes_num < len(all_genome_files):
                selected_genome_files = random.sample(all_genome_files, int(test_genomes_num))
            else:
                selected_genome_files = all_genome_files

            if exclude_genome_ids:
                ex = {g.strip() for g in exclude_genome_ids if isinstance(g, str) and g.strip()}
                before = len(selected_genome_files)
                selected_genome_files = [p for p in selected_genome_files if p.stem not in ex]
                self.logger.info(f"Excluded {before - len(selected_genome_files)} genomes by exclude list.")

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
                if cache_pkl_path:
                    pkl_path = cache_pkl_path
                else:
                    pkl_path = os.path.join(out_dir, "matched_peptides.pkl")

                with open(pkl_path, "wb") as f:
                    pickle.dump(all_matched_peptides, f)
                self.logger.info(f"Saved matched peptides cache: {pkl_path}")

        self.timing_stats["scan_genomes"] = float(time.time() - t_scan0)

        self.genome_matched_peptides = {}
        self.genome_total_theoretical_peptides = {}
        obs_set = set(self.peptide_score.keys())

        for genome_id, matched_peptides, total_cnt in all_matched_peptides:
            matched_peptides = set(matched_peptides).intersection(obs_set)
            self.genome_matched_peptides.setdefault(genome_id, set()).update(matched_peptides)
            prev = self.genome_total_theoretical_peptides.get(genome_id, 0)
            self.genome_total_theoretical_peptides[genome_id] = max(prev, int(total_cnt))

        t_deg0 = time.time()
        peptide_deg, genome_unique_counts = self._calculate_peptide_degeneracy_and_unique_counts(all_matched_peptides)
        self.timing_stats["compute_degeneracy"] = float(time.time() - t_deg0)
        self.peptide_degeneracy = peptide_deg

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
            use_peptide_error_for_unique_pvalue=use_peptide_error_for_unique_pvalue,
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

        self.logger.info(f"Saving results to: {output_tsv_path}")
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
            "shared_fraction",
            "mean_degeneracy",
            "p_presence",
            "q_presence",
            "presence_score",
            "pass_q_0_01",
            "pass_q_0_05",
        ])
        rename_map = {
            "p_presence": "pvalue",
            "q_presence": "qvalue",
        }
        missing = [c for c in source_cols if c not in df_scored.columns]
        if missing:
            raise ValueError(f"Missing required columns for main result table: {missing}")

        df_main = df_scored[source_cols].copy().rename(columns=rename_map)
        df_main.to_csv(output_tsv_path, sep="\t", index=False)

        self.timing_stats["save_tsv"] = float(time.time() - t_all0)

        # --- NEW: export extra artifacts for paper figures ---
        if export_temp:
            try:
                stem = Path(output_tsv_path).stem
                self._export_temp_artifacts(
                    out_dir=out_dir,
                    stem=stem,
                    df_scored=df_scored,
                    export_peptide_contrib_topN=export_peptide_contrib_topN,
                )
                self.timing_stats["export_temp"] = float(time.time() - t_all0)
            except Exception as e:
                self.logger.warning(f"Failed to export temp artifacts: {e}")

        self._print_summary()
        return df_scored if return_full_table else df_main

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

        print("\n======= Genome existence scoring (knockoff) summary =======")
        print(f"Genomes analyzed: {len(df)}")
        print(f"Genomes with matched>=1: {int(df['_genomes_with_any_match'].sum())}")

        if "q_presence" in df.columns:
            keep01 = int(df["pass_q_0_01"].fillna(False).sum())
            keep05 = int(df["pass_q_0_05"].fillna(False).sum())
            print(f"Genomes q<=0.01: {keep01}")
            print(f"Genomes q<=0.05: {keep05}")

        top = target.head(10)
        print("\nTop 10 target genomes by rank:")
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


# =========================
# __main__ (edit parameters here; no argparse)
# =========================
if __name__ == "__main__":
    print("\n======= Genome existence scoring (knockoff) =======")
    t0 = time.time()
    
    # Load test configs from JSON in test_data
    config_path = Path("test_data/test_configs.json")

    with open(config_path, "r", encoding="utf-8") as f:
        test_dict = json.load(f)


    test_proj = "mix24x" 
    
    if test_proj in test_dict:
        params = test_dict[test_proj]
        peptide_table_path = params.get("peptide_table")
        genome_digest_dirs = list(params.get("genome_digest_dirs") or [])
        # normalize entries to str and drop empties
        genome_digest_dirs = [str(g) for g in genome_digest_dirs if g]
        # add common digest dir to all tests (kept from original script)
        genome_digest_dirs.append(r"C:/Users/max/Desktop/digested_genomes/UHGP_digested")
        output_tsv_path = params.get("output_tsv")
        exclude_genome_ids = list(params.get("exclude_genome_ids") or [])
    else:
        raise ValueError(f"Unknown test_proj: {test_proj}. Valid keys: {list(test_dict.keys())}")

    ### TEST for ONE project
    # peptide_table_path = "C:/Users/max/OneDrive - University of Ottawa/code/TaxaSeeker/test_data/pro3/peptides.tsv"
    # genome_digest_dirs = [
    #     r"C:/Users/max/Desktop/digested_genomes/UHGP_digested",]
    # output_tsv_path = r"C:\Users\max\OneDrive - University of Ottawa\code\TaxaSeeker\test_data\pro3/genome_presence_results.tsv"
    # exclude_genome_ids = [] 
    # with open(r"test_data\removed_genomes.txt", "r", encoding="utf-8") as f:
    #     for line in f:
    #         genome_id = line.strip()
    #         if genome_id:
    #             exclude_genome_ids.append(genome_id)

    
    peptide_seq_col = "Sequence"
    peptide_score_col = "score"          # set to None if not available
    peptide_error_col = "Q.Value"            # set to None if not available; auto-detects PEP/FDR when None
    peptide_error_cutoff = 0.05

    # peptide-level decoy flag (optional)
    peptide_decoy_flag_col = "Reverse"       # set to None if not available
    decoy_flag_value = "+"
    genome_lineage_table_path = None         # optional TSV/TXT with genome ID and Lineage columns
    genome_lineage_genome_id_col = "Genome_id"
    genome_lineage_lineage_col = "Lineage"

        
    out_dir = os.path.dirname(output_tsv_path) or "."
    pickle_path = os.path.join(out_dir, "matched_peptides.pkl")  # set to e.g. r"test_data\6bacteria\matched_peptides_cache.pkl" to save/load matched peptides cache (speeds up repeated runs)
    
    use_cache_if_exists = False  # if False, ignore existing cache file and force recomputation
    use_peptide_error_for_unique_pvalue = False  # True: use peptide-level PEP/FDR; False: use (alpha)^U (alpha=peptide_error_cutoff, default 0.05)
    

    
    # ---- Calculator ----
    calc = GenomePresenceScorer(
        num_workers=min(32, max(1, (mp.cpu_count() or 1) - 1))
    )

    # Shared-peptide weight is fixed as w(p)=1/d(p)

    # Knockoff tuning
    calc.knockoff_mc_iterations = 500
    calc.knockoff_stage2_mc_iterations = 2000
    calc.knockoff_stage2_p_exist_ranges = [(0.005, 0.02), (0.02, 0.08)]
    calc.knockoff_random_seed = 1
    calc.knockoff_top_n_targets = None

    # Read peptides
    calc.read_peptide_file(
        peptide_table_path=peptide_table_path,
        peptide_seq_col=peptide_seq_col,
        peptide_score_col=peptide_score_col,
        peptide_decoy_flag_col=peptide_decoy_flag_col,
        decoy_flag_value=decoy_flag_value,
        peptide_table_sep="\t",
        peptide_error_col=peptide_error_col,
        peptide_error_cutoff=peptide_error_cutoff,
    )

    # Run
    calc.analyze_genomes(
        genome_digest_dirs=genome_digest_dirs,
        output_tsv_path=output_tsv_path,
        genome_lineage_table_path=genome_lineage_table_path,
        genome_lineage_genome_id_col=genome_lineage_genome_id_col,
        genome_lineage_lineage_col=genome_lineage_lineage_col,
        exclude_genome_ids=exclude_genome_ids,
        all_matched_peptides=None,
        save_matched_peptides_cache=True,
        matched_peptides_cache_path=pickle_path,
        use_cache_if_exists=use_cache_if_exists,
        use_peptide_error_for_unique_pvalue=use_peptide_error_for_unique_pvalue,
    )

    print(f"\nDone. Elapsed: {time.time() - t0:.1f}s")
