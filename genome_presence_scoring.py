# Genome existence scoring from a peptide list using peptide-space knockoff null
# Version: 4.3
# Date: 2026-02-11
#
# Workflow:
# 1) Read observed peptide list (optional peptide-level FDR filter).
# 2) For each genome (theoretical digest peptides), compute matched peptides = observed ∩ theoretical.
# 3) Compute peptide degeneracy d(p) across TARGET genomes (recommended).
# 4) Compute shared-aware evidence:
#       w(p)=1/d(p) (or optional IDF)
#       weighted_evidence = Σ w(p)*score(p)
#       unique_weighted_evidence = Σ score(p) for d(p)=1
#       weighted_evidence_shared = Σ w(p)*score(p) for d(p)>1
# 5) Peptide-space knockoff (no second database matching):
#    - Build pools of shared peptide contributions (w*s) stratified by degeneracy (and optional length).
#    - For each genome, sample from these pools according to that genome's shared-stratum counts to get
#      an empirical null for weighted_evidence_shared, yielding p_shared_knock.
#    - Unique evidence p-value upper bound (conservative): p_unique_upper = (peptide_fdr_cutoff) ** U.
#    - Combine with Fisher (2 p-values) => p_presence; BH => fdr_presence (per-genome existence q-value).
#
# Outputs:
# - rank_by_score: unified rank index (lexicographic; unique dominates).
# - fdr_presence: recommended per-genome existence q-value.
# - presence_strength: -log10(fdr_presence) for paper-friendly existence strength.

import os
import time
import pickle
import random
import logging
import multiprocessing as mp
import concurrent.futures
from pathlib import Path
from typing import Optional, List, Dict, Set, Tuple, Union, Literal
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
        self.peptide_fdr_cutoff: float = 0.05      # stored from read_peptide_file()
        self.peptide_table_dir: Optional[str] = None

        self.genome_matched_peptides: Dict[str, Set[str]] = {}  # genome -> matched peptides (observed ∩ theoretical)
        self.genome_total_theoretical_peptides: Dict[str, int] = {}  # genome -> total theoretical peptides count
        self.genome_scores_df: Optional[pd.DataFrame] = None

        # Shared-peptide / evidence parameters
        # weighting_mode:
        # - "inverse_degeneracy": w(p)=1/d(p)
        # - "idf": w(p)=log((N+1)/(d+1)) with optional log base
        self.weighting_mode: Literal["inverse_degeneracy", "idf"] = "inverse_degeneracy"
        self.idf_log_base = np.e

        # Unified ranking score scales (lexicographic; unique dominates)
        self.rank_lexico_scales = {
            "U": 10**12,   # unique_peptide_count
            "UW": 10**9,   # unique_weighted_evidence
            "WE": 10**6,   # weighted_evidence
            "EP": 10**3,   # effective_peptide_count
            "MR": 10**5,   # peptide_match_ratio (0..1)
            "M": 1         # matched_peptide_count
        }

        # Knockoff settings (scheme A)
        self.knockoff_enabled: bool = True

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

        # Optional speed knob: compute knockoff only for top-N TARGET genomes by rank_by_score, set others p=1
        self.knockoff_top_n_targets: Optional[int] = None

        # Internal caches for knockoff
        self.peptide_degeneracy: Optional[Dict[str, int]] = None
        self.num_target_genomes_for_degeneracy: Optional[int] = None
        self.knockoff_pools_weighted_contrib: Optional[Dict[Union[int, Tuple[int, int]], np.ndarray]] = None
        self.knockoff_shared_stratum_counts_by_genome: Dict[str, Counter] = {}  # genome_id -> Counter(stratum -> count)

        # --- NEW: paper-friendly run diagnostics ---
        self.run_stats: Dict[str, object] = {}
        self.timing_stats: Dict[str, float] = {}
        self.knockoff_pool_stats: Optional[pd.DataFrame] = None

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
        peptide_fdr_col: Optional[str] = "FDR",
        peptide_fdr_cutoff: float = 0.05,
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

            if peptide_fdr_col and peptide_fdr_col in available_columns:
                cols.append(peptide_fdr_col)
                dtype[peptide_fdr_col] = "float32"
            else:
                peptide_fdr_col = None

            df = pd.read_csv(peptide_file_path, sep=peptide_table_sep, usecols=cols, dtype=dtype, engine="c")
            self.logger.info(f"Loaded {len(df)} rows from peptide file.")

        if peptide_seq_col not in available_columns:
            raise ValueError(f"Missing peptide column '{peptide_seq_col}'.")

        self.peptide_fdr_cutoff = float(peptide_fdr_cutoff)

        # --- NEW: run-level input stats (for paper) ---
        self.run_stats["peptide_rows_loaded"] = int(len(df))
        self.run_stats["peptide_seq_col"] = peptide_seq_col
        self.run_stats["peptide_score_col"] = peptide_score_col if peptide_score_col else None
        self.run_stats["peptide_fdr_col"] = peptide_fdr_col if peptide_fdr_col else None
        self.run_stats["peptide_fdr_cutoff"] = float(peptide_fdr_cutoff)
        self.run_stats["peptide_decoy_flag_col"] = peptide_decoy_flag_col if peptide_decoy_flag_col else None
        self.run_stats["decoy_flag_value"] = decoy_flag_value

        if peptide_decoy_flag_col and peptide_decoy_flag_col in df.columns:
            before = len(df)
            df = df[(df[peptide_decoy_flag_col] != decoy_flag_value) | (df[peptide_decoy_flag_col].isna())]
            self.logger.info(f"Peptide-level decoy filter: {before} -> {len(df)} rows.")
            self.run_stats["peptide_rows_after_decoy_filter"] = int(len(df))

        if peptide_fdr_col and peptide_fdr_col in df.columns:
            before = len(df)
            df[peptide_fdr_col] = pd.to_numeric(df[peptide_fdr_col], errors="coerce")
            df = df[df[peptide_fdr_col] <= float(peptide_fdr_cutoff)]
            self.logger.info(f"Peptide-level FDR filter (<= {peptide_fdr_cutoff}): {before} -> {len(df)} rows.")
            self.run_stats["peptide_rows_after_fdr_filter"] = int(len(df))

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
        self.run_stats.setdefault("peptide_rows_after_fdr_filter", int(len(df)))

        self.logger.info(f"Observed peptides: {len(self.peptide_score)} (unique)")
        return True

    # =========================
    # Genome scan
    # =========================
    def _process_genome_batch(self, file_paths: List[Union[str, os.PathLike]]) -> List[Tuple[str, Set[str], int]]:
        """Process a batch of genome peptide files."""
        obs_peptides = set(self.peptide_score.keys())
        results: List[Tuple[str, Set[str], int]] = []

        for genome_peptides_path in file_paths:
            genome_peptides_path = str(genome_peptides_path)
            genome_id = Path(genome_peptides_path).stem
            try:
                genome_peptides_df = pd.read_csv(genome_peptides_path, sep="\t", engine="c")
                peptide_column_name = "Peptide" if "Peptide" in genome_peptides_df.columns else genome_peptides_df.columns[0]
                theoretical_peptides = set(genome_peptides_df[peptide_column_name].dropna().astype(str).values)
                matched_peptides = theoretical_peptides.intersection(obs_peptides)
                results.append((genome_id, matched_peptides, len(theoretical_peptides)))
            except Exception:
                results.append((genome_id, set(), 0))

        return results

    # =========================
    # Shared peptide degeneracy + weights
    # =========================
    def _compute_weight(self, d: int, N: int) -> float:
        """Shared-aware peptide weight given degeneracy d(p)=d."""
        d = int(max(d, 1))
        if self.weighting_mode == "idf":
            val = np.log((N + 1.0) / (d + 1.0))
            if self.idf_log_base != np.e:
                val = val / np.log(self.idf_log_base)
            return float(max(val, 0.0))
        return 1.0 / float(d)

    def _calculate_peptide_degeneracy_and_unique_counts(
        self,
        all_matched_peptides: List[Tuple[str, Set[str], int]],
    ) -> Tuple[Dict[str, int], Dict[str, int], int]:
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
        return peptide_deg, genome_unique_counts, num_target_genomes

    # =========================
    # Genome metrics
    # =========================
    def _calculate_genome_metrics(
        self,
        genome_data_list: List[Tuple[str, Set[str], int, int]],
        peptide_deg: Dict[str, int],
        N_targets_for_deg: int,
    ) -> pd.DataFrame:
        """Compute shared-aware metrics for each genome and record shared-stratum counts for knockoff."""
        self.knockoff_shared_stratum_counts_by_genome = {}

        out_rows = []
        default_score = 1.0

        for genome_id, matched_peptides, total_theoretical_peptides, unique_matched_peptides in tqdm(
            genome_data_list, desc="Computing genome metrics"
        ):
            matched_peptide_count = len(matched_peptides)

            if matched_peptide_count == 0 or total_theoretical_peptides == 0:
                out_rows.append({
                    "genome_id": genome_id,
                    "total_peptide_count": int(total_theoretical_peptides),
                    "matched_peptide_count": 0,
                    "unique_peptide_count": 0,
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
                weight = self._compute_weight(d=degeneracy, N=N_targets_for_deg)

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
            peptide_match_ratio = float(matched_peptide_count) / float(max(int(total_theoretical_peptides), 1))

            effective_peptide_count = float(np.sum(peptide_weights)) if peptide_weights else 0.0
            weighted_evidence = float(np.sum(weighted_contributions)) if weighted_contributions else 0.0

            effective_peptide_count_shared = float(np.sum(shared_peptide_weights)) if shared_peptide_weights else 0.0
            weighted_evidence_shared = float(np.sum(shared_weighted_contributions)) if shared_weighted_contributions else 0.0

            mean_degeneracy = float(np.mean(peptide_degeneracies)) if peptide_degeneracies else 0.0
            shared_fraction = (
                1.0 - (float(unique_matched_peptides) / float(matched_peptide_count))
                if matched_peptide_count > 0
                else 0.0
            )

            out_rows.append({
                "genome_id": genome_id,
                "total_peptide_count": int(total_theoretical_peptides),
                "matched_peptide_count": int(matched_peptide_count),
                "unique_peptide_count": int(unique_matched_peptides),
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
            1) unique_peptide_count (U)
            2) unique_weighted_evidence (UW)
            3) weighted_evidence (WE)
            4) effective_peptide_count (EP)
            5) peptide_match_ratio (MR)
            6) matched_peptide_count (M)
        Ties are broken deterministically by genome_id.

        The output rows are ordered from best to worst by the lexicographic rule above.
        """
        out = df_metrics.copy()

        # Mark target genomes (matched >= 1)
        if "matched_peptide_count" not in out.columns:
            raise ValueError("Missing required column: matched_peptide_count")
        out["_genomes_with_any_match"] = out["matched_peptide_count"].fillna(0).astype(int) >= 1

        # Ensure required ranking columns exist (fill missing with zeros)
        required_cols = [
            "unique_peptide_count",
            "unique_weighted_evidence",
            "weighted_evidence",
            "effective_peptide_count",
            "peptide_match_ratio",
            "matched_peptide_count",
        ]
        for c in required_cols:
            if c not in out.columns:
                out[c] = 0

        # Cast / sanitize types for stable sorting
        out["unique_peptide_count"] = pd.to_numeric(out["unique_peptide_count"], errors="coerce").fillna(0).astype(int)
        out["matched_peptide_count"] = pd.to_numeric(out["matched_peptide_count"], errors="coerce").fillna(0).astype(int)

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
            "unique_peptide_count",
            "unique_weighted_evidence",
            "weighted_evidence",
            "effective_peptide_count",
            "peptide_match_ratio",
            "matched_peptide_count",
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

    def _prepare_knockoff_pools(self, peptide_deg: Dict[str, int], N_targets_for_deg: int) -> None:
        """Build stratum pools of shared peptide contributions (w*s) for observed peptides with d(p)>1."""
        pools: Dict[Union[int, Tuple[int, int]], List[float]] = {}
        for pep, s in self.peptide_score.items():
            d = int(peptide_deg.get(pep, 0))
            if d <= 1:
                continue
            w = self._compute_weight(d=d, N=N_targets_for_deg)
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

    def _add_knockoff_existence_stats(self, df_scored: pd.DataFrame) -> pd.DataFrame:
        """Add per-genome knockoff existence p/q-values."""
        out = df_scored.copy()
        out["p_shared_knock"] = np.nan
        out["p_unique_upper"] = np.nan
        out["p_presence"] = np.nan
        out["fdr_presence"] = np.nan
        out["presence_strength"] = np.nan

        # --- NEW: knockoff null diagnostics ---
        out["null_mean_shared"] = np.nan
        out["null_sd_shared"] = np.nan
        out["null_p95_shared"] = np.nan
        out["null_p99_shared"] = np.nan
        out["z_shared"] = np.nan

        if not self.knockoff_enabled:
            return out

        if self.peptide_degeneracy is None or self.num_target_genomes_for_degeneracy is None:
            raise RuntimeError("Knockoff requires peptide_deg and N_targets_for_deg to be set.")

        if self.knockoff_pools_weighted_contrib is None:
            self._prepare_knockoff_pools(self.peptide_degeneracy, self.num_target_genomes_for_degeneracy)

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
        peptide_fdr_upper = float(min(max(self.peptide_fdr_cutoff, 1e-12), 0.5))

        target_mask = out["_genomes_with_any_match"]
        target_df = out.loc[target_mask].copy()

        if self.knockoff_top_n_targets is not None:
            topN = int(self.knockoff_top_n_targets)
            target_df = target_df.sort_values("rank_by_score", ascending=True, kind="mergesort").head(topN)

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

            unique_peptides = int(row.get("unique_peptide_count", 0))
            p_unique_upper = 1.0 if unique_peptides <= 0 else float(peptide_fdr_upper ** unique_peptides)

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

                        unique_peptides = int(row.get("unique_peptide_count", 0))
                        p_unique_upper = 1.0 if unique_peptides <= 0 else float(peptide_fdr_upper ** unique_peptides)
                        p_existence = self._fisher_p_2(p1=p_shared, p2=p_unique_upper)

                        out.at[idx, "p_shared_knock"] = p_shared
                        out.at[idx, "p_unique_upper"] = p_unique_upper
                        out.at[idx, "p_presence"] = p_existence

                        out.at[idx, "null_mean_shared"] = mu
                        out.at[idx, "null_sd_shared"] = sd
                        out.at[idx, "null_p95_shared"] = p95
                        out.at[idx, "null_p99_shared"] = p99
                        out.at[idx, "z_shared"] = (observed_shared_evidence - mu) / (sd + 1e-12)

        remaining_idx = out.index[target_mask & out["p_presence"].isna()]
        if len(remaining_idx) > 0:
            out.loc[remaining_idx, "p_shared_knock"] = 1.0
            out.loc[remaining_idx, "p_unique_upper"] = 1.0
            out.loc[remaining_idx, "p_presence"] = 1.0

        all_p = out.loc[target_mask, "p_presence"].to_numpy(dtype=float)
        out.loc[target_mask, "fdr_presence"] = self._bh_qvalues(all_p)
        pvals = pd.to_numeric(out["p_presence"], errors="coerce")
        valid = pvals.notna()
        out.loc[valid, "presence_strength"] = -np.log10(np.clip(pvals.loc[valid].to_numpy(dtype=float), 1e-300, 1.0))

        out["pass_fdr_0p01"] = (out["fdr_presence"] <= 0.01) & (out["_genomes_with_any_match"])
        out["pass_fdr_0p05"] = (out["fdr_presence"] <= 0.05) & (out["_genomes_with_any_match"])

        return out

    # =========================
    # Coverage (reference only; not used for final calling)
    # =========================
    def _add_coverage_stats(
        self,
        df_scored: pd.DataFrame,
        order_col: str = "rank_by_score",
    ) -> pd.DataFrame:
        """
        Add coverage statistics as a *human reference* (not used in q-value computation).

        Coverage is computed along the existing ranking order (default: score-based order),
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
        meta["weighting_mode"] = self.weighting_mode
        meta["idf_log_base"] = float(self.idf_log_base) if self.idf_log_base is not None else None
        meta["use_length_strata"] = bool(self.use_length_strata)
        meta["degeneracy_bin_edges"] = list(self.degeneracy_bin_edges)
        meta["peptide_length_bin_edges"] = list(self.peptide_length_bin_edges)
        meta["knockoff_enabled"] = bool(self.knockoff_enabled)
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
            if "pass_fdr_0p01" in df_scored.columns:
                meta["genomes_q_le_0p01"] = int(df_scored["pass_fdr_0p01"].fillna(False).sum())
            if "pass_fdr_0p05" in df_scored.columns:
                meta["genomes_q_le_0p05"] = int(df_scored["pass_fdr_0p05"].fillna(False).sum())
        except Exception:
            pass

        meta["timing_seconds"] = {k: float(v) for k, v in (self.timing_stats or {}).items()}

        with open(os.path.join(temp_dir, "run_summary.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

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
            if "unique_peptide_count" in target.columns:
                h2 = _hist(target.loc[target["unique_peptide_count"].fillna(0).astype(int) == 0, "p_shared_knock"], "unique0_targets")
            hs = pd.concat([h1, h2], axis=0, ignore_index=True)
            hs.to_csv(os.path.join(temp_dir, "p_shared_hist.tsv"), sep="\t", index=False)

        # --------------- q calling curve ---------------
        if "fdr_presence" in df_scored.columns and "_genomes_with_any_match" in df_scored.columns:
            target = df_scored.loc[df_scored["_genomes_with_any_match"]].copy()
            q = pd.to_numeric(target["fdr_presence"], errors="coerce")
            thresholds = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
            rows = []
            for t in thresholds:
                rows.append({"q_threshold": float(t), "n_called": int((q <= t).sum())})
            pd.DataFrame(rows).to_csv(os.path.join(temp_dir, "q_calling_curve.tsv"), sep="\t", index=False)

        # --------------- compact targets table ---------------
        if "_genomes_with_any_match" in df_scored.columns:
            target = df_scored.loc[df_scored["_genomes_with_any_match"]].copy()
            keep_cols = [
                "genome_id", "rank_by_score",
                "matched_peptide_count", "unique_peptide_count",
                "mean_degeneracy", "shared_fraction",
                "weighted_evidence", "unique_weighted_evidence", "weighted_evidence_shared",
                "p_shared_knock", "p_unique_upper", "p_presence", "presence_strength", "fdr_presence",
                "null_mean_shared", "null_sd_shared", "null_p95_shared", "null_p99_shared", "z_shared",
            ]
            keep_cols = [c for c in keep_cols if c in target.columns]
            target[keep_cols].to_csv(os.path.join(temp_dir, "targets_compact.tsv"), sep="\t", index=False)

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
        if topN > 0 and self.peptide_degeneracy is not None and self.num_target_genomes_for_degeneracy is not None:
            if "_genomes_with_any_match" in df_scored.columns:
                target = df_scored.loc[df_scored["_genomes_with_any_match"]].copy()
            else:
                target = df_scored.copy()
            if "rank_by_score" in target.columns:
                target = target.sort_values("rank_by_score", ascending=True, kind="mergesort")
            target = target.head(topN)
            out_rows = []
            N_targets = int(self.num_target_genomes_for_degeneracy)

            for _, r in target.iterrows():
                gid = str(r["genome_id"])
                peps = self.genome_matched_peptides.get(gid, set())
                if not peps:
                    continue
                for pep in peps:
                    d = int(self.peptide_degeneracy.get(pep, 1))
                    w = float(self._compute_weight(d=d, N=N_targets))
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
        genome_list: Optional[List[str]] = None,
        exclude_genome_ids: Optional[List[str]] = None,
        test_genomes_num: Optional[int] = None,
        all_matched_peptides: Optional[List[Tuple[str, Set[str], int]]] = None,
        save_matched_peptides_cache: bool = True,
        matched_peptides_cache_path: Optional[str] = None,
        compute_coverage: bool = True,
        export_temp: bool = True,
        export_peptide_contrib_topN: int = 0,
        include_null_diagnostics_in_main: bool = False,
    ) -> pd.DataFrame:
        """End-to-end analysis producing a genome-level q-value (fdr_presence)."""
        if output_tsv_path is None:
            out_dir = self.peptide_table_dir if self.peptide_table_dir else os.getcwd()
            output_tsv_path = os.path.join(out_dir, "genome_presence.tsv")
            self.logger.info(f"Output file not specified. Using: {output_tsv_path}")
        else:
            out_dir = os.path.dirname(output_tsv_path) or "."
            os.makedirs(out_dir, exist_ok=True)

        t_all0 = time.time()
        self.timing_stats = {}

        # Normalize cache path (if provided)
        cache_pkl_path: Optional[str] = None
        if matched_peptides_cache_path:
            cache_path = str(matched_peptides_cache_path)
            cache_pkl_path = cache_path if cache_path.lower().endswith(".pkl") else f"{cache_path}.pkl"

        # Prefer using existing matched-peptides cache if available.
        if all_matched_peptides is None and cache_pkl_path and os.path.exists(cache_pkl_path):
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
            all_matched_peptides = []
            futures = []
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers)
            try:
                for b in batches:
                    if len(b) == 0:
                        continue
                    futures.append(executor.submit(self._process_genome_batch, list(b)))

                for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Scanning genomes"):
                    all_matched_peptides.extend(fut.result())
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
        peptide_deg, genome_unique_counts, N_targets = self._calculate_peptide_degeneracy_and_unique_counts(all_matched_peptides)
        self.timing_stats["compute_degeneracy"] = float(time.time() - t_deg0)
        self.peptide_degeneracy = peptide_deg
        self.num_target_genomes_for_degeneracy = N_targets

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
        df_metrics = self._calculate_genome_metrics(genome_data_list, peptide_deg, N_targets)
        self.timing_stats["compute_metrics"] = float(time.time() - t_metrics0)

        self.logger.info("Ranking genomes ...")
        t_score0 = time.time()
        df_scored = self._rank_genomes(df_metrics)
        self.timing_stats["rank_genomes"] = float(time.time() - t_score0)
        df_scored["rank_by_score"] = np.arange(1, len(df_scored) + 1, dtype=int)

        if self.knockoff_enabled:
            self.logger.info("Computing per-genome existence q-values via knockoff (scheme A) ...")
            t_knock0 = time.time()
            df_scored = self._add_knockoff_existence_stats(df_scored)
            self.timing_stats["knockoff_pvalues"] = float(time.time() - t_knock0)

        if compute_coverage:
            self.logger.info("Computing coverage statistics (reference only; not used for final calling) ...")
            t_cov0 = time.time()
            df_scored = self._add_coverage_stats(df_scored, order_col="rank_by_score")
            self.timing_stats["coverage_stats"] = float(time.time() - t_cov0)

        self.genome_scores_df = df_scored

        self.logger.info(f"Saving results to: {output_tsv_path}")
        if include_null_diagnostics_in_main:
            df_to_save = df_scored
        else:
            null_diag_cols = [
                "null_mean_shared",
                "null_sd_shared",
                "null_p95_shared",
                "null_p99_shared",
                "z_shared",
            ]
            keep_cols = [c for c in df_scored.columns if c not in null_diag_cols]
            df_to_save = df_scored[keep_cols]
        df_to_save.to_csv(output_tsv_path, sep="\t", index=False)

        self.timing_stats["save_tsv"] = float(time.time() - t_all0)

        # --- NEW: export extra artifacts for paper figures ---
        if export_temp:
            try:
                stem = Path(output_tsv_path).stem
                self._export_temp_artifacts(out_dir=out_dir, stem=stem, df_scored=df_scored, export_peptide_contrib_topN=export_peptide_contrib_topN)
                self.timing_stats["export_temp"] = float(time.time() - t_all0)
            except Exception as e:
                self.logger.warning(f"Failed to export temp artifacts: {e}")

        self._print_summary()
        return self.genome_scores_df

    # =========================
    # Summary
    # =========================
    def _print_summary(self) -> None:
        if self.genome_scores_df is None or len(self.genome_scores_df) == 0:
            return

        df = self.genome_scores_df
        print("\n======= Genome existence scoring (knockoff) summary =======")
        print(f"Genomes analyzed: {len(df)}")
        print(f"Genomes with matched>=1: {int(df['_genomes_with_any_match'].sum())}")

        if "fdr_presence" in df.columns:
            keep01 = int(df["pass_fdr_0p01"].fillna(False).sum())
            keep05 = int(df["pass_fdr_0p05"].fillna(False).sum())
            print(f"Genomes q<=0.01: {keep01}")
            print(f"Genomes q<=0.05: {keep05}")

        top = df.loc[df["_genomes_with_any_match"]].head(10)
        print("\nTop 10 target genomes by rank_by_score:")
        for i, (_, r) in enumerate(top.iterrows(), 1):
            qv = r.get("fdr_presence", np.nan)
            print(
                f"{i}. {r['genome_id']} | Unique_Pep={int(r['unique_peptide_count'])}, "
                f"Matched_Pep={int(r['matched_peptide_count'])}, "
                f"FDR={qv if pd.notna(qv) else 'NA'}, "
                f"Coverage={float(r.get('cumulative_coverage_percent', 0.0)):.1f}%"
            )


# =========================
# __main__ (edit parameters here; no argparse)
# =========================
if __name__ == "__main__":
    print("\n======= Genome existence scoring (knockoff) =======")
    t0 = time.time()

    # ---- Input peptide file ----
    peptide_table_path = r"test_data/proj2/peptide_core.tsv"
    # peptide_table_path = r"test_data\sihumix\peptides.tsv"
    # peptide_table_path = r"test_data\proj1\peptides_all.tsv"
    # peptide_table_path = r"test_data\proj1\peptides_v48_PBS.tsv"
    # peptide_table_path = r"test_data\6bacteria\peptides.tsv"
    # peptide_table_path = r"test_data\mix24x\peptides.tsv"

    peptide_seq_col = "Sequence"
    peptide_score_col = "Score"          # set to None if not available
    peptide_fdr_col = "FDR"              # set to None if not available
    peptide_fdr_cutoff = 0.05

    # peptide-level decoy flag (optional)
    peptide_decoy_flag_col = "Reverse"       # set to None if not available
    decoy_flag_value = "+"

    # ---- Genome peptide folders (theoretical digests) ----
    # You only need TARGET folders for knockoff.
    genome_digest_dirs = [
        r"C:/Users/max/Desktop/digested_genomes/UHGP_digested",          # target digest peptides
        # r"test_data\mix24x\Mix24_digested",  # target digest peptides
        # r'test_data\sihumix\digested',  # target digest peptides
        # r'test_data\6bacteria\genomes\faa_digested'
        
    ]

    # ---- Output ----
    # output_tsv_path = r"test_data/proj1/genome_presence_proj1.tsv"
    output_tsv_path = r"test_data/proj2/genome_presence_proj2.tsv"
    # output_tsv_path = r"test_data/mix24x/genome_presence_mix24x.tsv"
    # output_tsv_path = r"test_data/mix24x/genome_presence_mix24x_only_UHGP.tsv"
    # output_tsv_path = r"test_data\6bacteria\genome_presence_6bacteria.tsv"
    # output_tsv_path = r"test_data\sihumix\genome_presence_sihumix_only_UHGP.tsv"
    
    out_dir = os.path.dirname(output_tsv_path) or "."
    pickle_path = os.path.join(out_dir, "matched_peptides.pkl")  # set to e.g. r"test_data\6bacteria\matched_peptides_cache.pkl" to save/load matched peptides cache (speeds up repeated runs)
    # pickle_path = None  # set to None to disable matched peptides caching
    
    
    # ---- Optional: exclude list ----
    exclude_genome_ids = []
    exclude_list_path = r"test_data/removed_genomes.txt"
    if os.path.exists(exclude_list_path):
        with open(exclude_list_path, "r") as f:
            exclude_genome_ids = [x.strip() for x in f if x.strip()]
    
    # exclude_genome_ids += ['MGYG000002386'] # for Sihumix
    # exclude_genome_ids += ['MGYG000002506', 'MGYG000002463', 'MGYG000002337', 'MGYG000000109', 'MGYG000003394'] # for 6bacteria
    # exclude_genome_ids += ['MGYG000002331'] # for Mix24x
    
    # ---- Calculator ----
    calc = GenomePresenceScorer(
        num_workers=min(32, max(1, (mp.cpu_count() or 1) - 1))
    )

    # Weighting
    calc.weighting_mode = "inverse_degeneracy"  # or "idf"

    # Knockoff tuning
    calc.knockoff_enabled = True
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
        peptide_fdr_col=peptide_fdr_col,
        peptide_fdr_cutoff=peptide_fdr_cutoff,
    )

    # Run
    calc.analyze_genomes(
        genome_digest_dirs=genome_digest_dirs,
        output_tsv_path=output_tsv_path,
        exclude_genome_ids=exclude_genome_ids,
        all_matched_peptides=None,
        save_matched_peptides_cache=True,
        matched_peptides_cache_path=pickle_path,
    )

    print(f"\nDone. Elapsed: {time.time() - t0:.1f}s")
