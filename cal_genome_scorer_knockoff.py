# Genome existence scoring from a peptide list using peptide-space knockoff null
# Version: 4.0 
# Date: 2026-02-09
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
# - genome_score: unified rank score (lexicographic; unique dominates).
# - fdr_presence: recommended per-genome existence q-value.

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
class GenomeScoreCalculator:
    """
    Genome-level existence scoring from an observed peptide table by mapping onto per-genome theoretical peptide files.

    Clean v4.0:
    - Shared peptide degeneracy d(p) is computed across TARGET genomes (recommended).
    - A peptide-space knockoff null is used to estimate a per-genome existence p/q-value without requiring
      any second database matching.
    """

    def __init__(self, num_workers: Optional[int] = None, log_file: Optional[str] = None):
        self.num_workers = num_workers if num_workers is not None else max(1, (mp.cpu_count() or 1) - 1)
        self.logger = setup_logger("GenomeScoreCalculator", log_file)

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

        # Optional speed knob: compute knockoff only for top-N TARGET genomes by genome_score, set others p=1
        self.knockoff_top_n_targets: Optional[int] = None

        # Internal caches for knockoff
        self.peptide_degeneracy: Optional[Dict[str, int]] = None
        self.num_target_genomes_for_degeneracy: Optional[int] = None
        self.knockoff_pools_weighted_contrib: Optional[Dict[Union[int, Tuple[int, int]], np.ndarray]] = None
        self.knockoff_shared_stratum_counts_by_genome: Dict[str, Counter] = {}  # genome_id -> Counter(stratum -> count)

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

        if peptide_decoy_flag_col and peptide_decoy_flag_col in df.columns:
            before = len(df)
            df = df[(df[peptide_decoy_flag_col] != decoy_flag_value) | (df[peptide_decoy_flag_col].isna())]
            self.logger.info(f"Peptide-level decoy filter: {before} -> {len(df)} rows.")

        if peptide_fdr_col and peptide_fdr_col in df.columns:
            before = len(df)
            df[peptide_fdr_col] = pd.to_numeric(df[peptide_fdr_col], errors="coerce")
            df = df[df[peptide_fdr_col] <= float(peptide_fdr_cutoff)]
            self.logger.info(f"Peptide-level FDR filter (<= {peptide_fdr_cutoff}): {before} -> {len(df)} rows.")

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
    # Unified integer score for ranking
    # =========================
    def _build_genome_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build lexicographically-weighted integer genome_score for unified ranking."""
        out = df.copy()
        out["matched_peptide_count"] = pd.to_numeric(out["matched_peptide_count"], errors="coerce").fillna(0).astype(np.int64)
        out["_genomes_with_any_match"] = out["matched_peptide_count"] >= 1

        out["unique_peptide_count"] = pd.to_numeric(out["unique_peptide_count"], errors="coerce").fillna(0).astype(np.int64)

        for col in ["unique_weighted_evidence", "weighted_evidence", "effective_peptide_count", "peptide_match_ratio"]:
            if col not in out.columns:
                out[col] = 0.0
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype(float)

        out["peptide_match_ratio"] = out["peptide_match_ratio"].clip(0.0, 1.0)

        S = self.rank_lexico_scales
        U_SCALE = int(S["U"])
        UW_SCALE = int(S["UW"])
        WE_SCALE = int(S["WE"])
        EP_SCALE = int(S["EP"])
        MR_SCALE = int(S["MR"])
        M_SCALE = int(S["M"])

        genome_score = (
            out["unique_peptide_count"] * U_SCALE
            + np.rint(out["unique_weighted_evidence"] * UW_SCALE).astype(np.int64)
            + np.rint(out["weighted_evidence"] * WE_SCALE).astype(np.int64)
            + np.rint(out["effective_peptide_count"] * EP_SCALE).astype(np.int64)
            + np.rint(out["peptide_match_ratio"] * MR_SCALE).astype(np.int64)
            + out["matched_peptide_count"] * M_SCALE
        )

        out["genome_score"] = genome_score.where(out["_genomes_with_any_match"], -1).astype(np.int64)
        return out

    # =========================
    # Knockoff (scheme A)
    # =========================
    def _knock_deg_bin(self, d: int) -> int:
        thr = self.degeneracy_bin_edges
        if d <= thr[0]:
            return 0
        if d <= thr[1]:
            return 1
        if d <= thr[2]:
            return 2
        if d <= thr[3]:
            return 3
        if d <= thr[4]:
            return 4
        return 5

    def _knock_len_bin(self, pep_len: int) -> int:
        thr = self.peptide_length_bin_edges
        if pep_len <= thr[0]:
            return 0
        if pep_len <= thr[1]:
            return 1
        if pep_len <= thr[2]:
            return 2
        if pep_len <= thr[3]:
            return 3
        if pep_len <= thr[4]:
            return 4
        return 5

    def _knock_stratum(self, d: int, pep_len: int) -> Union[int, Tuple[int, int]]:
        db = self._knock_deg_bin(d)
        if not self.use_length_strata:
            return db
        lb = self._knock_len_bin(pep_len)
        return (db, lb)

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

    def _mc_sum_from_pool(self, pool: np.ndarray, K: int, c: int, rng: np.random.Generator) -> np.ndarray:
        """Sample c items (with replacement) from pool, repeat K times, return K sums (blocking)."""
        out = np.zeros(K, dtype=np.float64)
        if c <= 0:
            return out
        if pool is None or pool.size == 0:
            return out

        remaining = int(c)
        block = int(max(1, self.knockoff_sample_block_size))
        while remaining > 0:
            b = min(remaining, block)
            draws = rng.choice(pool, size=(K, b), replace=True)
            out += draws.sum(axis=1)
            remaining -= b
        return out

    def _p_shared_knockoff_mc(self, gid: str, obs_shared_score: float, K: int, rng: np.random.Generator) -> float:
        """Empirical p-value for shared evidence via knockoff Monte Carlo."""
        counts = self.knockoff_shared_stratum_counts_by_genome.get(gid, None)
        if not counts:
            return 1.0

        null_sum = np.zeros(K, dtype=np.float64)
        for key, c in counts.items():
            pool = self.knockoff_pools_weighted_contrib.get(key, None) if self.knockoff_pools_weighted_contrib else None
            null_sum += self._mc_sum_from_pool(pool=pool, K=K, c=int(c), rng=rng)

        ge = float(np.sum(null_sum >= float(obs_shared_score)))
        return (1.0 + ge) / (1.0 + float(K))

    @staticmethod
    def _fisher_p_2(p1: float, p2: float) -> float:
        """Fisher combine 2 p-values without scipy (df=4)."""
        p1 = float(min(max(p1, 1e-300), 1.0))
        p2 = float(min(max(p2, 1e-300), 1.0))
        p_prod = p1 * p2
        return float(min(max(p_prod * (1.0 - np.log(p_prod)), 0.0), 1.0))

    @staticmethod
    def _bh_qvalues(pvals: np.ndarray) -> np.ndarray:
        """Benjamini-Hochberg q-values (monotone)."""
        p = np.asarray(pvals, dtype=float)
        m = p.size
        if m == 0:
            return p
        order = np.argsort(p)
        ranked = p[order]
        q = ranked * (m / (np.arange(1, m + 1, dtype=float)))
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
        rng_stage1 = np.random.default_rng(ss.spawn(1)[0])
        rng_stage2 = np.random.default_rng(ss.spawn(1)[0])

        K1 = int(max(50, self.knockoff_mc_iterations))
        K2 = None
        if self.knockoff_stage2_mc_iterations is not None:
            K2 = int(max(50, self.knockoff_stage2_mc_iterations))
        peptide_fdr_upper = float(min(max(self.peptide_fdr_cutoff, 1e-12), 0.5))

        target_mask = out["_genomes_with_any_match"]
        target_df = out.loc[target_mask].copy()

        if self.knockoff_top_n_targets is not None:
            topN = int(self.knockoff_top_n_targets)
            target_df = target_df.sort_values("genome_score", ascending=False, kind="mergesort").head(topN)

        # -----------------
        # Stage 1 (fast screen)
        # -----------------
        for idx, row in tqdm(
            target_df.iterrows(), total=len(target_df), desc=f"Knockoff p-values stage1 (K={K1})"
        ):
            genome_id = row["genome_id"]
            observed_shared_evidence = float(row.get("weighted_evidence_shared", 0.0))
            p_shared = self._p_shared_knockoff_mc(
                gid=genome_id,
                obs_shared_score=observed_shared_evidence,
                K=K1,
                rng=rng_stage1,
            )

            unique_peptides = int(row.get("unique_peptide_count", 0))
            p_unique_upper = 1.0 if unique_peptides <= 0 else float(peptide_fdr_upper ** unique_peptides)

            p_existence = self._fisher_p_2(p1=p_shared, p2=p_unique_upper)

            out.at[idx, "p_shared_knock"] = p_shared
            out.at[idx, "p_unique_upper"] = p_unique_upper
            out.at[idx, "p_presence"] = p_existence

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
                        p_shared = self._p_shared_knockoff_mc(
                            gid=genome_id,
                            obs_shared_score=observed_shared_evidence,
                            K=K2,
                            rng=rng_stage2,
                        )

                        unique_peptides = int(row.get("unique_peptide_count", 0))
                        p_unique_upper = 1.0 if unique_peptides <= 0 else float(peptide_fdr_upper ** unique_peptides)
                        p_existence = self._fisher_p_2(p1=p_shared, p2=p_unique_upper)

                        out.at[idx, "p_shared_knock"] = p_shared
                        out.at[idx, "p_unique_upper"] = p_unique_upper
                        out.at[idx, "p_presence"] = p_existence

        remaining_idx = out.index[target_mask & out["p_presence"].isna()]
        if len(remaining_idx) > 0:
            out.loc[remaining_idx, "p_shared_knock"] = 1.0
            out.loc[remaining_idx, "p_unique_upper"] = 1.0
            out.loc[remaining_idx, "p_presence"] = 1.0

        all_p = out.loc[target_mask, "p_presence"].to_numpy(dtype=float)
        out.loc[target_mask, "fdr_presence"] = self._bh_qvalues(all_p)

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
            # Fallback: derive order by genome_score
            tmp = out.sort_values("genome_score", ascending=False, kind="mergesort").reset_index()
            order_idx = tmp["index"].to_numpy()
        else:
            tmp = out.sort_values(order_col, ascending=True, kind="mergesort").reset_index()
            order_idx = tmp["index"].to_numpy()

        # Determine denominator: total matchable peptides
        if self.peptide_degeneracy is not None:
            total_matchable = int(len(self.peptide_degeneracy))
        else:
            # Robust fallback: union over all genomes
            total_matchable = 0
            seen = set()
            for gid, ps in self.genome_matched_peptides.items():
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
    ) -> pd.DataFrame:
        """End-to-end analysis producing a genome-level q-value (fdr_presence)."""
        if output_tsv_path is None:
            out_dir = self.peptide_table_dir if self.peptide_table_dir else os.getcwd()
            output_tsv_path = os.path.join(out_dir, "genome_scores_knockoff.tsv")
            self.logger.info(f"Output file not specified. Using: {output_tsv_path}")
        else:
            out_dir = os.path.dirname(output_tsv_path) or "."
            os.makedirs(out_dir, exist_ok=True)

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

        self.genome_matched_peptides = {}
        self.genome_total_theoretical_peptides = {}
        obs_set = set(self.peptide_score.keys())

        for genome_id, matched_peptides, total_cnt in all_matched_peptides:
            matched_peptides = set(matched_peptides).intersection(obs_set)
            self.genome_matched_peptides.setdefault(genome_id, set()).update(matched_peptides)
            prev = self.genome_total_theoretical_peptides.get(genome_id, 0)
            self.genome_total_theoretical_peptides[genome_id] = max(prev, int(total_cnt))

        peptide_deg, genome_unique_counts, N_targets = self._calculate_peptide_degeneracy_and_unique_counts(all_matched_peptides)
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
        df_metrics = self._calculate_genome_metrics(genome_data_list, peptide_deg, N_targets)

        self.logger.info("Building genome_score ...")
        df_scored = self._build_genome_score(df_metrics)
        df_scored = df_scored.sort_values("genome_score", ascending=False, kind="mergesort").reset_index(drop=True)
        df_scored["rank_by_score"] = np.arange(1, len(df_scored) + 1, dtype=int)

        if self.knockoff_enabled:
            self.logger.info("Computing per-genome existence q-values via knockoff (scheme A) ...")
            df_scored = self._add_knockoff_existence_stats(df_scored)

        if compute_coverage:
            self.logger.info("Computing coverage statistics (reference only; not used for final calling) ...")
            df_scored = self._add_coverage_stats(df_scored, order_col="rank_by_score")

        self.genome_scores_df = df_scored

        self.logger.info(f"Saving results to: {output_tsv_path}")
        df_scored.to_csv(output_tsv_path, sep="\t", index=False)

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
        print("\nTop 10 target genomes by genome_score:")
        for i, (_, r) in enumerate(top.iterrows(), 1):
            qv = r.get("fdr_presence", np.nan)
            print(
                f"{i}. {r['genome_id']} | Unique_peptides={int(r['unique_peptide_count'])}, "
                f"Matched_peptides={int(r['matched_peptide_count'])}, "
                f"Weighted_evidence={float(r['weighted_evidence']):.2f}, "
                f"Weighted_evidence_shared={float(r.get('weighted_evidence_shared', 0.0)):.2f}, "
                f"FDR={qv if pd.notna(qv) else 'NA'}"
            )


# =========================
# __main__ (edit parameters here; no argparse)
# =========================
if __name__ == "__main__":
    print("\n======= Genome existence scoring (knockoff) =======")
    t0 = time.time()

    # ---- Input peptide file ----
    # peptide_table_path = r"test_data/proj2/peptide_core.tsv"
    # peptide_table_path = r"test_data\sihumix\peptides.tsv"
    # peptide_table_path = r"test_data\proj1\peptides_all.tsv"
    # peptide_table_path = r"test_data\proj1\peptides_v48_PBS.tsv"
    # peptide_table_path = r"test_data\6bacteria\peptides.tsv"
    peptide_table_path = r"test_data\mix24x\peptides.tsv"

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
    # output_tsv_path = r"test_data/proj1/genome_scores_knockoff_proj1.tsv"
    # output_tsv_path = r"test_data/proj2/genome_scores_knockoff.tsv"
    # output_tsv_path = r"test_data/mix24x/genome_scores_knockoff_mix24x.tsv"
    output_tsv_path = r"test_data/mix24x/genome_scores_knockoff_mix24x_only_UHGP.tsv"
    # output_tsv_path = r"test_data\6bacteria\genome_scores_knockoff_6bacteria.tsv"
    # output_tsv_path = r"test_data\sihumix\genome_scores_knockoff_sihumix_only_UHGP.tsv"
    
    out_dir = os.path.dirname(output_tsv_path) or "."
    pickle_path = os.path.join(out_dir, "matched_peptides.pkl")  # set to e.g. r"test_data\6bacteria\matched_peptides_cache.pkl" to save/load matched peptides cache (speeds up repeated runs)

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
    calc = GenomeScoreCalculator(
        num_workers=min(32, max(1, (mp.cpu_count() or 1) - 1))
    )

    # Weighting
    calc.weighting_mode = "inverse_degeneracy"  # or "idf"

    # Knockoff tuning
    calc.knockoff_enabled = True
    # Two-stage MC (optional):
    # Stage 1: K1 for all genomes (fast screen)
    # Stage 2: K2 only for genomes with stage-1 p_presence in specified ranges
    calc.knockoff_mc_iterations = 500
    calc.knockoff_stage2_mc_iterations = 2000
    calc.knockoff_stage2_p_exist_ranges = [(0.005, 0.02), (0.02, 0.08)]
    calc.knockoff_random_seed = 1
    calc.knockoff_top_n_targets = None   # set e.g. 5000 for speed on very large datasets

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
        matched_peptides_cache_path=None,
    )

    print(f"\nDone. Elapsed: {time.time() - t0:.1f}s")
