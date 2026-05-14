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
        self.unique_pvalue_mode: str = "adaptive-exact"
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
        self.unit_aware_enabled: bool = False
        self.unit_presence_rule: str = "union"
        self.unit_shared_mode: str = "none"
        self.unit_sample_ids: List[str] = []
        self.unit_analysis_unit_ids: List[str] = []
        self.unit_peptides: List[str] = []
        self.unit_peptide_index: Dict[str, int] = {}
        self.unit_presence_matrix = None  # sparse peptide x analysis_unit matrix
        self.unit_sample_counts: Dict[str, int] = {}
        self.sample_unit_mapping_df: Optional[pd.DataFrame] = None
        self.unit_aware_output_paths: Dict[str, str] = {}
        self.unit_aware_cohort_summary_df: Optional[pd.DataFrame] = None

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
                "scipy is required for adaptive-exact unique p-values. "
                "Install scipy or use upper-bound or peptide-column mode."
            ) from exc

    def _unique_pvalue_stats_for_genome(self, gid: str, u_observed: int) -> dict:
        U = int(u_observed)
        mode = str(self.unique_pvalue_mode or "adaptive-exact").strip().lower()
        alpha = float(min(max(self.single_peptide_error_rate_upper_bound, 1e-12), 1.0))
        p_unique = 1.0
        p_unique_depth = 1.0
        S = int(self.observed_unique_peptide_pool_size)
        expected = 0.0
        fold = 0.0
        gate_pass = False
        null_model = ""
        theoretical_unique: Optional[int] = None

        if mode == "adaptive-exact":
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
        elif mode == "upper-bound":
            p_unique = float(alpha ** max(U, 0)) if U > 0 else 1.0
            p_unique_depth = p_unique
            gate_pass = U > 0
        elif mode == "peptide-column":
            matched = self.genome_matched_peptides.get(gid, set())
            uniq = [p for p in matched if int((self.peptide_degeneracy or {}).get(p, 1)) == 1]
            if uniq:
                errs = [self.peptide_error_upper_by_peptide.get(p, alpha) for p in uniq]
                p_unique = float(np.prod(np.clip(errs, 1e-12, 1.0)))
                p_unique_depth = p_unique
                gate_pass = True
        else:
            raise ValueError(
                "unique_pvalue_mode must be one of 'adaptive-exact', 'upper-bound', or 'peptide-column'."
            )

        return {
            "p_unique": _clip_pvalue(p_unique),
            "p_unique_depth": _clip_pvalue(p_unique_depth),
            "unique_observed": int(U),
            "unique_expected_null": float(expected),
            "unique_depth_fold": float(fold),
            "unique_depth_null_model": null_model,
            "unique_pvalue_mode": mode,
            "unique_gate_pass": bool(gate_pass),
            "theoretical_unique_peptides": theoretical_unique,
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
        intensity_col: str = "Precursor.Quantity",
        peptide_error_col: Optional[str] = "Q.Value",
        peptide_error_cutoff: float = 0.05,
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
            optional_cols = [error_col] if error_col else []
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
            columns_to_read = list(dict.fromkeys(required_cols + ([error_col] if error_col else [])))
            dtype = {sample_col: "string", seq_col: "string"}
            df = pd.read_csv(peptide_file_path, sep=sep, usecols=columns_to_read, dtype=dtype, engine="c")

        if error_col and error_col not in df.columns:
            self.logger.warning(
                f"Error column '{error_col}' not found; skipping peptide-level error filtering for unit-aware input."
            )
            error_col = None

        self.run_stats["unit_aware"] = True
        self.run_stats["unit_aware_peptide_rows_loaded"] = int(len(df))
        self.run_stats["unit_aware_sample_id_col"] = sample_col
        self.run_stats["unit_aware_peptide_seq_col"] = seq_col
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
        df[intensity_col] = pd.to_numeric(df[intensity_col], errors="coerce")
        if error_col:
            df[error_col] = pd.to_numeric(df[error_col], errors="coerce")

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
        self.unit_shared_mode = "none"
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
        self.peptide_score = {peptide: 1.0 for peptide in peptide_list}

        self.peptide_error_cutoff = float(peptide_error_cutoff)
        self.single_peptide_error_rate_upper_bound = float(
            min(max(float(peptide_error_cutoff), 1e-12), 1.0)
        )
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
        unique_pvalue_mode: str = "adaptive-exact",
        min_unique_for_unique_pvalue: int = 3,
    ) -> pd.DataFrame:
        """Add per-genome knockoff existence p/q-values."""
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
        mode = str(unique_pvalue_mode or "adaptive-exact").strip().lower()
        if int(min_unique_for_unique_pvalue) < 0:
            raise ValueError("min_unique_for_unique_pvalue must be >= 0.")
        if mode not in {"adaptive-exact", "upper-bound", "peptide-column"}:
            raise ValueError(
                "unique_pvalue_mode must be one of 'adaptive-exact', 'upper-bound', or 'peptide-column', "
                f"got {unique_pvalue_mode!r}."
            )
        self.unique_pvalue_mode = mode
        self.min_unique_for_unique_pvalue = int(min_unique_for_unique_pvalue)
        error_col = self.run_stats.get("peptide_error_col", None)
        has_per_peptide_error = bool(self.peptide_error_upper_by_peptide)
        use_per_peptide_error = bool(mode == "peptide-column" and has_per_peptide_error)
        uses_pep_column = bool(use_per_peptide_error and isinstance(error_col, str) and error_col.upper() == "PEP")
        source_col_display = str(error_col) if error_col is not None else "none"
        self.run_stats["unique_pvalue_mode"] = mode
        self.run_stats["min_unique_for_unique_pvalue"] = int(min_unique_for_unique_pvalue)
        self.run_stats["unique_pvalue_uses_per_peptide_error"] = bool(use_per_peptide_error)
        self.run_stats["unique_pvalue_error_source_col"] = str(error_col) if use_per_peptide_error and error_col is not None else None
        self.run_stats["unique_pvalue_uses_pep_column"] = bool(uses_pep_column)
        if mode == "adaptive-exact":
            self.logger.info(
                "Unique p-value mode: [adaptive-exact] "
                f"null_model=hypergeometric, min_unique={int(min_unique_for_unique_pvalue)}, "
                f"observed_unique_pool={int(self.observed_unique_peptide_pool_size)}, "
                f"theoretical_unique_universe={int(self.total_theoretical_unique_peptides_all_genomes)}"
            )
        elif use_per_peptide_error:
            self.logger.info(
                f"Unique p-value mode: {'[PEP]' if uses_pep_column else '[per-peptide error column]'} "
                f"source_col='{source_col_display}', "
                f"per_peptide_error_n={len(self.peptide_error_upper_by_peptide)}, "
                f"global_fallback={peptide_error_upper:.4g}"
            )
        else:
            reason = "upper_bound_mode" if mode == "upper-bound" else "per_peptide_error_not_available"
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
        self.unit_shared_mode = "none"

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

        lineage_map: Dict[str, object] = {}
        if "Lineage" in df_scored.columns:
            lineage_map = dict(zip(df_scored["genome_id"].astype(str), df_scored["Lineage"]))

        a_total = int(self.total_theoretical_unique_peptides_all_genomes)
        if a_total <= 0:
            raise ValueError("Unit-aware adaptive-exact p-values require theoretical unique peptide opportunity.")

        rows = []
        gate_min = int(min_unique_for_unique_pvalue)
        unit_log_interval = max(1, n_units // 10)
        self.logger.info(
            f"Computing unit-aware adaptive-exact p/q values for {n_units} unit(s) x {n_genomes} genome(s) ..."
        )
        for unit_idx, unit_id in enumerate(self.unit_analysis_unit_ids):
            if unit_idx == 0 or (unit_idx + 1) % unit_log_interval == 0 or unit_idx + 1 == n_units:
                self.logger.info(
                    f"Unit-aware p/q progress: {unit_idx + 1}/{n_units} analysis unit(s) processed."
                )
            matched_counts = np.asarray(matched_matrix[:, unit_idx].todense()).ravel().astype(int)
            unique_counts = np.asarray(unique_matrix[:, unit_idx].todense()).ravel().astype(int)
            observed_unique_pool_size = int(unique_counts.sum())
            pvals = np.ones(n_genomes, dtype=float)
            expected_values = np.zeros(n_genomes, dtype=float)
            fold_values = np.zeros(n_genomes, dtype=float)
            gate_values = np.zeros(n_genomes, dtype=bool)

            for genome_idx, genome_id in enumerate(genome_ids):
                observed_unique = int(unique_counts[genome_idx])
                theoretical_unique = int(self.genome_theoretical_unique_peptides.get(genome_id, 0))
                expected = (
                    float(observed_unique_pool_size) * float(theoretical_unique) / float(a_total)
                    if a_total > 0
                    else 0.0
                )
                gate_pass = bool(observed_unique >= gate_min)
                if gate_pass and observed_unique_pool_size > 0 and theoretical_unique > 0:
                    p_unique = self._hypergeom_tail_pvalue(
                        observed=observed_unique,
                        universe_size=a_total,
                        success_states=theoretical_unique,
                        draws=observed_unique_pool_size,
                    )
                else:
                    p_unique = 1.0

                pvals[genome_idx] = _clip_pvalue(p_unique)
                expected_values[genome_idx] = float(expected)
                fold_values[genome_idx] = float(observed_unique) / max(float(expected), 1e-12)
                gate_values[genome_idx] = gate_pass

            qvals = self._bh_qvalues(pvals)
            presence_scores = -np.log10(np.clip(qvals, 1e-300, 1.0))
            rank_order = np.lexsort((np.asarray(genome_ids, dtype=object), -matched_counts, -unique_counts, qvals))
            ranks = np.empty(n_genomes, dtype=int)
            ranks[rank_order] = np.arange(1, n_genomes + 1, dtype=int)

            n_samples = int(self.unit_sample_counts.get(unit_id, 0))
            for genome_idx, genome_id in enumerate(genome_ids):
                rows.append(
                    {
                        "analysis_unit_id": unit_id,
                        "genome_id": genome_id,
                        "Lineage": lineage_map.get(genome_id, pd.NA),
                        "num_peptides_matched": int(matched_counts[genome_idx]),
                        "num_peptides_unique": int(unique_counts[genome_idx]),
                        "theoretical_unique_peptides": int(self.genome_theoretical_unique_peptides.get(genome_id, 0)),
                        "observed_unique_peptide_pool_size": int(observed_unique_pool_size),
                        "expected_unique_null": float(expected_values[genome_idx]),
                        "unique_depth_fold": float(fold_values[genome_idx]),
                        "unique_gate_pass": bool(gate_values[genome_idx]),
                        "pvalue_unique": float(pvals[genome_idx]),
                        "pvalue_shared": 1.0,
                        "pvalue": float(pvals[genome_idx]),
                        "qvalue": float(qvals[genome_idx]),
                        "presence_score": float(presence_scores[genome_idx]),
                        "presence_rank": int(ranks[genome_idx]),
                        "pass_q_0_01": bool(qvals[genome_idx] <= 0.01 and matched_counts[genome_idx] >= 1),
                        "pass_q_0_05": bool(qvals[genome_idx] <= 0.05 and matched_counts[genome_idx] >= 1),
                        "n_samples_in_unit": int(n_samples),
                        "unit_presence_rule": "union",
                        "unit_shared_mode": "none",
                    }
                )

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
        self.run_stats["unit_aware_unique_pvalue_mode"] = "adaptive-exact"
        self.run_stats["unit_aware_presence_rule"] = "union"
        self.run_stats["unit_aware_shared_mode"] = "none"
        return unit_level_df, cohort_summary_df

    def _export_unit_aware_outputs(
        self,
        out_dir: str,
        stem: str,
        unit_level_df: pd.DataFrame,
        cohort_summary_df: pd.DataFrame,
        mapping_df: pd.DataFrame,
    ) -> None:
        """Write unit-aware genome presence outputs next to the main result table."""
        os.makedirs(out_dir or ".", exist_ok=True)
        unit_level_path = os.path.join(out_dir, f"{stem}_unit_level_genome_presence.tsv")
        cohort_path = os.path.join(out_dir, f"{stem}_cohort_genome_presence_summary.tsv")
        mapping_path = os.path.join(out_dir, f"{stem}_sample_unit_mapping.tsv")
        self.unit_aware_output_paths = {
            "unit_level": unit_level_path,
            "cohort_summary": cohort_path,
            "sample_unit_mapping": mapping_path,
        }
        self.unit_aware_cohort_summary_df = cohort_summary_df.copy()
        unit_level_df.to_csv(unit_level_path, sep="\t", index=False)
        cohort_summary_df.to_csv(cohort_path, sep="\t", index=False)
        mapping_df.to_csv(mapping_path, sep="\t", index=False)
        self.logger.info(f"Saved unit-aware genome presence table: {unit_level_path}")
        self.logger.info(f"Saved unit-aware cohort summary: {cohort_path}")
        self.logger.info(f"Saved sample-unit mapping: {mapping_path}")

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
        unique_pvalue_mode: str = "adaptive-exact",
        min_unique_for_unique_pvalue: int = 3,
        theoretical_opportunity_cache_path: Optional[str] = None,
        rebuild_theoretical_opportunity_cache: bool = False,
        num_workers_for_theoretical_opportunity: Optional[int] = None,
        return_full_table: bool = False,
        unit_aware: bool = False,
    ) -> pd.DataFrame:
        """End-to-end analysis producing a genome-level q-value (q_presence)."""
        mode = str(unique_pvalue_mode or "adaptive-exact").strip().lower()
        if int(min_unique_for_unique_pvalue) < 0:
            raise ValueError("min_unique_for_unique_pvalue must be >= 0.")
        if mode not in {"adaptive-exact", "upper-bound", "peptide-column"}:
            raise ValueError(
                "unique_pvalue_mode must be one of 'adaptive-exact', 'upper-bound', or 'peptide-column'."
            )
        if unit_aware:
            if not self.unit_aware_enabled:
                raise ValueError("unit_aware=True requires read_unit_aware_peptide_file() before analyze_genomes().")
            self.unit_presence_rule = "union"
            self.unit_shared_mode = "none"
        self.unique_pvalue_mode = mode
        self.min_unique_for_unique_pvalue = int(min_unique_for_unique_pvalue)
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
        if mode == "adaptive-exact" or unit_aware:
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
                    "Adaptive-exact unique p-values require digest TSV files for all selected genomes. "
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
        if mode == "adaptive-exact":
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
            min_unique_for_unique_pvalue=int(min_unique_for_unique_pvalue),
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
                mode="adaptive-exact",
                min_unique_for_unique_pvalue=int(min_unique_for_unique_pvalue),
            )
            mapping_df = self.sample_unit_mapping_df if self.sample_unit_mapping_df is not None else pd.DataFrame()
            self._export_unit_aware_outputs(
                out_dir=out_dir,
                stem=stem,
                unit_level_df=unit_level_df,
                cohort_summary_df=cohort_summary_df,
                mapping_df=mapping_df,
            )
            self.timing_stats["unit_aware_outputs"] = float(time.time() - t_unit0)

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
        ])
        if mode == "adaptive-exact":
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
            if self.unit_aware_output_paths:
                print(f"Pooled genomes q<=0.01: {keep01}")
                print(f"Pooled genomes q<=0.05: {keep05}")
            else:
                print(f"Genomes q<=0.01: {keep01}")
                print(f"Genomes q<=0.05: {keep05}")

        if self.unit_aware_output_paths:
            print("\nUnit-aware outputs written:")
            for key in ("unit_level", "cohort_summary", "sample_unit_mapping"):
                path = self.unit_aware_output_paths.get(key)
                if path:
                    print(f"  {path}")

            cohort = self.unit_aware_cohort_summary_df
            if cohort is not None and not cohort.empty:
                n_units = int(pd.to_numeric(cohort["n_units_tested"], errors="coerce").fillna(0).max())
                q05_units = pd.to_numeric(cohort["n_units_q_le_0_05"], errors="coerce").fillna(0).astype(int)
                tested_units = pd.to_numeric(cohort["n_units_tested"], errors="coerce").fillna(0).astype(int)
                print(f"Analysis units: {n_units}")
                print(f"Genomes q<=0.05 in >=1 unit: {int((q05_units >= 1).sum())}")
                print(f"Genomes q<=0.05 in all units: {int(((tested_units > 0) & (q05_units == tested_units)).sum())}")

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


if __name__ == "__main__":
    if __package__ in {None, ""}:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from metaumbra.cli import main as cli_main
    else:
        from .cli import main as cli_main
    raise SystemExit(cli_main(["score", *sys.argv[1:]]))
