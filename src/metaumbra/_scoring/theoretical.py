"""Theoretical digest readers and ProcessPool worker helpers."""

import hashlib
import os
from collections import Counter
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Union

import pandas as pd

from .normalization import (
    DEFAULT_PEPTIDE_NORMALIZATION_POLICY,
    normalize_peptide_collection,
    normalize_peptide_policy,
)


_OBS_PEPTIDES_WORKER: Union[Set[str], FrozenSet[str]] = set()
_PEPTIDE_NORMALIZATION_POLICY_WORKER = DEFAULT_PEPTIDE_NORMALIZATION_POLICY
_AA_ONLY_PATTERN = r"^[ACDEFGHIKLMNPQRSTVWYBXZJUO]+$"


def _init_genome_batch_worker(
    obs_peptides: Union[Set[str], FrozenSet[str]],
    peptide_normalization_policy: str = DEFAULT_PEPTIDE_NORMALIZATION_POLICY,
) -> None:
    """ProcessPool initializer: set read-only observed peptide universe once per worker."""
    global _OBS_PEPTIDES_WORKER, _PEPTIDE_NORMALIZATION_POLICY_WORKER
    _OBS_PEPTIDES_WORKER = obs_peptides
    _PEPTIDE_NORMALIZATION_POLICY_WORKER = normalize_peptide_policy(
        peptide_normalization_policy
    )


def _read_unique_peptides_from_digest(
    genome_peptides_path: Union[str, os.PathLike],
    peptide_normalization_policy: str = DEFAULT_PEPTIDE_NORMALIZATION_POLICY,
) -> Set[str]:
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

        chunk_unique = normalize_peptide_collection(
            col_series[col_series != ""].values.tolist(),
            peptide_normalization_policy,
        )
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
    peptide_normalization_policy: str = DEFAULT_PEPTIDE_NORMALIZATION_POLICY,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Read digest TSV files and write peptide/genome pairs into local shard files."""
    genome_total_theoretical_peptides: Dict[str, int] = {}
    shard_paths: Dict[int, str] = {}
    open_handles: Dict[int, object] = {}

    try:
        for genome_peptides_path in file_paths:
            genome_peptides_path = str(genome_peptides_path)
            genome_id = Path(genome_peptides_path).stem
            peptides = _read_unique_peptides_from_digest(
                genome_peptides_path,
                peptide_normalization_policy=peptide_normalization_policy,
            )
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
            seen_theoretical = _read_unique_peptides_from_digest(
                genome_peptides_path,
                peptide_normalization_policy=_PEPTIDE_NORMALIZATION_POLICY_WORKER,
            )
            matched_peptides = set(seen_theoretical).intersection(_OBS_PEPTIDES_WORKER)
            results.append((genome_id, matched_peptides, len(seen_theoretical), None))
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            results.append((genome_id, set(), 0, err))

    return results
