from __future__ import annotations

import contextlib
import csv
import io
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional


LogCallback = Callable[[str], None]


@dataclass
class DigestConfig:
    input_mode: str = "directory"
    input_file: str = ""
    input_dir: str = ""
    output_file: str = ""
    output_dir: str = ""
    enzyme_id: str = "42"
    min_length: int = 7
    max_length: int = 30
    max_num_miscleavages: int = 2
    processes: Optional[int] = None
    short_header: bool = True
    verbose: bool = True
    skip_existing: bool = True


@dataclass
class ScoringConfig:
    peptide_table_path: str = ""
    genome_lineage_table_path: str = ""
    genome_lineage_genome_id_col: str = ""
    genome_lineage_lineage_col: str = ""
    genome_digest_dirs: list[str] = field(default_factory=list)
    selected_genome_ids: list[str] = field(default_factory=list)
    output_tsv_path: str = ""
    peptide_seq_col: str = "Sequence"
    peptide_score_col: str = "Evidence"
    peptide_error_col: str = "Q.Value"
    peptide_error_cutoff: float = 0.05
    single_peptide_error_rate_upper_bound: float = 0.3
    unique_pvalue_mode: str = "hypergeometric-opportunity"
    min_unique_for_unique_pvalue: int = 3
    theoretical_opportunity_cache_path: str = ""
    rebuild_theoretical_opportunity_cache: bool = False
    num_workers_for_theoretical_opportunity: Optional[int] = None
    unit_aware: bool = False
    sample_id_col: str = "Run"
    intensity_col: str = "Precursor.Quantity"
    intensity_min_value: float = 0.0
    intensity_min_quantile: float = 0.0
    metadata_table_path: str = ""
    metadata_sample_id_col: str = "sample_id"
    metadata_analysis_unit_col: str = "analysis_unit_id"
    export_unit_derived_tables: bool = False
    peptide_decoy_flag_col: str = "Reverse"
    decoy_flag_value: str = "+"
    exclude_genome_ids: list[str] = field(default_factory=list)
    num_workers: Optional[int] = None
    knockoff_mc_iterations: int = 500
    knockoff_stage2_mc_iterations: Optional[int] = 2000
    knockoff_stage2_p_exist_ranges: list[list[float]] = field(
        default_factory=lambda: [[0.01, 0.05]]
    )
    knockoff_random_seed: int = 1
    knockoff_top_n_targets: Optional[int] = None
    matched_peptides_cache_path: str = ""
    save_matched_peptides_cache: bool = True
    use_cache_if_exists: bool = False
    compute_coverage: bool = True
    export_temp: bool = True
    export_peptide_contrib_topN: int = 0
    return_full_table: bool = False


@dataclass
class ParquetExtractionConfig:
    input_parquet_path: str = ""
    output_tsv_path: str = ""
    input_columns: list[str] = field(
        default_factory=lambda: ["Run", "Stripped.Sequence", "Evidence", "Q.Value"]
    )
    output_columns: list[str] = field(
        default_factory=lambda: ["Run", "Sequence", "Evidence", "Q.Value"]
    )
    batch_size: int = 65536
    force: bool = False


class CallbackLogHandler(logging.Handler):
    def __init__(self, callback: LogCallback):
        super().__init__()
        self.callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            message = record.getMessage()
        self.callback(message)


class StreamToCallback(io.TextIOBase):
    def __init__(self, callback: LogCallback):
        super().__init__()
        self.callback = callback
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        normalized = text.replace("\r", "\n")
        self._buffer += normalized
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self.callback(line)
        return len(text)

    def flush(self) -> None:
        if self._buffer.strip():
            self.callback(self._buffer.strip())
        self._buffer = ""


@contextlib.contextmanager
def capture_runtime_output(callback: Optional[LogCallback], logger_names: Iterable[str]):
    if callback is None:
        yield
        return

    handler = CallbackLogHandler(callback)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    attached_loggers: list[logging.Logger] = []

    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)
        logger.addHandler(handler)
        attached_loggers.append(logger)

    stdout_stream = StreamToCallback(callback)
    stderr_stream = StreamToCallback(callback)

    try:
        with contextlib.redirect_stdout(stdout_stream), contextlib.redirect_stderr(stderr_stream):
            yield
    finally:
        stdout_stream.flush()
        stderr_stream.flush()
        for logger in attached_loggers:
            logger.removeHandler(handler)


def _none_if_blank(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _normalize_output_path(path_str: str) -> str:
    if not path_str.strip():
        return ""
    return str(Path(path_str).expanduser())


PARQUET_EXTENSIONS = {".parquet", ".pq"}


def _is_parquet_path(path_str: str) -> bool:
    return Path(path_str).suffix.lower() in PARQUET_EXTENSIONS


def _normalize_parquet_column_key(name: str) -> str:
    return "".join(char.lower() for char in str(name) if char.isalnum())


def _resolve_parquet_column(
    schema_names: list[str],
    normalized_lookup: dict[str, str],
    preferred: Optional[str],
    candidates: list[str],
    used: set[str],
) -> tuple[Optional[str], bool]:
    if preferred:
        if preferred in schema_names and preferred not in used:
            used.add(preferred)
            return preferred, False
        key = _normalize_parquet_column_key(preferred)
        match = normalized_lookup.get(key)
        if match and match not in used:
            used.add(match)
            return match, True

    for candidate in candidates:
        key = _normalize_parquet_column_key(candidate)
        match = normalized_lookup.get(key)
        if match and match not in used:
            used.add(match)
            return match, True

    return None, False


def _load_parquet_peptide_table(
    parquet_path: str,
    peptide_seq_col: str,
    peptide_score_col: Optional[str],
    peptide_error_col: Optional[str],
    peptide_decoy_flag_col: Optional[str],
    log_callback: Optional[LogCallback] = None,
) -> tuple["pd.DataFrame", dict[str, Optional[str]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to read parquet files. Install it with: python -m pip install pyarrow"
        ) from exc

    import pandas as pd

    parquet_path = str(Path(parquet_path).expanduser())
    schema = pq.read_schema(parquet_path)
    schema_names = list(schema.names)
    normalized_lookup: dict[str, str] = {}
    for name in schema_names:
        key = _normalize_parquet_column_key(name)
        if key and key not in normalized_lookup:
            normalized_lookup[key] = name

    used: set[str] = set()
    seq_col, seq_auto = _resolve_parquet_column(
        schema_names,
        normalized_lookup,
        peptide_seq_col,
        ["Stripped.Sequence", "StrippedSequence", "Sequence", "Peptide.Sequence", "PeptideSequence"],
        used,
    )
    if not seq_col:
        expected = "Stripped.Sequence or Sequence"
        available = ", ".join(schema_names)
        raise ValueError(
            "Unable to locate a peptide sequence column in the parquet file. "
            f"Expected {expected}. Available columns: {available}"
        )

    score_col, score_auto = _resolve_parquet_column(
        schema_names,
        normalized_lookup,
        peptide_score_col,
        ["Evidence", "Score", "CScore"] if peptide_score_col is not None else [],
        used,
    )
    error_col, error_auto = _resolve_parquet_column(
        schema_names,
        normalized_lookup,
        peptide_error_col,
        ["Q.Value", "QValue", "Qval", "QVal", "PEP", "FDR"],
        used,
    )
    decoy_col, _ = _resolve_parquet_column(
        schema_names,
        normalized_lookup,
        peptide_decoy_flag_col,
        ["Reverse", "Target/Decoy", "TargetDecoy", "Decoy"] if peptide_decoy_flag_col is not None else [],
        used,
    )

    columns_to_read = [seq_col]
    for candidate in (score_col, error_col, decoy_col):
        if candidate and candidate not in columns_to_read:
            columns_to_read.append(candidate)

    if log_callback:
        log_callback("Detected parquet peptide table; loading required columns.")
        log_callback(
            "Parquet columns: "
            f"sequence={seq_col} ({'auto' if seq_auto else 'config'}), "
            f"score={score_col or 'none'} ({'auto' if score_auto else 'config'}), "
            f"error={error_col or 'none'} ({'auto' if error_auto else 'config'}), "
            f"decoy={decoy_col or 'none'}"
        )

    table = pq.read_table(parquet_path, columns=columns_to_read)
    df = table.to_pandas()

    resolved = {
        "peptide_seq_col": seq_col,
        "peptide_score_col": score_col,
        "peptide_error_col": error_col,
        "peptide_decoy_flag_col": decoy_col,
    }
    return df, resolved


def _clean_parquet_peptide_table(
    peptide_table_df: "pd.DataFrame",
    resolved_columns: dict[str, Optional[str]],
    log_callback: Optional[LogCallback] = None,
) -> "pd.DataFrame":
    seq_col = resolved_columns.get("peptide_seq_col")
    if not seq_col or seq_col not in peptide_table_df.columns:
        return peptide_table_df.copy()

    df = peptide_table_df.copy()
    before = int(len(df))
    df[seq_col] = df[seq_col].astype("string").str.strip()
    df = df[df[seq_col].notna() & (df[seq_col] != "")].copy()
    dropped = before - int(len(df))
    if dropped and log_callback:
        log_callback(f"Dropped {dropped} parquet row(s) with missing or empty peptide sequences.")
    return df


def _infer_decoy_flag_value(
    peptide_table_df: "pd.DataFrame",
    resolved_columns: dict[str, Optional[str]],
    configured_value: str,
    log_callback: Optional[LogCallback] = None,
) -> str:
    decoy_col = resolved_columns.get("peptide_decoy_flag_col")
    configured = str(configured_value)
    if not decoy_col or decoy_col not in peptide_table_df.columns or configured == "":
        return configured

    values = [str(value).strip() for value in peptide_table_df[decoy_col].dropna().unique()[:50]]
    value_set = set(values)
    if configured in value_set:
        return configured

    # Common parquet encodings for decoys include boolean True, integer 1, and string labels.
    # Only auto-adjust from the historical '+' default; explicit user choices are preserved.
    if configured != "+":
        return configured

    for candidate in ("True", "true", "1", "decoy", "Decoy", "DECOY", "T", "t"):
        if candidate in value_set:
            if log_callback:
                log_callback(
                    f"Auto-detected parquet decoy marker for column '{decoy_col}': "
                    f"using '{candidate}' instead of '+'."
                )
            return candidate

    return configured


def run_digest_workflow(config: DigestConfig, log_callback: Optional[LogCallback] = None) -> dict:
    import importlib

    digest_module = importlib.import_module("metaumbra.digest")

    start = time.time()
    target = config.input_file if config.input_mode == "file" else config.input_dir
    if log_callback:
        log_callback(f"Starting digest workflow for: {target}")

    with capture_runtime_output(log_callback, [digest_module.__name__]):
        if config.input_mode == "file":
            output_file = _normalize_output_path(config.output_file)
            peptide_count = digest_module.process_fasta(
                input_file=str(Path(config.input_file).expanduser()),
                enzyme_id=str(config.enzyme_id),
                output_file=output_file or None,
                min_length=int(config.min_length),
                max_length=int(config.max_length),
                max_num_miscleavages=int(config.max_num_miscleavages),
                processes=config.processes,
                short_header=bool(config.short_header),
                verbose=bool(config.verbose),
            )
            return {
                "mode": "file",
                "input": str(Path(config.input_file).expanduser()),
                "output": output_file,
                "peptides": int(peptide_count),
                "elapsed_seconds": round(time.time() - start, 2),
            }

        output_dir = str(Path(config.output_dir).expanduser())
        results = digest_module.process_directory(
            input_dir=str(Path(config.input_dir).expanduser()),
            output_dir=output_dir,
            enzyme_id=str(config.enzyme_id),
            min_length=int(config.min_length),
            max_length=int(config.max_length),
            max_num_miscleavages=int(config.max_num_miscleavages),
            processes=config.processes,
            short_header=bool(config.short_header),
            verbose=bool(config.verbose),
            skip_existing=bool(config.skip_existing),
        )
        return {
            "mode": "directory",
            "input": str(Path(config.input_dir).expanduser()),
            "output": output_dir,
            "files_processed": len(results),
            "peptides": int(sum(results.values())),
            "elapsed_seconds": round(time.time() - start, 2),
        }


def run_scoring_workflow(config: ScoringConfig, log_callback: Optional[LogCallback] = None) -> dict:
    import importlib

    scoring_module = importlib.import_module("metaumbra.scoring")

    start = time.time()
    if log_callback:
        log_callback(f"Starting genome presence scoring for: {config.peptide_table_path}")

    peptide_table_path = str(Path(config.peptide_table_path).expanduser())
    peptide_table_df = None
    resolved_columns: dict[str, Optional[str]] | None = None
    effective_decoy_flag_value = config.decoy_flag_value
    if (not config.unit_aware) and _is_parquet_path(peptide_table_path):
        if not os.path.isfile(peptide_table_path):
            raise FileNotFoundError(f"Peptide parquet file does not exist: {peptide_table_path}")
        peptide_table_df, resolved_columns = _load_parquet_peptide_table(
            parquet_path=peptide_table_path,
            peptide_seq_col=config.peptide_seq_col,
            peptide_score_col=_none_if_blank(config.peptide_score_col),
            peptide_error_col=_none_if_blank(config.peptide_error_col),
            peptide_decoy_flag_col=_none_if_blank(config.peptide_decoy_flag_col),
            log_callback=log_callback,
        )
        peptide_table_df = _clean_parquet_peptide_table(
            peptide_table_df=peptide_table_df,
            resolved_columns=resolved_columns,
            log_callback=log_callback,
        )
        effective_decoy_flag_value = _infer_decoy_flag_value(
            peptide_table_df=peptide_table_df,
            resolved_columns=resolved_columns,
            configured_value=config.decoy_flag_value,
            log_callback=log_callback,
        )

    normalized_ranges = [
        (float(bounds[0]), float(bounds[1]))
        for bounds in config.knockoff_stage2_p_exist_ranges
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2
    ]

    output_tsv_path = _normalize_output_path(config.output_tsv_path)
    cache_path = _normalize_output_path(config.matched_peptides_cache_path)
    theoretical_cache_path = _normalize_output_path(config.theoretical_opportunity_cache_path)
    genome_lineage_table_path = _normalize_output_path(config.genome_lineage_table_path)

    with capture_runtime_output(log_callback, ["GenomePresenceScorer"]):
        calc = scoring_module.GenomePresenceScorer(num_workers=config.num_workers)
        calc.knockoff_mc_iterations = int(config.knockoff_mc_iterations)
        calc.knockoff_stage2_mc_iterations = config.knockoff_stage2_mc_iterations
        calc.knockoff_stage2_p_exist_ranges = normalized_ranges
        calc.knockoff_random_seed = int(config.knockoff_random_seed)
        calc.knockoff_top_n_targets = config.knockoff_top_n_targets

        if config.unit_aware:
            calc.read_unit_aware_peptide_file(
                peptide_table_path=peptide_table_path,
                sample_id_col=config.sample_id_col,
                peptide_seq_col=config.peptide_seq_col,
                peptide_score_col=_none_if_blank(config.peptide_score_col),
                peptide_decoy_flag_col=_none_if_blank(config.peptide_decoy_flag_col),
                decoy_flag_value=config.decoy_flag_value,
                intensity_col=config.intensity_col,
                peptide_error_col=_none_if_blank(config.peptide_error_col),
                peptide_error_cutoff=float(config.peptide_error_cutoff),
                single_peptide_error_rate_upper_bound=float(config.single_peptide_error_rate_upper_bound),
                intensity_min_value=float(config.intensity_min_value),
                intensity_min_quantile=float(config.intensity_min_quantile),
                metadata_table_path=_normalize_output_path(config.metadata_table_path) or None,
                metadata_sample_id_col=config.metadata_sample_id_col,
                metadata_analysis_unit_col=config.metadata_analysis_unit_col,
                peptide_table_sep="\t",
            )
        elif peptide_table_df is None:
            calc.read_peptide_file(
                peptide_table_path=peptide_table_path,
                peptide_seq_col=config.peptide_seq_col,
                peptide_score_col=_none_if_blank(config.peptide_score_col),
                peptide_decoy_flag_col=_none_if_blank(config.peptide_decoy_flag_col),
                decoy_flag_value=config.decoy_flag_value,
                peptide_table_sep="\t",
                peptide_error_col=_none_if_blank(config.peptide_error_col),
                peptide_error_cutoff=float(config.peptide_error_cutoff),
                single_peptide_error_rate_upper_bound=float(config.single_peptide_error_rate_upper_bound),
            )
        else:
            effective_seq_col = resolved_columns.get("peptide_seq_col") if resolved_columns else None
            if not effective_seq_col:
                raise RuntimeError("Unable to resolve a peptide sequence column for parquet scoring.")
            calc.read_peptide_file(
                peptide_table_df=peptide_table_df,
                peptide_seq_col=effective_seq_col,
                peptide_score_col=resolved_columns.get("peptide_score_col") if resolved_columns else None,
                peptide_decoy_flag_col=resolved_columns.get("peptide_decoy_flag_col") if resolved_columns else None,
                decoy_flag_value=effective_decoy_flag_value,
                peptide_table_sep="\t",
                peptide_error_col=resolved_columns.get("peptide_error_col") if resolved_columns else None,
                peptide_error_cutoff=float(config.peptide_error_cutoff),
                single_peptide_error_rate_upper_bound=float(config.single_peptide_error_rate_upper_bound),
            )
            calc.peptide_table_dir = os.path.dirname(peptide_table_path)

        result_df = calc.analyze_genomes(
            genome_digest_dirs=[str(Path(p).expanduser()) for p in config.genome_digest_dirs],
            output_tsv_path=output_tsv_path or None,
            genome_lineage_table_path=genome_lineage_table_path or None,
            genome_lineage_genome_id_col=_none_if_blank(config.genome_lineage_genome_id_col),
            genome_lineage_lineage_col=_none_if_blank(config.genome_lineage_lineage_col),
            genome_list=config.selected_genome_ids or None,
            exclude_genome_ids=config.exclude_genome_ids or None,
            all_matched_peptides=None,
            save_matched_peptides_cache=bool(config.save_matched_peptides_cache),
            matched_peptides_cache_path=cache_path or None,
            compute_coverage=bool(config.compute_coverage),
            export_temp=bool(config.export_temp),
            export_peptide_contrib_topN=int(config.export_peptide_contrib_topN),
            use_cache_if_exists=bool(config.use_cache_if_exists),
            unique_pvalue_mode=str(config.unique_pvalue_mode),
            min_unique_for_unique_pvalue=int(config.min_unique_for_unique_pvalue),
            theoretical_opportunity_cache_path=theoretical_cache_path or None,
            rebuild_theoretical_opportunity_cache=bool(config.rebuild_theoretical_opportunity_cache),
            num_workers_for_theoretical_opportunity=config.num_workers_for_theoretical_opportunity,
            return_full_table=bool(config.return_full_table),
            unit_aware=bool(config.unit_aware),
            export_unit_derived_tables=bool(config.export_unit_derived_tables),
        )

    saved_output = output_tsv_path
    if not saved_output:
        output_dir = calc.peptide_table_dir if calc.peptide_table_dir else os.getcwd()
        saved_output = os.path.join(output_dir, "genome_presence.tsv")

    return {
        "input": peptide_table_path,
        "output": saved_output,
        "rows": int(len(result_df)),
        "elapsed_seconds": round(time.time() - start, 2),
    }


def run_parquet_extraction_workflow(
    config: ParquetExtractionConfig,
    log_callback: Optional[LogCallback] = None,
) -> dict:
    def log(message: str) -> None:
        if log_callback:
            log_callback(message)

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to read parquet files. Install it with: python -m pip install pyarrow"
        ) from exc

    start = time.time()
    input_path = Path(config.input_parquet_path).expanduser()
    output_path = Path(config.output_tsv_path).expanduser() if config.output_tsv_path else input_path.with_suffix(".tsv")
    input_columns = list(config.input_columns)
    output_columns = list(config.output_columns)
    batch_size = int(config.batch_size)

    log("Starting parquet peptide extraction")
    log(f"Input parquet: {input_path}")
    log(f"Output TSV: {output_path}")
    log(f"Overwrite enabled: {'yes' if config.force else 'no'}")
    log(f"Batch size: {batch_size}")
    log("Column mapping:")
    for source_column, output_column in zip(input_columns, output_columns):
        log(f"  {source_column} -> {output_column}")

    if not input_path.is_file():
        raise FileNotFoundError(f"Input parquet file does not exist: {input_path}")
    if len(input_columns) != len(output_columns):
        raise ValueError("Input columns and output columns must have the same length.")
    if batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")
    if output_path.exists() and not config.force:
        raise FileExistsError(f"Output file already exists: {output_path}. Enable overwrite to replace it.")

    log("Reading parquet schema")
    schema = pq.read_schema(input_path)
    missing = [column for column in input_columns if column not in schema.names]
    if missing:
        raise ValueError(
            "Missing required parquet columns: "
            + ", ".join(missing)
            + "\nAvailable columns: "
            + ", ".join(schema.names)
        )
    log(f"Schema check passed. Available columns: {len(schema.names)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_file = pq.ParquetFile(input_path)
    total_rows = int(parquet_file.metadata.num_rows) if parquet_file.metadata is not None else None
    row_groups = int(parquet_file.metadata.num_row_groups) if parquet_file.metadata is not None else None
    rows_written = 0
    batches_written = 0
    next_progress_row = batch_size * 10
    if total_rows is not None:
        log(f"Parquet metadata: {total_rows} rows across {row_groups} row groups")
    else:
        log("Parquet metadata row count is unavailable")

    log("Writing TSV header")
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(output_columns)

        log("Streaming parquet batches to TSV")
        for batch in parquet_file.iter_batches(columns=input_columns, batch_size=batch_size):
            batches_written += 1
            batch_rows = int(batch.num_rows)
            arrays = [batch.column(idx).to_pylist() for idx in range(batch.num_columns)]
            for row in zip(*arrays):
                writer.writerow(row)
                rows_written += 1

            should_log_batch = batches_written <= 3 or rows_written >= next_progress_row
            if total_rows is not None and rows_written >= total_rows:
                should_log_batch = True
            if should_log_batch:
                if total_rows:
                    percent = (rows_written / total_rows) * 100
                    log(
                        f"Processed batch {batches_written}: {batch_rows} rows, "
                        f"{rows_written}/{total_rows} total ({percent:.1f}%)"
                    )
                else:
                    log(
                        f"Processed batch {batches_written}: {batch_rows} rows, "
                        f"{rows_written} total"
                    )
                while rows_written >= next_progress_row:
                    next_progress_row += batch_size * 10

    elapsed_seconds = round(time.time() - start, 2)
    log(f"Finished parquet extraction: {rows_written} rows written in {elapsed_seconds} s")
    log(f"Saved TSV: {output_path}")

    return {
        "input": str(input_path),
        "output": str(output_path),
        "rows": int(rows_written),
        "elapsed_seconds": elapsed_seconds,
    }


if __name__ == "__main__":
    if __package__ in {None, ""}:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from metaumbra.cli import main as cli_main
    else:
        from .cli import main as cli_main
    raise SystemExit(cli_main(sys.argv[1:]))
