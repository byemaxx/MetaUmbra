from __future__ import annotations

import contextlib
import csv
import io
import json
import logging
import os
import platform
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional


LogCallback = Callable[[str], None]


def format_elapsed_seconds(elapsed_seconds: object) -> str:
    try:
        elapsed = float(elapsed_seconds)
    except (TypeError, ValueError):
        elapsed = 0.0
    elapsed = max(0.0, elapsed)
    minutes = int(elapsed // 60)
    seconds = elapsed - (minutes * 60)
    return f"{minutes} min {seconds:05.2f} s"


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
    single_peptide_error_rate_upper_bound: float = 0.05
    unique_pvalue_mode: str = "empirical-background"
    unique_peptide_error_source: str = "global-alpha"
    unique_count_power: float = 1.0
    unique_empirical_background_threshold_quantile: float = 0.95
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
    export_unit_derived_tables: Optional[bool] = None
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


class ArtifactLogTee:
    def __init__(self, log_path: Path, callback: Optional[LogCallback], mode: str = "a"):
        self.log_path = log_path
        self.callback = callback
        self._next_mode = "w" if str(mode).lower().startswith("w") else "a"

    def __call__(self, message: str) -> None:
        text = str(message)
        with self.log_path.open(self._next_mode, encoding="utf-8", newline="") as handle:
            handle.write(text + "\n")
        self._next_mode = "a"
        if self.callback is not None:
            self.callback(text)

    def close(self) -> None:
        return


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


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _scoring_artifact_dir(output_tsv_path: str) -> Optional[Path]:
    normalized_output = _normalize_output_path(output_tsv_path)
    if not normalized_output:
        return None
    output_path = Path(normalized_output)
    return output_path.parent / f"{output_path.stem}_artifacts"


def _write_json_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _clean_scoring_artifacts_for_new_run(artifact_dir: Path, config: ScoringConfig) -> None:
    """Remove generated diagnostics that should not carry across independent runs."""
    known_files = {
        "run_summary.json",
        "full_internal_metrics.tsv",
        "knockoff_pools.tsv",
        "degeneracy_hist.tsv",
        "p_shared_hist.tsv",
        "q_calling_curve.tsv",
        "shared_stratum_counts.tsv",
        "pooled_genome_presence.tsv",
    }
    if (
        str(config.unique_pvalue_mode).strip().lower() != "hypergeometric-opportunity"
        and not str(config.theoretical_opportunity_cache_path or "").strip()
    ):
        known_files.add("theoretical_opportunity_cache.pkl")

    cleanup_paths = [artifact_dir / name for name in known_files]
    cleanup_paths.extend(artifact_dir.glob("top*_peptide_contrib.tsv"))
    unit_aware_dir = artifact_dir / "unit_aware"
    if unit_aware_dir.exists():
        for pattern in (
            "unit_empirical_background_calibration.tsv",
            "unit_threshold_summary.tsv",
            "unit_call_counts.tsv",
            "unit_q001_genomes.tsv",
            "unit_q005_genomes.tsv",
            "genome_union_q001.tsv",
            "genome_union_q005.tsv",
            "genome_by_unit_q001_matrix.tsv",
            "genome_by_unit_q005_matrix.tsv",
            "genome_by_unit_qvalue_matrix.tsv",
        ):
            cleanup_paths.extend(unit_aware_dir.glob(pattern))

    artifact_root = artifact_dir.resolve()
    for path in cleanup_paths:
        try:
            resolved = path.resolve()
            if resolved.is_file() and (resolved.parent == artifact_root or resolved.parent.parent == artifact_root):
                resolved.unlink()
        except Exception:
            pass


def _detect_cpu_model() -> Optional[str]:
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            cleaned = str(value).strip()
            if cleaned:
                return re.sub(r"\s+", " ", cleaned)
        except Exception:
            pass

    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.lower().startswith("model name"):
                    _, value = line.split(":", 1)
                    cleaned = value.strip()
                    if cleaned:
                        return re.sub(r"\s+", " ", cleaned)
        except Exception:
            pass

    for candidate in (platform.processor(), os.environ.get("PROCESSOR_IDENTIFIER")):
        if candidate:
            cleaned = str(candidate).strip()
            if cleaned:
                return re.sub(r"\s+", " ", cleaned)
    return None


def _detect_total_memory_bytes() -> Optional[int]:
    if sys.platform == "win32":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        except Exception:
            pass

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages and page_size:
            return int(pages) * int(page_size)
    except (AttributeError, OSError, ValueError):
        pass
    return None


def _environment_metadata() -> dict:
    total_memory_bytes = _detect_total_memory_bytes()
    return {
        "cpu_model": _detect_cpu_model(),
        "cpu_count_logical": os.cpu_count(),
        "total_memory_bytes": total_memory_bytes,
        "total_memory_gib": round(total_memory_bytes / (1024**3), 2) if total_memory_bytes else None,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "architecture": platform.architecture()[0],
    }


def _workflow_metadata(workflow: str, started_at_utc: str) -> dict:
    try:
        from metaumbra import __version__ as app_version
    except Exception:
        app_version = "unknown"

    return {
        "workflow": workflow,
        "started_at_utc": started_at_utc,
        "metaumbra_version": str(app_version),
        "python_version": sys.version,
        "platform": platform.platform(),
        "environment": _environment_metadata(),
    }


def _initialize_scoring_artifacts(
    config: ScoringConfig,
    output_tsv_path: str,
    log_callback: Optional[LogCallback],
) -> tuple[Optional[Path], Optional[ArtifactLogTee], str]:
    started_at_utc = _utc_timestamp()
    artifact_dir = _scoring_artifact_dir(output_tsv_path)
    if artifact_dir is None:
        return None, None, started_at_utc

    artifact_dir.mkdir(parents=True, exist_ok=True)
    _clean_scoring_artifacts_for_new_run(artifact_dir, config)
    parameters_payload = {
        **_workflow_metadata("score", started_at_utc),
        "output_tsv_path": output_tsv_path,
        "artifact_dir": str(artifact_dir),
        "config": asdict(config),
    }
    _write_json_file(artifact_dir / "run_parameters.json", parameters_payload)
    log_tee = ArtifactLogTee(artifact_dir / "run.log", log_callback, mode="w")
    log_tee(f"=== MetaUmbra score started at {started_at_utc} UTC ===")
    log_tee(f"Artifacts directory: {artifact_dir}")
    return artifact_dir, log_tee, started_at_utc


def _write_scoring_status(
    artifact_dir: Optional[Path],
    status: str,
    started_at_utc: str,
    result: Optional[dict] = None,
    error: Optional[BaseException] = None,
) -> None:
    if artifact_dir is None:
        return

    payload = {
        **_workflow_metadata("score", started_at_utc),
        "finished_at_utc": _utc_timestamp(),
        "status": status,
    }
    if result is not None:
        payload["result"] = result
    if error is not None:
        payload["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    _write_json_file(artifact_dir / "run_status.json", payload)


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


def _run_scoring_workflow_uncaught(config: ScoringConfig, log_callback: Optional[LogCallback] = None) -> dict:
    import importlib

    scoring_module = importlib.import_module("metaumbra.scoring")

    start = time.time()
    output_tsv_path = _normalize_output_path(config.output_tsv_path)
    artifact_dir, artifact_log_callback, started_at_utc = _initialize_scoring_artifacts(
        config=config,
        output_tsv_path=output_tsv_path,
        log_callback=log_callback,
    )
    active_log_callback = artifact_log_callback if artifact_log_callback is not None else log_callback
    if active_log_callback:
        active_log_callback(f"Starting genome presence scoring for: {config.peptide_table_path}")

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
            log_callback=active_log_callback,
        )
        peptide_table_df = _clean_parquet_peptide_table(
            peptide_table_df=peptide_table_df,
            resolved_columns=resolved_columns,
            log_callback=active_log_callback,
        )
        effective_decoy_flag_value = _infer_decoy_flag_value(
            peptide_table_df=peptide_table_df,
            resolved_columns=resolved_columns,
            configured_value=config.decoy_flag_value,
            log_callback=active_log_callback,
        )

    normalized_ranges = [
        (float(bounds[0]), float(bounds[1]))
        for bounds in config.knockoff_stage2_p_exist_ranges
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2
    ]

    cache_path = _normalize_output_path(config.matched_peptides_cache_path)
    theoretical_cache_path = _normalize_output_path(config.theoretical_opportunity_cache_path)
    genome_lineage_table_path = _normalize_output_path(config.genome_lineage_table_path)

    with capture_runtime_output(active_log_callback, ["GenomePresenceScorer"]):
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

        unique_empirical_background_threshold_quantile = float(
            config.unique_empirical_background_threshold_quantile
        )
        if (
            str(config.unique_pvalue_mode).strip().lower() == "empirical-background"
            and not (0.90 <= unique_empirical_background_threshold_quantile <= 0.99)
        ):
            raise ValueError(
                "Empirical background threshold quantile must be between 0.90 and 0.99."
            )
        calc.unique_empirical_background_threshold_quantile = unique_empirical_background_threshold_quantile

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
            unique_peptide_error_source=str(config.unique_peptide_error_source),
            unique_count_power=float(config.unique_count_power),
            theoretical_opportunity_cache_path=theoretical_cache_path or None,
            rebuild_theoretical_opportunity_cache=bool(config.rebuild_theoretical_opportunity_cache),
            num_workers_for_theoretical_opportunity=config.num_workers_for_theoretical_opportunity,
            return_full_table=bool(config.return_full_table),
            unit_aware=bool(config.unit_aware),
            export_unit_derived_tables=config.export_unit_derived_tables,
        )

    saved_output = output_tsv_path
    if not saved_output:
        output_dir = calc.peptide_table_dir if calc.peptide_table_dir else os.getcwd()
        saved_output = os.path.join(output_dir, "genome_presence.tsv")

    result = {
        "input": peptide_table_path,
        "output": saved_output,
        "artifacts": str(artifact_dir) if artifact_dir is not None else "",
        "rows": int(len(result_df)),
        "elapsed_seconds": round(time.time() - start, 2),
    }
    _write_scoring_status(artifact_dir, "success", started_at_utc, result=result)
    if active_log_callback:
        active_log_callback(
            f"Finished genome presence scoring in {format_elapsed_seconds(result['elapsed_seconds'])}."
        )
    return result


def run_scoring_workflow(config: ScoringConfig, log_callback: Optional[LogCallback] = None) -> dict:
    try:
        return _run_scoring_workflow_uncaught(config, log_callback)
    except Exception as exc:
        artifact_dir = _scoring_artifact_dir(config.output_tsv_path)
        started_at_utc = _utc_timestamp()
        if artifact_dir is not None:
            try:
                parameters_path = artifact_dir / "run_parameters.json"
                if parameters_path.exists():
                    with parameters_path.open("r", encoding="utf-8") as handle:
                        payload = json.load(handle)
                    started_at_utc = str(payload.get("started_at_utc") or started_at_utc)
            except Exception:
                pass
            artifact_dir.mkdir(parents=True, exist_ok=True)
            log_mode = "a" if (artifact_dir / "run_parameters.json").exists() else "w"
            ArtifactLogTee(artifact_dir / "run.log", log_callback, mode=log_mode)(
                f"Scoring workflow failed: {type(exc).__name__}: {exc}"
            )
            _write_scoring_status(artifact_dir, "failed", started_at_utc, error=exc)
        raise


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
    log(f"Finished parquet extraction: {rows_written} rows written in {format_elapsed_seconds(elapsed_seconds)}")
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

