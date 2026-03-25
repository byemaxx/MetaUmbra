from __future__ import annotations

import contextlib
import importlib
import io
import logging
import os
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
    output_tsv_path: str = ""
    peptide_seq_col: str = "Sequence"
    peptide_score_col: str = "score"
    peptide_error_col: str = "Q.Value"
    peptide_error_cutoff: float = 0.05
    peptide_decoy_flag_col: str = "Reverse"
    decoy_flag_value: str = "+"
    exclude_genome_ids: list[str] = field(default_factory=list)
    num_workers: Optional[int] = None
    knockoff_mc_iterations: int = 500
    knockoff_stage2_mc_iterations: Optional[int] = 2000
    knockoff_stage2_p_exist_ranges: list[list[float]] = field(
        default_factory=lambda: [[0.005, 0.02], [0.02, 0.08]]
    )
    knockoff_random_seed: int = 1
    knockoff_top_n_targets: Optional[int] = None
    matched_peptides_cache_path: str = ""
    save_matched_peptides_cache: bool = True
    use_cache_if_exists: bool = False
    use_peptide_error_for_unique_pvalue: bool = False
    compute_coverage: bool = True
    export_temp: bool = True
    export_peptide_contrib_topN: int = 0
    return_full_table: bool = False


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


def run_digest_workflow(config: DigestConfig, log_callback: Optional[LogCallback] = None) -> dict:
    digest_module = importlib.import_module("digest_fasta_to_peptides")

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
    scoring_module = importlib.import_module("genome_presence_scoring")

    start = time.time()
    if log_callback:
        log_callback(f"Starting genome presence scoring for: {config.peptide_table_path}")

    normalized_ranges = [
        (float(bounds[0]), float(bounds[1]))
        for bounds in config.knockoff_stage2_p_exist_ranges
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2
    ]

    output_tsv_path = _normalize_output_path(config.output_tsv_path)
    cache_path = _normalize_output_path(config.matched_peptides_cache_path)
    genome_lineage_table_path = _normalize_output_path(config.genome_lineage_table_path)

    with capture_runtime_output(log_callback, ["GenomePresenceScorer"]):
        calc = scoring_module.GenomePresenceScorer(num_workers=config.num_workers)
        calc.knockoff_mc_iterations = int(config.knockoff_mc_iterations)
        calc.knockoff_stage2_mc_iterations = config.knockoff_stage2_mc_iterations
        calc.knockoff_stage2_p_exist_ranges = normalized_ranges
        calc.knockoff_random_seed = int(config.knockoff_random_seed)
        calc.knockoff_top_n_targets = config.knockoff_top_n_targets

        calc.read_peptide_file(
            peptide_table_path=str(Path(config.peptide_table_path).expanduser()),
            peptide_seq_col=config.peptide_seq_col,
            peptide_score_col=_none_if_blank(config.peptide_score_col),
            peptide_decoy_flag_col=_none_if_blank(config.peptide_decoy_flag_col),
            decoy_flag_value=config.decoy_flag_value,
            peptide_table_sep="\t",
            peptide_error_col=_none_if_blank(config.peptide_error_col),
            peptide_error_cutoff=float(config.peptide_error_cutoff),
        )

        result_df = calc.analyze_genomes(
            genome_digest_dirs=[str(Path(p).expanduser()) for p in config.genome_digest_dirs],
            output_tsv_path=output_tsv_path or None,
            genome_lineage_table_path=genome_lineage_table_path or None,
            genome_lineage_genome_id_col=_none_if_blank(config.genome_lineage_genome_id_col),
            genome_lineage_lineage_col=_none_if_blank(config.genome_lineage_lineage_col),
            exclude_genome_ids=config.exclude_genome_ids or None,
            all_matched_peptides=None,
            save_matched_peptides_cache=bool(config.save_matched_peptides_cache),
            matched_peptides_cache_path=cache_path or None,
            compute_coverage=bool(config.compute_coverage),
            export_temp=bool(config.export_temp),
            export_peptide_contrib_topN=int(config.export_peptide_contrib_topN),
            use_cache_if_exists=bool(config.use_cache_if_exists),
            use_peptide_error_for_unique_pvalue=bool(config.use_peptide_error_for_unique_pvalue),
            return_full_table=bool(config.return_full_table),
        )

    saved_output = output_tsv_path
    if not saved_output:
        output_dir = calc.peptide_table_dir if calc.peptide_table_dir else os.getcwd()
        saved_output = os.path.join(output_dir, "genome_presence.tsv")

    return {
        "input": str(Path(config.peptide_table_path).expanduser()),
        "output": saved_output,
        "rows": int(len(result_df)),
        "elapsed_seconds": round(time.time() - start, 2),
    }
