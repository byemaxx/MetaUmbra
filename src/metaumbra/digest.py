#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Directly call RPG internal functions for protein digestion, and output only
Protein and Peptide columns.

This version is optimized for MAG-scale workloads:
- single-file mode uses streaming, in-memory digestion record by record
- directory mode parallelizes across files (best for many small/medium FASTA files)
- terminal '*' stop markers are stripped in memory before digestion

Requires Rapid Peptides Generator (RPG) package: pip install rpg==2.0.5
"""

import gc
import logging
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

try:
    from rpg import RapidPeptidesGenerator as rpg_main
    from rpg import core
    from rpg import digest
    from rpg import sequence as rpg_sequence
    from rpg.enzymes_definition import AVAILABLE_ENZYMES

    sys.path.insert(0, str(Path.home()))
    try:
        from rpg_user import AVAILABLE_ENZYMES_USER  # type: ignore

        ALL_ENZYMES = AVAILABLE_ENZYMES + AVAILABLE_ENZYMES_USER
    except ImportError:
        ALL_ENZYMES = AVAILABLE_ENZYMES
except ImportError:
    print(
        "Error: Could not import RPG package. Make sure it's installed: pip install rpg==2.0.5",
        file=sys.stderr,
    )
    sys.exit(1)


WRITE_BUFFER_LINES = 5000
PARALLEL_BATCH_MIN_RECORDS = 128
PARALLEL_BATCH_TARGET_RESIDUES = 500_000
PARALLEL_FILE_SIZE_THRESHOLD = 64 * 1024 * 1024
PARALLEL_MAX_PENDING_MULTIPLIER = 2

_BATCH_WORKER_CONTEXT = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _format_elapsed_seconds(elapsed_seconds):
    try:
        elapsed = float(elapsed_seconds)
    except (TypeError, ValueError):
        elapsed = 0.0
    elapsed = max(0.0, elapsed)
    minutes = int(elapsed // 60)
    seconds = elapsed - (minutes * 60)
    return f"{minutes} min {seconds:05.2f} s"


@contextmanager
def timer(description="Execution time"):
    """Context manager for timing operations."""
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"{description}: {_format_elapsed_seconds(elapsed)}")


@contextmanager
def open_output_stream(output_file=None):
    """Yield a writable text stream without closing stdout."""
    if output_file:
        with open(output_file, "w", encoding="utf-8") as handle:
            yield handle
    else:
        yield sys.stdout


@lru_cache(maxsize=1024)
def process_header(header, short_header=True):
    """Process and cache header transformation for better performance."""
    if short_header and " " in header:
        return header.split(" ")[0]
    return header


def new_digest_stats():
    """Create a stats dictionary for one digest run."""
    return {
        "records": 0,
        "records_with_terminal_stop": 0,
        "total_terminal_stops": 0,
        "records_with_internal_stop": 0,
        "internal_stop_examples": [],
        "skipped_empty_records": 0,
        "total_original_peptides": 0,
        "total_kept_peptides": 0,
    }


def merge_digest_stats(target, source):
    """Merge digest stats in place."""
    for key in (
        "records",
        "records_with_terminal_stop",
        "total_terminal_stops",
        "records_with_internal_stop",
        "skipped_empty_records",
        "total_original_peptides",
        "total_kept_peptides",
    ):
        target[key] += source[key]

    for example in source["internal_stop_examples"]:
        if len(target["internal_stop_examples"]) >= 5:
            break
        if example not in target["internal_stop_examples"]:
            target["internal_stop_examples"].append(example)


def log_sequence_quality_stats(stats):
    """Log FASTA sanitation statistics."""
    if stats["records_with_terminal_stop"] > 0:
        logger.info(
            "Removed terminal '*' stop markers from %s protein sequence(s) (%s total markers) before digestion",
            stats["records_with_terminal_stop"],
            stats["total_terminal_stops"],
        )

    if stats["records_with_internal_stop"] > 0:
        examples = ", ".join(stats["internal_stop_examples"])
        logger.warning(
            "Found internal '*' in %s protein sequence(s); these were left unchanged. Example headers: %s",
            stats["records_with_internal_stop"],
            examples,
        )

    if stats["skipped_empty_records"] > 0:
        logger.warning(
            "Skipped %s protein sequence(s) that became empty after trimming terminal '*'",
            stats["skipped_empty_records"],
        )


def log_peptide_retention_stats(stats):
    """Log peptide filtering retention."""
    total_original = stats["total_original_peptides"]
    total_kept = stats["total_kept_peptides"]

    if total_original > 0:
        retention_rate = (total_kept / total_original) * 100
        logger.info(
            f"Length filter: kept {total_kept} out of {total_original} peptides "
            f"(retention rate: {retention_rate:.2f}%)"
        )


def flush_buffer(buffer, out_handle):
    """Write a buffered chunk of lines to the output handle."""
    if buffer:
        out_handle.writelines(buffer)
        buffer.clear()


def iter_fasta_records(file_path):
    """Yield FASTA records as (header, sequence) tuples."""
    header = None
    seq_lines = []

    with open(file_path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_lines)
                header = line
                seq_lines = []
                continue

            if header is None:
                raise ValueError(
                    f"Invalid FASTA format in {file_path}: sequence content found before header at line {line_number}"
                )

            seq_lines.append(line)

    if header is not None:
        yield header, "".join(seq_lines)


def iter_fasta_record_batches(file_path, min_records=PARALLEL_BATCH_MIN_RECORDS, target_residues=PARALLEL_BATCH_TARGET_RESIDUES):
    """Yield batches of FASTA records sized for balanced parallel digestion."""
    batch = []
    total_residues = 0

    for header, sequence in iter_fasta_records(file_path):
        batch.append((header, sequence))
        total_residues += len(sequence)

        if len(batch) >= min_records and total_residues >= target_residues:
            yield batch
            batch = []
            total_residues = 0

    if batch:
        yield batch


def normalize_protein_sequence(header, sequence, stats):
    """Trim terminal stop markers while tracking sanitation stats."""
    stats["records"] += 1

    trimmed_sequence = sequence.rstrip("*")
    terminal_stop_count = len(sequence) - len(trimmed_sequence)

    if terminal_stop_count > 0:
        stats["records_with_terminal_stop"] += 1
        stats["total_terminal_stops"] += terminal_stop_count

    if "*" in trimmed_sequence:
        stats["records_with_internal_stop"] += 1
        example_header = header[1:] if header.startswith(">") else header
        if (
            len(stats["internal_stop_examples"]) < 5
            and example_header not in stats["internal_stop_examples"]
        ):
            stats["internal_stop_examples"].append(example_header)

    if not trimmed_sequence:
        stats["skipped_empty_records"] += 1
        return None

    return trimmed_sequence


def create_digest_context(enzyme_id, max_num_miscleavages=0):
    """Create reusable digestion settings for one run."""
    aa_mass = core.AA_MASS_AVERAGE
    water_mass = core.WATER_MASS
    aa_pka = core.AA_PKA_IPC_2

    rpg_main.restricted_enzyme_id(enzyme_id)
    enzymes_to_use = rpg_main.create_enzymes_to_use([enzyme_id], [0])

    if not enzymes_to_use:
        raise ValueError(f"Error: Could not create enzyme {enzyme_id}")

    enzyme_name = enzymes_to_use[0].name
    miscleavage_dict = None
    if max_num_miscleavages > 0:
        miscleavage_dict = {enzyme_name: int(max_num_miscleavages)}

    return {
        "aa_mass": aa_mass,
        "water_mass": water_mass,
        "aa_pka": aa_pka,
        "enzymes_to_use": enzymes_to_use,
        "enzyme_name": enzyme_name,
        "miscleavage_dict": miscleavage_dict,
    }


def digest_one_record(header, sequence, digest_context):
    """Digest one protein sequence using RPG in-memory APIs."""
    header_text = header[1:] if header.startswith(">") else header
    seq_obj = rpg_sequence.Sequence(header_text, rpg_sequence.check_sequence(sequence))

    one_seq_digested = digest.digest_one_sequence(
        seq_obj,
        digest_context["enzymes_to_use"],
        "sequential",
        digest_context["aa_pka"],
        digest_context["aa_mass"],
        digest_context["water_mass"],
    )

    if digest_context["miscleavage_dict"] is not None:
        digest.theoretical_peptides([one_seq_digested], digest_context["miscleavage_dict"])

    return one_seq_digested


def emit_filtered_peptides(one_seq_digested, out_handle, min_length, max_length, short_header=True, buffer=None):
    """Filter peptides by length and write them to an output handle."""
    if buffer is None:
        buffer = []

    total_original = 0
    total_kept = 0

    for one_enz_res in one_seq_digested:
        for peptide in one_enz_res.peptides:
            total_original += 1
            if min_length <= peptide.size <= max_length:
                header = process_header(peptide.header, short_header)
                buffer.append(f"{header}\t{peptide.sequence}\n")
                total_kept += 1

                if len(buffer) >= WRITE_BUFFER_LINES:
                    flush_buffer(buffer, out_handle)

    return total_original, total_kept


def digest_records_to_handle(records, digest_context, out_handle, min_length, max_length, short_header=True):
    """Digest an iterable of FASTA records and stream filtered peptides to output."""
    stats = new_digest_stats()
    buffer = []

    for header, raw_sequence in records:
        normalized_sequence = normalize_protein_sequence(header, raw_sequence, stats)
        if normalized_sequence is None:
            continue

        one_seq_digested = digest_one_record(header, normalized_sequence, digest_context)
        total_original, total_kept = emit_filtered_peptides(
            one_seq_digested,
            out_handle,
            min_length,
            max_length,
            short_header=short_header,
            buffer=buffer,
        )
        stats["total_original_peptides"] += total_original
        stats["total_kept_peptides"] += total_kept

    flush_buffer(buffer, out_handle)
    return stats


def write_header(out_handle):
    """Write the digest TSV header."""
    out_handle.write("Protein\tPeptide\n")


def copy_text_file_to_stream(input_file, out_handle):
    """Append a text file to a writable stream."""
    with open(input_file, "r", encoding="utf-8") as in_handle:
        shutil.copyfileobj(in_handle, out_handle, length=1024 * 1024)


def resolve_process_count(processes):
    """Normalize requested process count against local CPU limits."""
    cpu_count = os.cpu_count() or 1

    if processes is None:
        processes = cpu_count
    else:
        processes = max(1, int(processes))

    if sys.platform == "win32" and processes > 60:
        logger.warning(f"Windows system detected: Process count {processes} exceeds handle limit, adjusted to 60")
        processes = 60
    elif processes > cpu_count:
        logger.warning(f"Warning: Process count {processes} exceeds system CPU cores {cpu_count}, adjusted")
        processes = cpu_count

    return processes


def is_fasta_file(file_path):
    """Check if a file is in FASTA format by examining its first line."""
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            first_line = handle.readline().strip()
            return first_line.startswith(">")
    except Exception:
        return False


def process_fasta_streaming(input_file, digest_context, output_file=None, min_length=7, max_length=30, short_header=True):
    """Digest a FASTA file serially using a streaming in-memory pipeline."""
    with open_output_stream(output_file) as out_handle:
        write_header(out_handle)

        with timer("Digestion process..."):
            stats = digest_records_to_handle(
                iter_fasta_records(input_file),
                digest_context,
                out_handle,
                min_length=min_length,
                max_length=max_length,
                short_header=short_header,
            )

    log_sequence_quality_stats(stats)
    log_peptide_retention_stats(stats)
    return stats["total_kept_peptides"]


def init_batch_worker(enzyme_id, max_num_miscleavages, min_length, max_length, short_header):
    """Initialize one worker for batch-based single-file digestion."""
    global _BATCH_WORKER_CONTEXT
    _BATCH_WORKER_CONTEXT = {
        "digest_context": create_digest_context(enzyme_id, max_num_miscleavages),
        "min_length": int(min_length),
        "max_length": int(max_length),
        "short_header": bool(short_header),
    }


def digest_batch_worker(batch_index, records):
    """Digest one batch of protein records and write its output to a temp file."""
    if _BATCH_WORKER_CONTEXT is None:
        raise RuntimeError("Batch worker was not initialized")

    fd, temp_name = tempfile.mkstemp(prefix=f"digest_batch_{batch_index:06d}_", suffix=".tsv")
    os.close(fd)
    temp_path = Path(temp_name)

    try:
        with open(temp_path, "w", encoding="utf-8") as out_handle:
            stats = digest_records_to_handle(
                records,
                _BATCH_WORKER_CONTEXT["digest_context"],
                out_handle,
                min_length=_BATCH_WORKER_CONTEXT["min_length"],
                max_length=_BATCH_WORKER_CONTEXT["max_length"],
                short_header=_BATCH_WORKER_CONTEXT["short_header"],
            )
        return {
            "batch_index": batch_index,
            "part_path": str(temp_path),
            "stats": stats,
        }
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def process_fasta_parallel(input_file, enzyme_id, output_file=None, min_length=7, max_length=30, max_num_miscleavages=0, processes=2, short_header=True):
    """Digest one large FASTA file using process-parallel record batches."""
    total_stats = new_digest_stats()
    temp_parts_to_cleanup = set()
    completed_parts = {}
    next_batch_to_write = 0
    pending_futures = {}
    batch_index = 0
    no_more_batches = False

    max_workers = resolve_process_count(processes)
    max_pending = max_workers * PARALLEL_MAX_PENDING_MULTIPLIER
    batch_iter = iter_fasta_record_batches(input_file)

    try:
        with open_output_stream(output_file) as out_handle:
            write_header(out_handle)

            with timer("Digestion process..."):
                with ProcessPoolExecutor(
                    max_workers=max_workers,
                    initializer=init_batch_worker,
                    initargs=(enzyme_id, max_num_miscleavages, min_length, max_length, short_header),
                ) as executor:
                    try:
                        while pending_futures or not no_more_batches:
                            while not no_more_batches and len(pending_futures) < max_pending:
                                try:
                                    records = next(batch_iter)
                                except StopIteration:
                                    no_more_batches = True
                                    break

                                future = executor.submit(digest_batch_worker, batch_index, records)
                                pending_futures[future] = batch_index
                                batch_index += 1

                            if not pending_futures:
                                break

                            done, _ = wait(pending_futures, return_when=FIRST_COMPLETED)
                            for future in done:
                                pending_futures.pop(future)
                                result = future.result()
                                completed_parts[result["batch_index"]] = result
                                temp_parts_to_cleanup.add(result["part_path"])

                            while next_batch_to_write in completed_parts:
                                result = completed_parts.pop(next_batch_to_write)
                                copy_text_file_to_stream(result["part_path"], out_handle)
                                merge_digest_stats(total_stats, result["stats"])
                                Path(result["part_path"]).unlink(missing_ok=True)
                                temp_parts_to_cleanup.discard(result["part_path"])
                                next_batch_to_write += 1
                    finally:
                        for future in pending_futures:
                            future.cancel()
    finally:
        for part_path in list(temp_parts_to_cleanup):
            Path(part_path).unlink(missing_ok=True)

    log_sequence_quality_stats(total_stats)
    log_peptide_retention_stats(total_stats)
    return total_stats["total_kept_peptides"]


def process_directory_file_worker(task):
    """Worker entrypoint for directory-level parallel digestion."""
    input_file = task["input_file"]

    try:
        peptide_count = process_fasta(
            input_file=input_file,
            enzyme_id=task["enzyme_id"],
            output_file=task["output_file"],
            min_length=task["min_length"],
            max_length=task["max_length"],
            max_num_miscleavages=task["max_num_miscleavages"],
            processes=1,
            short_header=task["short_header"],
            verbose=task["verbose"],
        )
        return {
            "file_name": Path(input_file).name,
            "peptide_count": peptide_count,
            "error": None,
        }
    except Exception as exc:
        return {
            "file_name": Path(input_file).name,
            "peptide_count": None,
            "error": str(exc),
        }


def process_directory(input_dir, output_dir, enzyme_id, min_length=7, max_length=30, max_num_miscleavages=0, processes=None, short_header=True, verbose=True, skip_existing=False):
    """
    Process all FASTA files in a directory and save results to another directory.

    In directory mode, `processes` means the number of files digested in parallel.
    Each file is processed with the optimized single-file streaming pipeline.
    """
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        error_msg = f"Error: Input directory {input_dir} does not exist or is not a directory"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    output_path = Path(output_dir)
    if not output_path.exists():
        logger.info(f"Creating output directory: {output_dir}")
        output_path.mkdir(parents=True, exist_ok=True)
    elif not output_path.is_dir():
        error_msg = f"Error: Output path {output_dir} exists but is not a directory"
        logger.error(error_msg)
        raise NotADirectoryError(error_msg)

    results = {}
    processed_files = 0
    skipped_files = 0
    process_errors = 0

    all_files = list(input_path.glob("*"))
    logger.info(f"Found {len(all_files)} files in {input_dir}")

    file_workers = resolve_process_count(processes)
    files_to_process = []

    for file_path in all_files:
        if not file_path.is_file():
            logger.debug(f"Skipping non-file: {file_path}")
            continue

        if not is_fasta_file(file_path):
            logger.warning(f"Skipping non-FASTA file: {file_path}")
            skipped_files += 1
            continue

        output_file = output_path / f"{file_path.stem}.tsv"

        if skip_existing and output_file.exists():
            logger.info(f"Skipping existing output file: {output_file}")
            skipped_files += 1
            continue

        files_to_process.append(
            {
                "input_file": str(file_path),
                "output_file": str(output_file),
                "enzyme_id": enzyme_id,
                "min_length": min_length,
                "max_length": max_length,
                "max_num_miscleavages": max_num_miscleavages,
                "short_header": short_header,
                "verbose": verbose,
            }
        )

    if not files_to_process:
        logger.info(
            f"Directory processing complete: {processed_files} files processed, "
            f"{skipped_files} files skipped, {process_errors} files failed"
        )
        return results

    if len(files_to_process) == 1:
        task = files_to_process[0]
        file_name = Path(task["input_file"]).name
        peptide_count = process_fasta(
            input_file=task["input_file"],
            enzyme_id=task["enzyme_id"],
            output_file=task["output_file"],
            min_length=task["min_length"],
            max_length=task["max_length"],
            max_num_miscleavages=task["max_num_miscleavages"],
            processes=file_workers,
            short_header=task["short_header"],
            verbose=task["verbose"],
        )
        results[file_name] = peptide_count
        processed_files += 1
    elif file_workers > 1:
        max_workers = min(file_workers, len(files_to_process))
        logger.info(
            f"Directory mode: dispatching {len(files_to_process)} files across {max_workers} worker processes"
        )

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(process_directory_file_worker, task): Path(task["input_file"]).name
                for task in files_to_process
            }

            for completed_idx, future in enumerate(as_completed(future_map), start=1):
                expected_file_name = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:
                    process_errors += 1
                    logger.error(f"Worker crashed while processing {expected_file_name}: {exc}")
                    continue

                if result["error"] is None:
                    results[result["file_name"]] = result["peptide_count"]
                    processed_files += 1
                    logger.info(
                        f"Completed file {completed_idx}/{len(files_to_process)}: {result['file_name']}"
                    )
                else:
                    process_errors += 1
                    logger.error(f"Error processing {result['file_name']}: {result['error']}")
    else:
        for file_idx, task in enumerate(files_to_process, start=1):
            file_name = Path(task["input_file"]).name
            try:
                logger.info(f"Processing file {file_idx}/{len(files_to_process)}: {file_name}")
                peptide_count = process_fasta(
                    input_file=task["input_file"],
                    enzyme_id=task["enzyme_id"],
                    output_file=task["output_file"],
                    min_length=task["min_length"],
                    max_length=task["max_length"],
                    max_num_miscleavages=task["max_num_miscleavages"],
                    processes=1,
                    short_header=task["short_header"],
                    verbose=task["verbose"],
                )
                results[file_name] = peptide_count
                processed_files += 1
            except Exception as exc:
                process_errors += 1
                logger.error(f"Error processing {file_name}: {exc}")

    logger.info(
        f"Directory processing complete: {processed_files} files processed, "
        f"{skipped_files} files skipped, {process_errors} files failed"
    )
    return results


def process_fasta(input_file, enzyme_id, output_file=None, min_length=7, max_length=30, max_num_miscleavages=0, processes=None, short_header=True, verbose=True):
    """
    Process a protein FASTA file and write filtered theoretical peptides.

    Single-file mode behavior:
    - `processes == 1`: streaming serial digestion in memory
    - `processes > 1`: parallel in-memory digestion over record batches for large files
    """
    logger.setLevel(logging.INFO if verbose else logging.WARNING)

    input_path = Path(input_file)
    if not input_path.exists():
        error_msg = f"Error: Input file {input_file} does not exist"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    if int(min_length) > int(max_length):
        error_msg = "Error: Minimum peptide length cannot be greater than maximum peptide length"
        logger.error(error_msg)
        raise ValueError(error_msg)

    processes = resolve_process_count(processes)

    try:
        with timer("Creating enzyme object"):
            digest_context = create_digest_context(enzyme_id, max_num_miscleavages)
    except Exception as exc:
        logger.error(f"Error creating enzyme: {exc}")
        raise

    logger.info(f"Processing file: {input_file}")
    logger.info(f"Using enzyme: {digest_context['enzyme_name']}")
    logger.info(f"Configured worker processes: {processes}")
    logger.info(f"Peptide length filter: {min_length}-{max_length}")

    file_size = input_path.stat().st_size
    use_parallel_file_mode = processes > 1 and file_size >= PARALLEL_FILE_SIZE_THRESHOLD

    if processes > 1 and not use_parallel_file_mode:
        logger.info(
            f"Input file is {file_size / (1024 * 1024):.2f} MB; "
            "using streaming serial digestion because parallel batch overhead would dominate"
        )

    try:
        if use_parallel_file_mode:
            logger.info(f"Single-file parallel mode enabled with {processes} worker processes")
            total_peptides = process_fasta_parallel(
                input_file=input_file,
                enzyme_id=enzyme_id,
                output_file=output_file,
                min_length=min_length,
                max_length=max_length,
                max_num_miscleavages=max_num_miscleavages,
                processes=processes,
                short_header=short_header,
            )
        else:
            logger.info("Single-file streaming mode enabled")
            total_peptides = process_fasta_streaming(
                input_file=input_file,
                digest_context=digest_context,
                output_file=output_file,
                min_length=min_length,
                max_length=max_length,
                short_header=short_header,
            )

        gc.collect()

        if output_file:
            logger.info(f"Processing complete. Generated {total_peptides} peptides, saved to {output_file}")
        else:
            logger.info(f"Processing complete. Generated {total_peptides} peptides.")

        return total_peptides
    except Exception as exc:
        logger.error(f"Error during processing: {exc}")
        raise


if __name__ == "__main__":
    if __package__ in {None, ""}:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from metaumbra.cli import main as cli_main
    else:
        from .cli import main as cli_main
    raise SystemExit(cli_main(["digest", *sys.argv[1:]]))
