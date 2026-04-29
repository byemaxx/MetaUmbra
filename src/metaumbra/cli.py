from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metaumbra import __version__
    from metaumbra.workflows import (
        DigestConfig,
        ParquetExtractionConfig,
        ScoringConfig,
        run_digest_workflow,
        run_parquet_extraction_workflow,
        run_scoring_workflow,
    )
else:
    from . import __version__
    from .workflows import (
        DigestConfig,
        ParquetExtractionConfig,
        ScoringConfig,
        run_digest_workflow,
        run_parquet_extraction_workflow,
        run_scoring_workflow,
    )


def _print_result(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _add_common_version_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="metaumbra",
        description="MetaUmbra packaging-friendly command line interface.",
    )
    _add_common_version_flag(parser)
    subparsers = parser.add_subparsers(dest="command")

    gui_parser = subparsers.add_parser("gui", help="Launch the Qt GUI.")
    _add_common_version_flag(gui_parser)

    digest_parser = subparsers.add_parser("digest", help="Digest FASTA files into peptide tables.")
    _add_common_version_flag(digest_parser)
    digest_input = digest_parser.add_mutually_exclusive_group(required=True)
    digest_input.add_argument("--input-file", help="Single FASTA file to digest.")
    digest_input.add_argument("--input-dir", help="Directory of FASTA files to digest.")
    digest_parser.add_argument("--output-file", help="Output TSV path for single-file mode.")
    digest_parser.add_argument("--output-dir", help="Output directory for directory mode.")
    digest_parser.add_argument("--enzyme-id", default="42", help="RPG enzyme ID. Default: 42 (Trypsin).")
    digest_parser.add_argument("--min-length", type=int, default=7, help="Minimum peptide length.")
    digest_parser.add_argument("--max-length", type=int, default=30, help="Maximum peptide length.")
    digest_parser.add_argument("--max-miscleavages", type=int, default=2, help="Maximum missed cleavages.")
    digest_parser.add_argument("--processes", type=int, help="Worker process count.")
    digest_parser.add_argument(
        "--full-header",
        action="store_true",
        help="Keep full FASTA headers instead of truncating at the first space.",
    )
    digest_parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Rebuild existing output files in directory mode.",
    )
    digest_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce runtime log output.",
    )

    score_parser = subparsers.add_parser("score", help="Score genome presence from peptide observations.")
    _add_common_version_flag(score_parser)
    score_parser.add_argument("--peptide-table", required=True, help="Observed peptide TSV path.")
    score_parser.add_argument(
        "--genome-digest-dir",
        action="append",
        required=True,
        help="Genome digest directory. Repeat for multiple directories.",
    )
    score_parser.add_argument("--output", required=True, help="Output TSV path.")
    score_parser.add_argument("--peptide-seq-col", default="Sequence", help="Peptide sequence column name.")
    score_parser.add_argument("--peptide-score-col", default="score", help="Peptide score column name.")
    score_parser.add_argument("--peptide-error-col", default="Q.Value", help="Peptide error/FDR column name.")
    score_parser.add_argument("--peptide-error-cutoff", type=float, default=0.05, help="Peptide error cutoff.")
    score_parser.add_argument(
        "--peptide-decoy-flag-col",
        default="Reverse",
        help="Optional decoy flag column. Pass an empty string to disable it.",
    )
    score_parser.add_argument("--decoy-flag-value", default="+", help="Decoy marker value.")
    score_parser.add_argument("--num-workers", type=int, help="Worker process count.")
    score_parser.add_argument(
        "--selected-genome-id",
        action="append",
        default=[],
        help="Restrict scoring to specific genome IDs. Repeat as needed.",
    )
    score_parser.add_argument(
        "--exclude-genome-id",
        action="append",
        default=[],
        help="Genome IDs to exclude. Repeat as needed.",
    )
    score_parser.add_argument("--lineage-table", default="", help="Optional genome lineage table.")
    score_parser.add_argument("--lineage-genome-id-col", default="", help="Genome ID column in the lineage table.")
    score_parser.add_argument("--lineage-lineage-col", default="", help="Lineage column in the lineage table.")
    score_parser.add_argument("--cache-path", default="", help="Optional matched peptide cache path.")
    score_parser.add_argument(
        "--use-cache-if-exists",
        action="store_true",
        help="Reuse an existing matched peptide cache if available.",
    )
    score_parser.add_argument(
        "--no-save-cache",
        action="store_true",
        help="Do not persist matched peptide cache output.",
    )
    score_parser.add_argument(
        "--no-compute-coverage",
        action="store_true",
        help="Skip cumulative coverage calculations.",
    )
    score_parser.add_argument(
        "--no-export-temp",
        action="store_true",
        help="Skip temporary artifact exports.",
    )
    score_parser.add_argument(
        "--return-full-table",
        action="store_true",
        help="Return and write the full internal result table.",
    )

    parquet_parser = subparsers.add_parser(
        "extract-parquet",
        help="Extract selected columns from a parquet peptide table into TSV.",
    )
    _add_common_version_flag(parquet_parser)
    parquet_parser.add_argument("--input", required=True, help="Input parquet file.")
    parquet_parser.add_argument("--output", required=True, help="Output TSV file.")
    parquet_parser.add_argument(
        "--input-column",
        action="append",
        default=[],
        help="Input column to extract. Repeat to control order.",
    )
    parquet_parser.add_argument(
        "--output-column",
        action="append",
        default=[],
        help="Output column name. Repeat to match --input-column order.",
    )
    parquet_parser.add_argument("--batch-size", type=int, default=65536, help="Parquet streaming batch size.")
    parquet_parser.add_argument("--force", action="store_true", help="Overwrite an existing TSV output.")

    return parser


def _run_gui() -> int:
    from .gui import main as gui_main

    gui_main()
    return 0


def _run_digest(args: argparse.Namespace) -> int:
    input_mode = "file" if args.input_file else "directory"
    if input_mode == "file" and not args.output_file:
        raise SystemExit("--output-file is required when using --input-file.")
    if input_mode == "directory" and not args.output_dir:
        raise SystemExit("--output-dir is required when using --input-dir.")

    config = DigestConfig(
        input_mode=input_mode,
        input_file=args.input_file or "",
        input_dir=args.input_dir or "",
        output_file=args.output_file or "",
        output_dir=args.output_dir or "",
        enzyme_id=str(args.enzyme_id),
        min_length=args.min_length,
        max_length=args.max_length,
        max_num_miscleavages=args.max_miscleavages,
        processes=args.processes,
        short_header=not args.full_header,
        verbose=not args.quiet,
        skip_existing=not args.no_skip_existing,
    )
    _print_result(run_digest_workflow(config))
    return 0


def _run_score(args: argparse.Namespace) -> int:
    config = ScoringConfig(
        peptide_table_path=args.peptide_table,
        genome_lineage_table_path=args.lineage_table,
        genome_lineage_genome_id_col=args.lineage_genome_id_col,
        genome_lineage_lineage_col=args.lineage_lineage_col,
        genome_digest_dirs=args.genome_digest_dir,
        selected_genome_ids=args.selected_genome_id,
        output_tsv_path=args.output,
        peptide_seq_col=args.peptide_seq_col,
        peptide_score_col=args.peptide_score_col,
        peptide_error_col=args.peptide_error_col,
        peptide_error_cutoff=args.peptide_error_cutoff,
        peptide_decoy_flag_col=args.peptide_decoy_flag_col,
        decoy_flag_value=args.decoy_flag_value,
        exclude_genome_ids=args.exclude_genome_id,
        num_workers=args.num_workers,
        matched_peptides_cache_path=args.cache_path,
        save_matched_peptides_cache=not args.no_save_cache,
        use_cache_if_exists=args.use_cache_if_exists,
        compute_coverage=not args.no_compute_coverage,
        export_temp=not args.no_export_temp,
        return_full_table=args.return_full_table,
    )
    _print_result(run_scoring_workflow(config))
    return 0


def _run_parquet_extraction(args: argparse.Namespace) -> int:
    input_columns = args.input_column or ["Run", "Stripped.Sequence", "Evidence", "Q.Value"]
    output_columns = args.output_column or ["Run", "Sequence", "score", "Q.Value"]

    config = ParquetExtractionConfig(
        input_parquet_path=args.input,
        output_tsv_path=args.output,
        input_columns=input_columns,
        output_columns=output_columns,
        batch_size=args.batch_size,
        force=args.force,
    )
    _print_result(run_parquet_extraction_workflow(config))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "gui":
        return _run_gui()
    if args.command == "digest":
        return _run_digest(args)
    if args.command == "score":
        return _run_score(args)
    if args.command == "extract-parquet":
        return _run_parquet_extraction(args)

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
