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


DEFAULT_PARQUET_INPUT_COLUMNS = ["Run", "Stripped.Sequence", "Evidence", "Q.Value"]
DEFAULT_PARQUET_OUTPUT_COLUMNS = ["Run", "Sequence", "score", "Q.Value"]


class MetaUmbraHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Help formatter that labels optionality and shows effective defaults."""

    def _get_help_string(self, action: argparse.Action) -> str:
        help_text = action.help or ""
        if not action.option_strings or isinstance(action, (argparse._HelpAction, argparse._VersionAction)):
            return help_text

        display_default = getattr(action, "display_default", action.default)
        if display_default is not argparse.SUPPRESS and display_default is not None:
            return f"{help_text} (default: {self._format_display_default(display_default)})"

        return help_text

    @staticmethod
    def _format_display_default(value: Any) -> str:
        if isinstance(value, bool):
            return str(value).lower()
        if value == "":
            return "none"
        if isinstance(value, (list, tuple)):
            if not value:
                return "none"
            return "[" + ", ".join(str(item) for item in value) + "]"
        return str(value)


def _add_argument(
    parser: argparse.ArgumentParser | argparse._ArgumentGroup | argparse._MutuallyExclusiveGroup,
    *name_or_flags: str,
    display_default: Any = None,
    **kwargs: Any,
) -> argparse.Action:
    action = parser.add_argument(*name_or_flags, **kwargs)
    if display_default is not None:
        setattr(action, "display_default", display_default)
    return action


def _add_common_version_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="metaumbra",
        description="MetaUmbra command line interface.",
        epilog="Run `metaumbra <command> --help` for command-specific options and defaults.",
        formatter_class=MetaUmbraHelpFormatter,
    )
    _add_common_version_flag(parser)
    subparsers = parser.add_subparsers(dest="command", title="commands")

    gui_parser = subparsers.add_parser(
        "gui",
        help="Launch the Qt GUI.",
        description="Launch the MetaUmbra Qt GUI.",
        formatter_class=MetaUmbraHelpFormatter,
    )
    _add_common_version_flag(gui_parser)

    digest_parser = subparsers.add_parser(
        "digest",
        help="Digest FASTA files into peptide tables.",
        description="Digest protein FASTA files into theoretical peptide TSV tables.",
        formatter_class=MetaUmbraHelpFormatter,
    )
    _add_common_version_flag(digest_parser)
    digest_input = digest_parser.add_mutually_exclusive_group(required=True)
    digest_required = digest_parser.add_argument_group("Required arguments")
    digest_optional = digest_parser.add_argument_group("Optional arguments")

    _add_argument(
        digest_required,
        "--input-file",
        help="Input FASTA file to digest in single-file mode. Mutually exclusive with --input-dir.",
    )
    _add_argument(
        digest_required,
        "--input-dir",
        help="Directory of FASTA files to digest in batch mode. Mutually exclusive with --input-file.",
    )
    _add_argument(
        digest_required,
        "--output-file",
        help="Output TSV path for single-file mode. Required when --input-file is used.",
    )
    _add_argument(
        digest_required,
        "--output-dir",
        help="Output directory for directory mode. Required when --input-dir is used.",
    )
    _add_argument(digest_optional, "--enzyme-id", default="42", help="RPG enzyme ID.")
    _add_argument(digest_optional, "--min-length", type=int, default=7, help="Minimum peptide length to keep.")
    _add_argument(digest_optional, "--max-length", type=int, default=30, help="Maximum peptide length to keep.")
    _add_argument(
        digest_optional,
        "--max-miscleavages",
        type=int,
        default=2,
        help="Maximum number of missed cleavages.",
    )
    _add_argument(
        digest_optional,
        "--processes",
        type=int,
        display_default="all CPU cores",
        help="Worker process count. Use 1 for serial streaming; values above 1 allow parallel digestion for large files.",
    )
    _add_argument(
        digest_optional,
        "--full-header",
        action="store_true",
        help="Keep full FASTA headers instead of truncating at the first space.",
    )
    _add_argument(
        digest_optional,
        "--no-skip-existing",
        action="store_true",
        help="Rebuild existing output files in directory mode instead of skipping them.",
    )
    _add_argument(
        digest_optional,
        "--quiet",
        action="store_true",
        help="Reduce runtime log output.",
    )

    score_parser = subparsers.add_parser(
        "score",
        help="Score genome presence from peptide observations.",
        description="Score genome presence by matching observed peptides against genome digest tables.",
        formatter_class=MetaUmbraHelpFormatter,
    )
    _add_common_version_flag(score_parser)
    score_required = score_parser.add_argument_group("Required arguments")
    score_optional = score_parser.add_argument_group("Optional arguments")

    _add_argument(score_required, "--peptide-table", required=True, help="Observed peptide TSV path.")
    _add_argument(
        score_required,
        "--genome-digest-dir",
        action="append",
        required=True,
        help="Directory containing digested genome TSV files. Repeat for multiple directories.",
    )
    _add_argument(score_required, "--output", required=True, help="Output TSV path.")
    _add_argument(score_optional, "--peptide-seq-col", default="Sequence", help="Peptide sequence column name.")
    _add_argument(score_optional, "--peptide-score-col", default="score", help="Peptide score column name.")
    _add_argument(score_optional, "--peptide-error-col", default="Q.Value", help="Peptide error or FDR column name.")
    _add_argument(score_optional, "--peptide-error-cutoff", type=float, default=0.05, help="Peptide error cutoff.")
    _add_argument(
        score_optional,
        "--peptide-decoy-flag-col",
        default="Reverse",
        help="Optional decoy flag column. Pass an empty string to disable it.",
    )
    _add_argument(score_optional, "--decoy-flag-value", default="+", help="Decoy marker value.")
    _add_argument(
        score_optional,
        "--num-workers",
        type=int,
        display_default="max(1, cpu_count - 1)",
        help="Worker process count for genome scoring.",
    )
    _add_argument(
        score_optional,
        "--selected-genome-id",
        action="append",
        default=[],
        help="Restrict scoring to specific genome IDs. Repeat as needed.",
    )
    _add_argument(
        score_optional,
        "--exclude-genome-id",
        action="append",
        default=[],
        help="Genome IDs to exclude. Repeat as needed.",
    )
    _add_argument(
        score_optional,
        "--lineage-table",
        default="",
        display_default="none",
        help="Genome lineage table used to add lineage annotations.",
    )
    _add_argument(
        score_optional,
        "--lineage-genome-id-col",
        default="",
        display_default="none",
        help="Genome ID column in --lineage-table. Required when --lineage-table is set.",
    )
    _add_argument(
        score_optional,
        "--lineage-lineage-col",
        default="",
        display_default="none",
        help="Lineage column in --lineage-table. Required when --lineage-table is set.",
    )
    _add_argument(
        score_optional,
        "--cache-path",
        default="",
        display_default="none",
        help="Matched peptide cache file path. Leave unset to disable explicit cache output.",
    )
    _add_argument(
        score_optional,
        "--use-cache-if-exists",
        action="store_true",
        help="Reuse an existing matched peptide cache if available.",
    )
    _add_argument(
        score_optional,
        "--no-save-cache",
        action="store_true",
        help="Do not persist matched peptide cache output.",
    )
    _add_argument(
        score_optional,
        "--no-compute-coverage",
        action="store_true",
        help="Skip cumulative coverage calculations.",
    )
    _add_argument(
        score_optional,
        "--no-export-temp",
        action="store_true",
        help="Skip temporary artifact exports.",
    )
    _add_argument(
        score_optional,
        "--return-full-table",
        action="store_true",
        help="Return and write the full internal result table.",
    )

    parquet_parser = subparsers.add_parser(
        "extract-parquet",
        help="Extract selected columns from a parquet peptide table into TSV.",
        description="Extract selected columns from a parquet peptide table and write them to TSV.",
        formatter_class=MetaUmbraHelpFormatter,
    )
    _add_common_version_flag(parquet_parser)
    parquet_required = parquet_parser.add_argument_group("Required arguments")
    parquet_optional = parquet_parser.add_argument_group("Optional arguments")

    _add_argument(parquet_required, "--input", required=True, help="Input parquet file.")
    _add_argument(parquet_required, "--output", required=True, help="Output TSV file.")
    _add_argument(
        parquet_optional,
        "--input-column",
        action="append",
        default=None,
        display_default=DEFAULT_PARQUET_INPUT_COLUMNS,
        help="Input column to extract. Repeat to control order.",
    )
    _add_argument(
        parquet_optional,
        "--output-column",
        action="append",
        default=None,
        display_default=DEFAULT_PARQUET_OUTPUT_COLUMNS,
        help="Output column name. Repeat to match --input-column order.",
    )
    _add_argument(parquet_optional, "--batch-size", type=int, default=65536, help="Parquet streaming batch size.")
    _add_argument(parquet_optional, "--force", action="store_true", help="Overwrite an existing TSV output.")

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
    input_columns = args.input_column or DEFAULT_PARQUET_INPUT_COLUMNS
    output_columns = args.output_column or DEFAULT_PARQUET_OUTPUT_COLUMNS

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
