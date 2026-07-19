import json

import pandas as pd
import pytest

from metaumbra.cli import DEFAULT_PARQUET_INPUT_COLUMNS, DEFAULT_PARQUET_OUTPUT_COLUMNS
from metaumbra.scoring import GenomePresenceScorer
from metaumbra.workflows import (
    ParquetExtractionConfig,
    ScoringConfig,
    _clean_scoring_artifacts_for_new_run,
    migrate_legacy_scoring_config_payload,
    run_parquet_extraction_workflow,
    run_scoring_workflow,
)


def test_default_parquet_extraction_preserves_required_scoring_columns(tmp_path):
    pytest.importorskip("pyarrow")
    parquet_path = tmp_path / "report.parquet"
    output_path = tmp_path / "peptides.tsv"
    pd.DataFrame(
        {
            "Run": ["s1.raw"],
            "Stripped.Sequence": ["PEPTIDEA"],
            "Precursor.Quantity": [100.0],
            "Evidence": [1.0],
            "Q.Value": [0.01],
        }
    ).to_parquet(parquet_path, index=False)

    run_parquet_extraction_workflow(
        ParquetExtractionConfig(
            input_parquet_path=str(parquet_path),
            output_tsv_path=str(output_path),
            input_columns=list(DEFAULT_PARQUET_INPUT_COLUMNS),
            output_columns=list(DEFAULT_PARQUET_OUTPUT_COLUMNS),
        )
    )

    assert pd.read_csv(output_path, sep="\t", nrows=0).columns.tolist() == [
        "Run",
        "Sequence",
        "Precursor.Quantity",
        "Evidence",
        "Q.Value",
    ]
    scorer = GenomePresenceScorer(num_workers=1)
    scorer.read_analysis_unit_peptide_file(
        peptide_table_path=str(output_path),
        unit_mode="all-samples",
        sample_id_col="Run",
        peptide_seq_col="Sequence",
    )
    direct_scorer = GenomePresenceScorer(num_workers=1)
    direct_scorer.read_analysis_unit_peptide_file(
        peptide_table_path=str(parquet_path),
        unit_mode="all-samples",
        sample_id_col="Run",
        peptide_seq_col="Sequence",
    )
    assert scorer.unit_sample_ids == ["s1"]
    assert scorer.unit_sample_ids == direct_scorer.unit_sample_ids


def test_failed_directory_run_writes_status_to_results_artifacts(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    stale_outputs = [
        "unit_genome_results.tsv",
        "cohort_genome_summary.tsv",
        "sample_unit_mapping.tsv",
        "genome_selection_manifest.json",
    ]
    for name in stale_outputs:
        (results_dir / name).write_text("stale", encoding="utf-8")
    unrelated_path = results_dir / "notes.txt"
    unrelated_path.write_text("keep", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Peptide file does not exist"):
        run_scoring_workflow(
            ScoringConfig(
                peptide_table_path=str(tmp_path / "missing.tsv"),
                genome_digest_dirs=[str(tmp_path / "digests")],
                output_tsv_path=str(results_dir),
            )
        )

    status_path = results_dir / "artifacts" / "run_status.json"
    assert json.loads(status_path.read_text(encoding="utf-8"))["status"] == "failed"
    assert not any((results_dir / name).exists() for name in stale_outputs)
    assert unrelated_path.read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / "artifacts").exists()


def test_artifact_cleanup_preserves_configured_inputs_and_unknown_files(tmp_path):
    artifact_dir = tmp_path / "results" / "artifacts"
    diagnostics_dir = artifact_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True)
    generated_paths = [
        artifact_dir / "run_summary.json",
        diagnostics_dir / "full_internal_metrics.tsv",
        diagnostics_dir / "top5_peptide_contrib.tsv",
        diagnostics_dir / "unit_call_counts.tsv",
        diagnostics_dir / "unit_genome_presence_full.tsv",
        diagnostics_dir / "genome_union_q001.tsv",
        diagnostics_dir / "unit_empirical_background_calibration.tsv",
    ]
    for path in generated_paths:
        path.write_text("stale", encoding="utf-8")
    metadata_path = diagnostics_dir / "metadata.tsv"
    matched_cache_path = diagnostics_dir / "custom_matched.pkl"
    theoretical_cache_path = diagnostics_dir / "custom_theoretical.pkl"
    unknown_path = diagnostics_dir / "notes.tsv"
    for path in [metadata_path, matched_cache_path, theoretical_cache_path, unknown_path]:
        path.write_text("keep", encoding="utf-8")

    _clean_scoring_artifacts_for_new_run(
        artifact_dir,
        ScoringConfig(
            metadata_table_path=str(metadata_path),
            matched_peptides_cache_path=str(matched_cache_path),
            theoretical_opportunity_cache_path=str(theoretical_cache_path),
            use_cache_if_exists=True,
            unique_pvalue_mode="hypergeometric-opportunity",
        ),
    )

    assert not any(path.exists() for path in generated_paths)
    assert all(
        path.read_text(encoding="utf-8") == "keep"
        for path in [metadata_path, matched_cache_path, theoretical_cache_path, unknown_path]
    )


def test_scoring_output_cannot_overwrite_peptide_input(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    peptide_path = results_dir / "unit_genome_results.tsv"
    peptide_path.write_text("Run\tSequence\tPrecursor.Quantity\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must not overwrite"):
        run_scoring_workflow(
            ScoringConfig(
                peptide_table_path=str(peptide_path),
                genome_digest_dirs=[str(tmp_path / "digests")],
                output_tsv_path=str(results_dir),
            )
        )

    assert peptide_path.exists()


@pytest.mark.parametrize("suffix", [".tsv", ".TXT"])
def test_scoring_workflow_rejects_file_output_paths(tmp_path, suffix):
    output_path = tmp_path / f"run{suffix}"

    with pytest.raises(ValueError, match="must be a unified results directory"):
        run_scoring_workflow(
            ScoringConfig(
                peptide_table_path=str(tmp_path / "peptides.tsv"),
                genome_digest_dirs=[str(tmp_path / "digests")],
                output_tsv_path=str(output_path),
            )
        )

    assert not output_path.exists()
    assert not (tmp_path / "cohort_genome_summary.tsv").exists()
    assert not (tmp_path / "artifacts").exists()


def test_scoring_output_cannot_share_genome_digest_directory(tmp_path):
    peptide_path = tmp_path / "peptides.tsv"
    peptide_path.write_text("Sequence\nPEPTIDE\n", encoding="utf-8")
    digest_dir = tmp_path / "digests"
    digest_dir.mkdir()
    digest_path = digest_dir / "g1.tsv"
    digest_contents = "Peptide\nPEPTIDE\n"
    digest_path.write_text(digest_contents, encoding="utf-8")

    with pytest.raises(ValueError, match="must not be the same as a genome digest directory"):
        run_scoring_workflow(
            ScoringConfig(
                peptide_table_path=str(peptide_path),
                genome_digest_dirs=[str(digest_dir)],
                output_tsv_path=str(digest_dir),
            )
        )

    assert digest_path.read_text(encoding="utf-8") == digest_contents
    assert not (digest_dir / "unit_genome_results.tsv").exists()
    assert not (digest_dir / "artifacts").exists()


def test_all_samples_workflow_accepts_peptide_only_input(tmp_path):
    peptide_path = tmp_path / "peptides.tsv"
    pd.DataFrame(
        {
            "Sequence": ["PEPA", "PEPB"],
            "Evidence": [1.0, 2.0],
            "Q.Value": [0.01, 0.01],
        }
    ).to_csv(peptide_path, sep="\t", index=False)
    digest_dir = tmp_path / "digests"
    digest_dir.mkdir()
    pd.DataFrame({"Peptide": ["PEPA"]}).to_csv(
        digest_dir / "g1.tsv", sep="\t", index=False
    )
    pd.DataFrame({"Peptide": ["PEPB"]}).to_csv(
        digest_dir / "g2.tsv", sep="\t", index=False
    )
    results_dir = tmp_path / "results"

    result = run_scoring_workflow(
        ScoringConfig(
            peptide_table_path=str(peptide_path),
            genome_digest_dirs=[str(digest_dir)],
            output_tsv_path=str(results_dir),
            num_workers=1,
            knockoff_mc_iterations=20,
            knockoff_stage2_mc_iterations=None,
            compute_coverage=False,
        )
    )

    assert result["n_units"] == 1
    assert (results_dir / "genome_selection_manifest.json").is_file()
    assert (results_dir / "unit_genome_results.tsv").is_file()
    manifest = json.loads(
        (results_dir / "genome_selection_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["artifacts"]["run_summary"] == "artifacts/run_summary.json"
    assert "run_parameters" in manifest["artifacts"]
    assert "logs" in manifest["artifacts"]
    assert all((results_dir / path).exists() for path in manifest["artifacts"].values())
    mapping = pd.read_csv(results_dir / "sample_unit_mapping.tsv", sep="\t", dtype="string")
    assert mapping[["sample_id", "analysis_unit_id"]].values.tolist() == [
        ["__global__", "__global__"]
    ]


@pytest.mark.parametrize(
    ("payload", "expected_mode"),
    [
        ({"unit_specific": False}, "all-samples"),
        ({"unit_specific": True}, "per-sample"),
        (
            {"unit_specific": True, "metadata_table_path": "metadata.tsv"},
            "metadata",
        ),
        (
            {"unit_specific": True, "unit_mode": "all-samples"},
            "all-samples",
        ),
    ],
)
def test_legacy_scoring_config_migrates_unit_mode(payload, expected_mode):
    migrated = migrate_legacy_scoring_config_payload(payload)
    assert migrated["unit_mode"] == expected_mode
    assert "unit_specific" not in migrated


@pytest.mark.parametrize("suffix", [".tsv", ".TXT"])
def test_legacy_scoring_config_migrates_output_file_to_results_directory(tmp_path, suffix):
    legacy_output = tmp_path / f"genome_presence{suffix}"

    migrated = migrate_legacy_scoring_config_payload(
        {"unit_mode": "all-samples", "output_tsv_path": str(legacy_output)}
    )

    assert migrated["output_tsv_path"] == str(tmp_path / "genome_presence")
