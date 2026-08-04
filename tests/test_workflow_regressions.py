import json

import pandas as pd
import pytest

from metaumbra.cli import (
    DEFAULT_PARQUET_INPUT_COLUMNS,
    DEFAULT_PARQUET_OUTPUT_COLUMNS,
    build_parser,
)
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


def test_production_empirical_method_defaults_to_alpha_excess():
    assert ScoringConfig().unique_empirical_pvalue_method == "alpha-excess"
    assert ScoringConfig().presence_combination_method == "bonferroni-min"
    assert ScoringConfig().hmp_require_unique_evidence is True
    parser = build_parser()
    command = next(action for action in parser._actions if action.dest == "command")
    score_parser = command.choices["score"]
    method = next(
        action
        for action in score_parser._actions
        if action.dest == "unique_empirical_pvalue_method"
    )
    assert method.default == "alpha-excess"
    combination = next(
        action
        for action in score_parser._actions
        if action.dest == "presence_combination_method"
    )
    assert combination.default == "bonferroni-min"


def test_legacy_scoring_configuration_preserves_fisher_combination():
    migrated = migrate_legacy_scoring_config_payload({"unit_specific": True})
    assert migrated["presence_combination_method"] == "fisher"
    explicit = migrate_legacy_scoring_config_payload(
        {"presence_combination_method": "harmonic-mean-calibrated"}
    )
    assert explicit["presence_combination_method"] == "harmonic-mean-calibrated"


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


def _write_digest(digest_dir, genome_id, peptides):
    pd.DataFrame({"Peptide": peptides}).to_csv(
        digest_dir / f"{genome_id}.tsv", sep="\t", index=False
    )


def _run_auto_mode(tmp_path, digest_peptides, observed_peptides):
    peptide_path = tmp_path / "peptides.tsv"
    pd.DataFrame(
        {
            "Sequence": observed_peptides,
            "Evidence": [1.0] * len(observed_peptides),
            "Q.Value": [0.01] * len(observed_peptides),
        }
    ).to_csv(peptide_path, sep="\t", index=False)
    digest_dir = tmp_path / "digests"
    digest_dir.mkdir()
    for genome_id, peptides in digest_peptides.items():
        _write_digest(digest_dir, genome_id, peptides)
    results_dir = tmp_path / "results"
    run_scoring_workflow(
        ScoringConfig(
            peptide_table_path=str(peptide_path),
            genome_digest_dirs=[str(digest_dir)],
            output_tsv_path=str(results_dir),
            unique_pvalue_mode="auto",
            presence_combination_method="fisher",
            num_workers=1,
            knockoff_mc_iterations=20,
            knockoff_stage2_mc_iterations=None,
            compute_coverage=False,
            export_diagnostics=True,
        )
    )
    return results_dir


def test_scoring_workflow_records_custom_degeneracy_bin_edges(tmp_path):
    peptide_path = tmp_path / "peptides.tsv"
    pd.DataFrame(
        {
            "Sequence": ["SHARED", "UNIQUEA"],
            "Evidence": [1.0, 1.0],
            "Q.Value": [0.01, 0.01],
        }
    ).to_csv(peptide_path, sep="\t", index=False)
    digest_dir = tmp_path / "digests"
    digest_dir.mkdir()
    _write_digest(digest_dir, "g1", ["SHARED", "UNIQUEA"])
    _write_digest(digest_dir, "g2", ["SHARED"])
    results_dir = tmp_path / "results"

    run_scoring_workflow(
        ScoringConfig(
            peptide_table_path=str(peptide_path),
            genome_digest_dirs=[str(digest_dir)],
            output_tsv_path=str(results_dir),
            degeneracy_bin_edges=[1, 3, 10],
            num_workers=1,
            knockoff_mc_iterations=20,
            knockoff_stage2_mc_iterations=None,
            compute_coverage=False,
        )
    )

    parameters = json.loads(
        (results_dir / "artifacts" / "run_parameters.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (results_dir / "artifacts" / "run_summary.json").read_text(encoding="utf-8")
    )
    assert parameters["config"]["degeneracy_bin_edges"] == [1, 3, 10]
    assert parameters["config"]["unique_empirical_pvalue_method"] == "alpha-excess"
    assert summary["degeneracy_bin_edges"] == [1, 3, 10]


@pytest.mark.parametrize("edges", [[], [0, 5], [1, 1, 5], [5, 1]])
def test_scoring_workflow_rejects_invalid_degeneracy_bin_edges(tmp_path, edges):
    with pytest.raises(ValueError, match="strictly increasing"):
        run_scoring_workflow(
            ScoringConfig(
                peptide_table_path=str(tmp_path / "peptides.tsv"),
                genome_digest_dirs=[str(tmp_path / "digests")],
                output_tsv_path=str(tmp_path / "results"),
                degeneracy_bin_edges=edges,
            )
        )


def test_auto_mode_falls_back_to_alpha_upper_bound_for_structurally_inadequate_background(tmp_path):
    digest_peptides = {
        f"g{index}": [f"G{index}A", f"G{index}B", f"G{index}C"]
        for index in range(5)
    }
    observed_peptides = [peptide for peptides in digest_peptides.values() for peptide in peptides]
    results_dir = _run_auto_mode(tmp_path, digest_peptides, observed_peptides)

    result = pd.read_csv(
        results_dir / "artifacts" / "diagnostics" / "full_internal_metrics.tsv",
        sep="\t",
    )
    calibration = pd.read_csv(
        results_dir / "artifacts" / "diagnostics" / "unit_empirical_background_calibration.tsv",
        sep="\t",
    )
    assert set(result["unique_pvalue_mode_requested"]) == {"auto"}
    assert set(result["unique_pvalue_mode_resolved"]) == {"alpha-upper-bound"}
    assert not bool(calibration.loc[0, "unit_auto_eligibility_decision"])
    assert calibration.loc[0, "unit_auto_candidate_count"] == 5
    assert calibration.loc[0, "unit_auto_max_comparable_observed"] == 4
    assert "comparable backgrounds" in calibration.loc[0, "unit_auto_eligibility_reason"]


def test_auto_mode_does_not_use_observed_sparsity_to_override_small_panel_ineligibility(tmp_path):
    digest_peptides = {"target": ["COMMON", "T1", "T2", "T3"]}
    digest_peptides.update({f"background_{index}": ["COMMON"] for index in range(6)})
    results_dir = _run_auto_mode(
        tmp_path,
        digest_peptides,
        ["COMMON", "T1", "T2", "T3"],
    )

    result = pd.read_csv(
        results_dir / "artifacts" / "diagnostics" / "full_internal_metrics.tsv",
        sep="\t",
    )
    calibration = pd.read_csv(
        results_dir / "artifacts" / "diagnostics" / "unit_empirical_background_calibration.tsv",
        sep="\t",
    )
    assert set(result["unique_pvalue_mode_requested"]) == {"auto"}
    assert set(result["unique_pvalue_mode_resolved"]) == {"alpha-upper-bound"}
    assert not bool(calibration.loc[0, "unit_auto_eligibility_decision"])
    assert calibration.loc[0, "unit_auto_candidate_count"] == 7


def test_auto_mode_keeps_structurally_eligible_empirical_mode_when_diagnostics_flag_cap_pressure(tmp_path):
    digest_peptides = {"target": ["COMMON", "T1", "T2", "T3"]}
    digest_peptides.update({f"background_{index:03d}": ["COMMON"] for index in range(100)})
    results_dir = _run_auto_mode(
        tmp_path,
        digest_peptides,
        ["COMMON", "T1", "T2", "T3"],
    )
    result = pd.read_csv(
        results_dir / "artifacts" / "diagnostics" / "full_internal_metrics.tsv",
        sep="\t",
    )
    calibration = pd.read_csv(
        results_dir / "artifacts" / "diagnostics" / "unit_empirical_background_calibration.tsv",
        sep="\t",
    )
    assert set(result["unique_pvalue_mode_resolved"]) == {"empirical-background"}
    assert bool(calibration.loc[0, "unit_auto_eligibility_decision"])
    assert calibration.loc[0, "unit_auto_min_comparable_observed"] == 100
    assert bool(calibration.loc[0, "unit_auto_empirical_cap_reached"])
    assert not bool(calibration.loc[0, "unit_auto_empirical_suitability"])


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
