import pandas as pd
import pytest

from metaumbra.analysis_units import (
    AnalysisUnitDefinition,
    GLOBAL_UNIT_ID,
    build_sample_unit_mapping,
)


def test_all_samples_produces_one_global_unit():
    mapping, metadata = build_sample_unit_mapping(
        ["s1", "s2"], AnalysisUnitDefinition(mode="all-samples")
    )
    assert metadata is None
    assert mapping["analysis_unit_id"].tolist() == [GLOBAL_UNIT_ID, GLOBAL_UNIT_ID]


def test_per_sample_produces_one_unit_per_sample():
    mapping, _ = build_sample_unit_mapping(
        ["s1", "s2"], AnalysisUnitDefinition(mode="per-sample")
    )
    assert mapping.to_dict("records") == [
        {"sample_id": "s1", "analysis_unit_id": "s1"},
        {"sample_id": "s2", "analysis_unit_id": "s2"},
    ]


def test_metadata_groups_samples_and_requires_complete_mapping(tmp_path):
    metadata_path = tmp_path / "metadata.tsv"
    pd.DataFrame({"sample": ["s1", "s2"], "unit": ["u1", "u1"]}).to_csv(
        metadata_path, sep="\t", index=False
    )
    definition = AnalysisUnitDefinition(
        mode="metadata", sample_id_column="Run", analysis_unit_column="unit"
    )
    mapping, _ = build_sample_unit_mapping(
        ["s1", "s2"], definition,
        metadata_table_path=metadata_path,
        metadata_sample_id_column="sample",
    )
    assert mapping["analysis_unit_id"].tolist() == ["u1", "u1"]
    with pytest.raises(ValueError, match="no analysis unit assignment"):
        build_sample_unit_mapping(
            ["s1", "s2", "s3"], definition,
            metadata_table_path=metadata_path,
            metadata_sample_id_column="sample",
        )


def test_metadata_rejects_duplicate_sample_assignments(tmp_path):
    path = tmp_path / "metadata.tsv"
    pd.DataFrame({"sample": ["s1", "s1"], "unit": ["u1", "u2"]}).to_csv(
        path, sep="\t", index=False
    )
    with pytest.raises(ValueError, match="duplicate sample IDs"):
        build_sample_unit_mapping(
            ["s1"],
            AnalysisUnitDefinition(mode="metadata", analysis_unit_column="unit"),
            metadata_table_path=path,
            metadata_sample_id_column="sample",
        )


def test_diann_parquet_builds_global_analysis_unit(tmp_path):
    pytest.importorskip("pyarrow")
    from metaumbra.scoring import GenomePresenceScorer

    path = tmp_path / "report.parquet"
    pd.DataFrame(
        {
            "Run": ["s1.raw", "s2.raw"],
            "Stripped.Sequence": ["PEPTIDEA", "PEPTIDEB"],
            "Precursor.Quantity": [100.0, 200.0],
            "Evidence": [1.0, 2.0],
            "Q.Value": [0.01, 0.01],
            "Decoy": [False, False],
        }
    ).to_parquet(path, index=False)
    scorer = GenomePresenceScorer(num_workers=1)
    scorer.read_analysis_unit_peptide_file(
        peptide_table_path=str(path), unit_mode="all-samples",
        sample_id_col="Run", peptide_seq_col="Sequence",
        peptide_decoy_flag_col="Decoy", decoy_flag_value="True",
    )
    assert scorer.unit_sample_ids == ["s1", "s2"]
    assert scorer.unit_analysis_unit_ids == [GLOBAL_UNIT_ID]
    assert scorer.unit_presence_matrix.shape == (2, 1)


def test_cli_exposes_unit_mode_and_rejects_removed_boolean():
    from metaumbra.cli import build_parser

    parser = build_parser()
    score = next(action for action in parser._actions if action.dest == "command")
    score_parser = score.choices["score"]
    destinations = {action.dest for action in score_parser._actions}
    assert "unit_mode" in destinations
    assert "unit_specific" not in destinations
