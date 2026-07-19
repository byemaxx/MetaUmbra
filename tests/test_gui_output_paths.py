import csv

import pytest

from metaumbra.workflows import ScoringConfig

try:
    from metaumbra.gui import ScoringTab
except SystemExit:
    pytest.skip("A Qt binding is not installed", allow_module_level=True)


class _MappingHost:
    _sample_unit_mapping_source_path = "peptides.tsv"
    _sample_unit_mapping_rows = [
        {
            "sample_id": "s1",
            "analysis_unit_id": "u1",
            "included": True,
            "n_valid_peptides": 2,
            "n_total_rows": 3,
        }
    ]


def test_gui_mapping_is_materialized_inside_results_artifacts(tmp_path):
    results_dir = tmp_path / "results"
    config = ScoringConfig(
        peptide_table_path="peptides.tsv",
        output_tsv_path=str(results_dir),
    )

    ScoringTab._materialize_sample_unit_mapping(_MappingHost(), config)

    expected_path = results_dir / "artifacts" / "gui_sample_unit_mapping.tsv"
    assert config.metadata_table_path == str(expected_path)
    assert expected_path.is_file()
    with expected_path.open("r", encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle, delimiter="\t")) == [
            {
                "sample_id": "s1",
                "analysis_unit_id": "u1",
                "included": "true",
                "n_valid_peptides": "2",
                "n_total_rows": "3",
            }
        ]
    assert not (tmp_path / "results_gui_sample_unit_mapping.tsv").exists()
