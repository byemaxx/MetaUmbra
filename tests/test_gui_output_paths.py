import csv

import pytest

from metaumbra.workflows import ScoringConfig

try:
    from metaumbra.gui import QApplication, ScoringTab
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


def test_build_config_materializes_manual_mapping_before_metadata_validation(tmp_path):
    app = QApplication.instance() or QApplication([])
    peptide_table = tmp_path / "peptides.tsv"
    peptide_table.write_text("Sequence\tEvidence\tRun\tPrecursor.Quantity\n", encoding="utf-8")
    digest_dir = tmp_path / "digest"
    digest_dir.mkdir()
    results_dir = tmp_path / "results"

    tab = ScoringTab()
    tab.peptide_table_edit.setText(str(peptide_table))
    tab.output_tsv_edit.setText(str(results_dir))
    tab.genome_dir_list.addItem(str(digest_dir))
    tab.unit_mode_combo.setCurrentIndex(2)
    tab.metadata_table_edit.clear()
    tab._sample_unit_mapping_source_path = str(peptide_table)
    tab._sample_unit_mapping_rows = [
        {
            "sample_id": "s1",
            "analysis_unit_id": "u1",
            "included": True,
            "n_valid_peptides": 2,
            "n_total_rows": 3,
        }
    ]

    config = tab.build_config(require_required_fields=True)

    expected_path = results_dir / "artifacts" / "gui_sample_unit_mapping.tsv"
    assert config.metadata_table_path == str(expected_path)
    assert expected_path.is_file()
    app.processEvents()
