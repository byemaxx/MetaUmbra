import json

import pandas as pd
import pytest

from metaumbra.cli import DEFAULT_PARQUET_INPUT_COLUMNS, DEFAULT_PARQUET_OUTPUT_COLUMNS
from metaumbra.scoring import GenomePresenceScorer
from metaumbra.workflows import (
    ParquetExtractionConfig,
    ScoringConfig,
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
    assert scorer.unit_sample_ids == ["s1.raw"]


def test_failed_directory_run_writes_status_to_results_artifacts(tmp_path):
    results_dir = tmp_path / "results"
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
    assert not (tmp_path / "artifacts").exists()


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
