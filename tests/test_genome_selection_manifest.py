from pathlib import Path

import pandas as pd
import pytest

from metaumbra.genome_selection_manifest import (
    SCHEMA_VERSION,
    build_genome_selection_manifest,
    validate_genome_selection_manifest,
)


def _manifest(tmp_path, mode="all-samples"):
    mapping = pd.DataFrame(
        {"sample_id": ["s1", "s2"], "analysis_unit_id": ["__global__", "__global__"]}
    )
    results = pd.DataFrame(
        {
            "analysis_unit_id": ["__global__", "__global__"],
            "genome_id": ["g1", "g2"],
            "pass_q_0_01": [True, False],
            "pass_q_0_05": [True, True],
        }
    )
    peptide = tmp_path / "report.parquet"
    peptide.touch()
    return build_genome_selection_manifest(
        mapping_df=mapping,
        unit_genome_results=results,
        unit_mode=mode,
        sample_id_column="Run",
        analysis_unit_column=None,
        peptide_table_path=str(peptide),
        metadata_table_path=None,
        genome_digest_directories=[str(tmp_path / "digests")],
        artifacts={"unit_genome_results": "unit_genome_results.tsv"},
        scoring_method="per-analysis-unit/empirical-background",
        run_id="test",
    )


def test_manifest_schema_and_threshold_lists(tmp_path):
    manifest = _manifest(tmp_path)
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["unit_definition"]["n_units"] == 1
    assert manifest["units"]["__global__"]["sample_ids"] == ["s1", "s2"]
    assert manifest["units"]["__global__"]["genome_ids_q005"] == ["g1", "g2"]
    assert manifest["units"]["__global__"]["genome_ids_q001"] == ["g1"]


def test_manifest_rejects_duplicate_cross_unit_sample(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["units"]["u2"] = {
        "sample_ids": ["s1"], "n_samples": 1,
        "genome_ids_q005": [], "genome_ids_q001": [],
    }
    manifest["unit_definition"]["n_units"] = 2
    with pytest.raises(ValueError, match="multiple units"):
        validate_genome_selection_manifest(manifest)

