import copy

import pandas as pd
import pytest

from metaumbra._scoring.normalization import (
    normalize_peptide_collection,
    normalize_peptide_sequence,
)
from metaumbra._scoring.theoretical import _read_unique_peptides_from_digest
from metaumbra.scoring import (
    MATCHED_PEPTIDES_CACHE_VERSION,
    GenomePresenceScorer,
    _validate_matched_peptides_cache_provenance,
)


def test_il_equivalence_is_limited_to_isoleucine_and_leucine():
    assert normalize_peptide_sequence("AIL", "il-equivalent") == "AJJ"
    assert normalize_peptide_collection(["PEPTIDE", "PEPTLDE"], "il-equivalent") == {
        "PEPTJDE"
    }
    assert normalize_peptide_sequence("QKADW", "il-equivalent") == "QKADW"
    assert normalize_peptide_sequence("AIL", "exact") == "AIL"


def test_theoretical_digest_deduplicates_after_il_normalization(tmp_path):
    digest_path = tmp_path / "g1.tsv"
    pd.DataFrame({"Peptide": ["PEPTIDE", "PEPTLDE", "QKADW"]}).to_csv(
        digest_path, sep="\t", index=False
    )

    assert _read_unique_peptides_from_digest(digest_path, "il-equivalent") == {
        "PEPTJDE",
        "QKADW",
    }
    assert len(_read_unique_peptides_from_digest(digest_path, "exact")) == 3


def test_default_observed_reader_collapses_il_before_deduplication(tmp_path):
    peptide_path = tmp_path / "observed.tsv"
    pd.DataFrame(
        {
            "Sequence": ["PEPTIDE", "PEPTLDE"],
            "Q.Value": [0.01, 0.01],
        }
    ).to_csv(peptide_path, sep="\t", index=False)

    scorer = GenomePresenceScorer(num_workers=1)
    scorer.read_analysis_unit_peptide_file(
        peptide_table_path=str(peptide_path),
        unit_mode="all-samples",
        sample_id_col="Run",
        peptide_seq_col="Sequence",
        peptide_score_col=None,
        peptide_decoy_flag_col=None,
    )

    assert scorer.unit_peptides == ["PEPTJDE"]
    assert scorer.run_stats["observed_peptide_normalization_collisions"] == 1


def _expected_cache_provenance():
    return {
        "cache_schema_version": MATCHED_PEPTIDES_CACHE_VERSION,
        "software_version": "1.4.0",
        "peptide_normalization_policy": "il-equivalent",
        "observed_peptide_sha256": "observed",
        "reference_genome_list_sha256": "reference",
        "digest_bundle_manifest_sha256": "digest",
        "digestion_parameters": {
            "source": "precomputed_digest_tsv",
            "enzyme": "trypsin",
        },
    }


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("cache_schema_version", 1),
        ("peptide_normalization_policy", "exact"),
        ("observed_peptide_sha256", "different-observed"),
        ("reference_genome_list_sha256", "different-reference"),
        ("digest_bundle_manifest_sha256", "different-digest"),
        ("digestion_parameters", {"source": "other"}),
    ],
)
def test_matched_cache_rejects_every_stale_provenance_dimension(field, replacement):
    expected = _expected_cache_provenance()
    stale = {
        "provenance": copy.deepcopy(expected),
        "matched_peptides": [("g1", {"PEPTJDE"}, 1)],
    }
    stale["provenance"][field] = replacement

    with pytest.raises(ValueError, match=field):
        _validate_matched_peptides_cache_provenance(stale, expected)


def test_matched_cache_rejects_legacy_unversioned_payload():
    with pytest.raises(TypeError, match="Legacy"):
        _validate_matched_peptides_cache_provenance([], _expected_cache_provenance())


def test_matched_cache_accepts_exact_provenance_match():
    expected = _expected_cache_provenance()
    payload = [("g1", {"PEPTJDE"}, 1)]
    cached = {"provenance": copy.deepcopy(expected), "matched_peptides": payload}
    assert _validate_matched_peptides_cache_provenance(cached, expected) == payload
