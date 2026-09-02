from pathlib import Path

import pandas as pd
import pytest

from metaumbra._scoring.empirical import _compute_empirical_background_stats_for_table
from metaumbra.scoring import GenomePresenceScorer


def _write_digest(path: Path, peptides) -> None:
    pd.DataFrame({"Sequence": list(peptides)}).to_csv(path, sep="\t", index=False)


def _empirical_table(unique_counts, opportunities=None) -> pd.DataFrame:
    counts = list(unique_counts)
    if opportunities is None:
        opportunities = [100] * len(counts)
    return pd.DataFrame(
        {
            "genome_id": [f"g{index}" for index in range(len(counts))],
            "num_peptides_unique": counts,
            "theoretical_panel_unique_peptide_opportunity": list(opportunities),
            "_genomes_with_any_match": [True] * len(counts),
        }
    )


def _empirical_stats(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    out, _ = _compute_empirical_background_stats_for_table(
        df,
        alpha=0.05,
        top_exclude_fraction=0.0,
        threshold_quantile=0.95,
        n_bins=kwargs.pop("n_bins", 1),
        min_bin_size=kwargs.pop("min_bin_size", 1),
        **kwargs,
    )
    return out


def test_theoretical_panel_unique_opportunity_respects_normalization(tmp_path):
    paths = []
    for genome_id, peptides in {
        "g1": ["PEPTIDE", "PEPTLDE", "SHARED"],
        "g2": ["OTHER", "SHARED"],
    }.items():
        path = tmp_path / f"{genome_id}.tsv"
        _write_digest(path, peptides)
        paths.append(path)

    scorer = GenomePresenceScorer(num_workers=1)
    scorer.peptide_normalization_policy = "exact"
    exact = scorer._build_theoretical_opportunity_serial(paths)
    scorer.peptide_normalization_policy = "il-equivalent"
    il_equivalent = scorer._build_theoretical_opportunity_serial(paths)

    assert exact["genome_theoretical_unique_peptides"] == {"g1": 2, "g2": 1}
    assert il_equivalent["genome_theoretical_unique_peptides"] == {"g1": 1, "g2": 1}


def test_empirical_background_requires_panel_unique_opportunity():
    with pytest.raises(ValueError, match="no total-peptide fallback"):
        _empirical_stats(
            pd.DataFrame(
                {
                    "genome_id": ["g1", "g2"],
                    "num_peptides_unique": [1, 0],
                    "total_peptide_count": [100, 100],
                    "_genomes_with_any_match": [True, True],
                }
            )
        )


def test_alpha_excess_is_default_and_empirical_tail_is_explicit():
    default = _empirical_stats(_empirical_table([0, 1, 2, 3]))
    tail = _empirical_stats(
        _empirical_table([0, 1, 2, 3]),
        pvalue_method="empirical-tail",
    )

    assert default["p_unique_empirical_formal"].tolist() == pytest.approx(
        default["unique_alpha_excess_index"].tolist()
    )
    assert tail["p_unique_empirical_formal"].tolist() == pytest.approx(
        [1.0, 0.75, 0.5, 0.25]
    )


def test_empirical_background_excludes_the_scored_candidate_from_its_null():
    out = _empirical_stats(_empirical_table([0, 10]))
    assert out.loc[1, "unique_empirical_background_size"] == 1
    assert out.loc[1, "p_unique_empirical_tail"] == pytest.approx(0.5)


def test_theoretical_opportunity_cache_rejects_stale_reference_provenance(tmp_path):
    paths = [tmp_path / "g1.tsv", tmp_path / "g2.tsv"]
    _write_digest(paths[0], ["A"])
    _write_digest(paths[1], ["B"])
    scorer = GenomePresenceScorer(num_workers=1)
    scorer.peptide_normalization_policy = "exact"
    cache = scorer._build_theoretical_opportunity_serial(paths)
    cache["reference_genome_list_sha256"] = "stale"

    assert not scorer._theoretical_cache_matches_digest_files(cache, paths)
