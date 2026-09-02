import json

import pandas as pd
import pytest

from metaumbra.scoring import GenomePresenceScorer
from metaumbra._scoring.unit_specific import (
    _empirical_background_calibration_for_unit_mode,
)
from metaumbra._scoring.ranking import bh_qvalues


def test_all_samples_uses_moderately_permissive_empirical_background_profile():
    pooled = _empirical_background_calibration_for_unit_mode("all-samples")
    grouped = _empirical_background_calibration_for_unit_mode("metadata")

    assert pooled == {
        "profile": "all-samples-moderately-permissive",
        "initial_exclude_fraction": 0.10,
        "min_exclude_fraction": 0.05,
        "max_exclude_fraction": 0.20,
        "candidate_q": 0.20,
        "max_iterations": 5,
    }
    assert grouped == {
        "profile": "grouped-unit-conservative",
        "initial_exclude_fraction": 0.05,
        "min_exclude_fraction": 0.02,
        "max_exclude_fraction": 0.20,
        "candidate_q": 0.20,
        "max_iterations": 3,
    }


def test_analysis_unit_worker_is_the_only_qvalue_engine(tmp_path):
    assert not hasattr(GenomePresenceScorer, "_rank_genomes")
    assert not hasattr(GenomePresenceScorer, "_add_knockoff_existence_stats")

    peptide_table = tmp_path / "peptides.tsv"
    pd.DataFrame(
        {
            "Run": ["s1", "s2"],
            "Sequence": ["PEPA", "PEPB"],
            "Intensity": [100.0, 200.0],
            "Evidence": [1.0, 1.0],
            "Q.Value": [0.01, 0.01],
        }
    ).to_csv(peptide_table, sep="\t", index=False)
    pd.DataFrame({"Sequence": ["PEPA"]}).to_csv(tmp_path / "g1.tsv", sep="\t", index=False)
    pd.DataFrame({"Sequence": ["PEPB"]}).to_csv(tmp_path / "g2.tsv", sep="\t", index=False)

    scorer = GenomePresenceScorer(num_workers=1)
    scorer.knockoff_mc_iterations = 50
    scorer.knockoff_stage2_mc_iterations = None
    scorer.read_analysis_unit_peptide_file(
        peptide_table_path=str(peptide_table),
        unit_mode="per-sample",
        sample_id_col="Run",
        peptide_seq_col="Sequence",
        peptide_score_col="Evidence",
        peptide_error_col="Q.Value",
        intensity_col="Intensity",
    )

    output = tmp_path / "unit_genome_results.tsv"
    result = scorer.analyze_genomes(
        genome_digest_dirs=[str(tmp_path)],
        output_tsv_path=str(output),
        all_matched_peptides=[
            ("g1", {"PEPA"}, 1),
            ("g2", {"PEPB"}, 1),
        ],
        compute_coverage=False,
        export_diagnostics=True,
        export_peptide_contrib_topN=3,
    )

    assert set(result["analysis_unit_id"]) == {"s1", "s2"}
    assert len(result) == 4
    assert result.groupby("analysis_unit_id")["genome_id"].apply(list).tolist() == [
        ["g1", "g2"],
        ["g1", "g2"],
    ]
    peptide_contrib = pd.read_csv(
        tmp_path / "artifacts" / "diagnostics" / "top3_peptide_contrib.tsv",
        sep="\t",
    )
    assert peptide_contrib[["analysis_unit_id", "genome_id", "peptide"]].values.tolist() == [
        ["s1", "g1", "PEPA"],
        ["s2", "g2", "PEPB"],
    ]
    manifest = json.loads(
        (tmp_path / "genome_selection_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["artifacts"]["run_summary"] == "artifacts/run_summary.json"
    assert "run_parameters" not in manifest["artifacts"]
    assert "logs" not in manifest["artifacts"]
    assert all((tmp_path / path).exists() for path in manifest["artifacts"].values())
    summary = json.loads((tmp_path / "artifacts" / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["scoring_engine"] == "per-analysis-unit"
    assert summary["pooled_scoring_performed"] is False
    assert summary["genomes_q_fields_scope"].startswith("per-analysis-unit scoring")
    assert summary["empirical_background_calibration_profile"] == "grouped-unit-conservative"
    assert "pooled_genomes_q_le_0p05" not in summary


def test_public_peptide_reader_adapts_to_all_samples_scoring(tmp_path):
    peptide_table = tmp_path / "legacy_peptides.tsv"
    pd.DataFrame({"Sequence": ["PEPA", "PEPB"]}).to_csv(
        peptide_table,
        sep="\t",
        index=False,
    )
    pd.DataFrame({"Sequence": ["PEPA"]}).to_csv(tmp_path / "g1.tsv", sep="\t", index=False)
    pd.DataFrame({"Sequence": ["PEPB"]}).to_csv(tmp_path / "g2.tsv", sep="\t", index=False)

    scorer = GenomePresenceScorer(num_workers=1)
    scorer.knockoff_mc_iterations = 50
    scorer.knockoff_stage2_mc_iterations = None
    scorer.read_peptide_file(
        peptide_table_path=str(peptide_table),
        peptide_seq_col="Sequence",
        peptide_score_col=None,
        peptide_decoy_flag_col=None,
    )

    result = scorer.analyze_genomes(
        genome_digest_dirs=[str(tmp_path)],
        output_tsv_path=str(tmp_path / "unit_genome_results.tsv"),
        all_matched_peptides=[
            ("g1", {"PEPA"}, 1),
            ("g2", {"PEPB"}, 1),
        ],
        compute_coverage=False,
    )

    assert set(result["analysis_unit_id"]) == {"__global__"}
    assert scorer.unit_sample_ids == ["__global__"]
    assert scorer.sample_unit_mapping_df[["sample_id", "analysis_unit_id"]].values.tolist() == [
        ["__global__", "__global__"]
    ]
    assert (
        scorer.run_stats["empirical_background_calibration_profile"]
        == "all-samples-moderately-permissive"
    )


def test_harmonic_mean_selection_exports_all_combined_pvalues(tmp_path):
    peptide_table = tmp_path / "peptides.tsv"
    pd.DataFrame({"Sequence": ["UNIQUEA", "UNIQUEB"]}).to_csv(
        peptide_table, sep="\t", index=False
    )
    pd.DataFrame({"Sequence": ["UNIQUEA"]}).to_csv(tmp_path / "g1.tsv", sep="\t", index=False)
    pd.DataFrame({"Sequence": ["UNIQUEB"]}).to_csv(tmp_path / "g2.tsv", sep="\t", index=False)

    scorer = GenomePresenceScorer(num_workers=1)
    scorer.knockoff_mc_iterations = 50
    scorer.knockoff_stage2_mc_iterations = None
    scorer.read_peptide_file(
        peptide_table_path=str(peptide_table),
        peptide_seq_col="Sequence",
        peptide_score_col=None,
        peptide_decoy_flag_col=None,
    )
    result = scorer.analyze_genomes(
        genome_digest_dirs=[str(tmp_path)],
        output_tsv_path=str(tmp_path / "unit_genome_results.tsv"),
        all_matched_peptides=[
            ("g1", {"UNIQUEA"}, 1),
            ("g2", {"UNIQUEB"}, 1),
        ],
        compute_coverage=False,
        presence_combination_method="harmonic-mean",
        hmp_require_unique_evidence=True,
        return_full_table=True,
    )

    assert {
        "pvalue_combined_fisher",
        "pvalue_simes_intersection",
        "pvalue_simes_closed",
        "pvalue_combined_harmonic_calibrated",
        "pvalue_combined_harmonic",
        "pvalue_combined_bonferroni",
    }.issubset(result.columns)
    assert result["presence_combination_method"].eq("harmonic-mean").all()
    assert result["hmp_require_unique_evidence"].eq(True).all()
    assert result["pvalue"].tolist() == pytest.approx(
        result["pvalue_combined_harmonic"].tolist()
    )


def test_default_simes_bh_uses_all_and_only_matched_genomes(tmp_path, monkeypatch):
    peptide_table = tmp_path / "peptides.tsv"
    pd.DataFrame({"Sequence": ["G1ONLY", "G2ONLY", "SHARED"]}).to_csv(
        peptide_table, sep="\t", index=False
    )
    scorer = GenomePresenceScorer(num_workers=1)
    scorer.knockoff_mc_iterations = 10
    scorer.knockoff_stage2_mc_iterations = None
    scorer.read_peptide_file(
        peptide_table_path=str(peptide_table),
        peptide_seq_col="Sequence",
        peptide_score_col=None,
        peptide_decoy_flag_col=None,
    )

    monkeypatch.setattr(
        "metaumbra._scoring.unit_specific.shared_knockoff_mc",
        lambda **kwargs: (1.0, 0.0, 0.0, 0.0, 0.0),
    )
    result = scorer.analyze_genomes(
        genome_digest_dirs=[str(tmp_path)],
        output_tsv_path=str(tmp_path / "unit_genome_results.tsv"),
        all_matched_peptides=[
            ("g1", {"G1ONLY", "SHARED"}, 2),
            ("g2", {"G2ONLY", "SHARED"}, 2),
            ("g3", {"SHARED"}, 1),
            ("g4", {"NO_MATCH"}, 1),
        ],
        compute_coverage=False,
        unique_pvalue_mode="alpha-upper-bound",
        return_full_table=True,
    ).set_index("genome_id")

    active = result.loc[["g1", "g2", "g3"]]
    assert active["num_peptides_matched"].ge(1).all()
    assert result.at["g4", "num_peptides_matched"] == 0
    assert active["pvalue"].tolist() == pytest.approx(
        active["pvalue_simes_closed"].tolist()
    )
    assert result.at["g3", "num_peptides_unique"] == 0
    assert result.at["g3", "pvalue_simes_closed"] == 1.0
    assert result.at["g3", "pvalue"] == 1.0
    assert active["qvalue"].tolist() == pytest.approx(bh_qvalues(active["pvalue"].to_numpy()))
    assert result.at["g1", "qvalue"] > result.at["g1", "pvalue"]
    assert result.at["g4", "qvalue"] == 1.0
    assert result.at["g4", "presence_score"] == 0.0
    assert not bool(result.at["g4", "pass_q_0_05"])


def test_knockoff_top_n_targets_limits_unit_worker_inference(tmp_path, monkeypatch):
    peptide_table = tmp_path / "peptides.tsv"
    pd.DataFrame({"Sequence": ["UNIQUEA", "UNIQUEB", "UNIQUEC", "SHARED"]}).to_csv(
        peptide_table,
        sep="\t",
        index=False,
    )
    pd.DataFrame({"Sequence": ["UNIQUEA", "UNIQUEB", "SHARED"]}).to_csv(
        tmp_path / "g1.tsv", sep="\t", index=False
    )
    pd.DataFrame({"Sequence": ["UNIQUEC", "SHARED"]}).to_csv(
        tmp_path / "g2.tsv", sep="\t", index=False
    )

    called_genomes = []

    def fake_shared_knockoff_mc(**kwargs):
        called_genomes.append(str(kwargs["gid"]))
        return 0.5, 0.0, 0.0, 0.0, 0.0

    monkeypatch.setattr(
        "metaumbra._scoring.unit_specific.shared_knockoff_mc",
        fake_shared_knockoff_mc,
    )
    scorer = GenomePresenceScorer(num_workers=1)
    scorer.knockoff_mc_iterations = 50
    scorer.knockoff_stage2_mc_iterations = None
    scorer.knockoff_top_n_targets = 1
    scorer.read_peptide_file(
        peptide_table_path=str(peptide_table),
        peptide_seq_col="Sequence",
        peptide_score_col=None,
        peptide_decoy_flag_col=None,
    )

    result = scorer.analyze_genomes(
        genome_digest_dirs=[str(tmp_path)],
        output_tsv_path=str(tmp_path / "unit_genome_results.tsv"),
        all_matched_peptides=[
            ("g1", {"UNIQUEA", "UNIQUEB", "SHARED"}, 3),
            ("g2", {"UNIQUEC", "SHARED"}, 2),
        ],
        compute_coverage=False,
        return_full_table=True,
    )

    assert called_genomes == ["g1"]
    assert scorer.run_stats["unit_specific_total_knockoff_target_genomes"] == 1
    by_genome = result.set_index("genome_id")
    assert bool(by_genome.at["g1", "knockoff_target"]) is True
    assert bool(by_genome.at["g2", "knockoff_target"]) is False
    assert by_genome.at["g2", "pvalue_shared"] == pytest.approx(1.0)
    assert by_genome.at["g2", "pvalue_unique"] == pytest.approx(1.0)
    assert by_genome.at["g2", "pvalue"] == pytest.approx(1.0)
    assert by_genome.at["g2", "qvalue"] == pytest.approx(1.0)
