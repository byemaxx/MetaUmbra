import numpy as np
import pandas as pd
import pytest

from metaumbra._scoring.knockoff import (
    _compound_shared_sum_draws,
    _empirical_upper_tail,
    brown_scaled_chi_square_moment_fit,
    conditional_joint_null_fisher_mc,
)
from metaumbra._scoring.joint_null_reporting import (
    build_joint_null_shared_impact,
    build_joint_null_method_comparison,
    compare_calls_to_current,
    summarize_joint_null_calls,
)
from metaumbra.scoring import GenomePresenceScorer


def test_compound_shared_sum_draws_is_deterministic_and_reports_gamma_use():
    counts = np.asarray([0, 1, 2, 8, 9], dtype=int)
    pool = np.asarray([0.1, 0.2, 0.4], dtype=float)

    first, first_approximate = _compound_shared_sum_draws(
        counts,
        pool,
        np.random.default_rng(7),
        exact_count_cutoff=8,
        sample_block_size=2,
    )
    second, second_approximate = _compound_shared_sum_draws(
        counts,
        pool,
        np.random.default_rng(7),
        exact_count_cutoff=8,
        sample_block_size=2,
    )

    np.testing.assert_allclose(first, second)
    assert first_approximate == second_approximate == 1
    assert first[0] == 0.0
    assert bool(np.all(first[1:] >= 0.0))


def test_empirical_upper_tail_counts_ties_conservatively():
    calibration = np.asarray([1.0, 2.0, 2.0, 4.0])
    query = np.asarray([2.0, 3.0, 5.0])

    without_correction = _empirical_upper_tail(
        calibration,
        query,
        add_one=False,
    )
    with_correction = _empirical_upper_tail(
        calibration,
        query,
        add_one=True,
    )

    np.testing.assert_allclose(without_correction, [0.75, 0.25, 0.0])
    np.testing.assert_allclose(with_correction, [0.8, 0.4, 0.2])


def test_brown_scaled_chi_square_fit_matches_moment_formulas():
    fit = brown_scaled_chi_square_moment_fit(
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        np.asarray([0.0, 2.5, 10.0]),
    )

    assert fit["null_fisher_mean"] == pytest.approx(2.5)
    assert fit["null_fisher_variance"] == pytest.approx(5.0 / 3.0)
    assert fit["brown_scale"] == pytest.approx(1.0 / 3.0)
    assert fit["brown_df"] == pytest.approx(7.5)
    assert fit["brown_estimable"] is True
    pvalues = np.asarray(fit["pvalues"])
    assert pvalues.shape == (3,)
    assert pvalues[0] == pytest.approx(1.0)
    assert bool(np.all(np.diff(pvalues) <= 0.0))


def test_brown_scaled_chi_square_fit_handles_degenerate_null():
    fit = brown_scaled_chi_square_moment_fit(
        np.zeros(10),
        np.asarray([0.0, 1.0]),
    )

    assert fit["brown_estimable"] is False
    assert fit["brown_scale"] == 0.0
    assert fit["brown_df"] == 0.0
    np.testing.assert_array_equal(fit["pvalues"], np.ones(2))


def test_conditional_joint_null_is_deterministic_and_preserves_slot_dependence():
    kwargs = {
        "observed_unique_count": 8,
        "observed_shared_score": 3.0,
        "observed_matched_count": 20,
        "theoretical_total_peptides": 100,
        "theoretical_unique_peptides": 40,
        "iterations": 250,
        "validation_iterations": 200,
        "shared_contribution_pool": np.asarray([1.0]),
        "exact_shared_count_cutoff": 64,
        "sample_block_size": 32,
    }

    first = conditional_joint_null_fisher_mc(
        **kwargs,
        rng=np.random.default_rng(1),
    )
    second = conditional_joint_null_fisher_mc(
        **kwargs,
        rng=np.random.default_rng(1),
    )

    assert first == second
    assert first["null_model"] == "conditional-opportunity-compound-v0"
    assert first["minimum_attainable_p"] == pytest.approx(1.0 / 251.0)
    assert first["null_statistic_spearman"] == pytest.approx(-1.0)
    assert 0.0 < first["pvalue_joint_null_fisher"] <= 1.0
    assert first["gate_applied"] is False
    assert set(first["brown_prefix_results"]) == {"250"}
    brown = first["brown_prefix_results"]["250"]
    assert 0.0 < brown["pvalue_brown"] <= 1.0
    assert brown["calibration_iterations"] == 250
    assert brown["validation_iterations"] == 200


def test_conditional_joint_null_applies_zero_unique_gate():
    result = conditional_joint_null_fisher_mc(
        observed_unique_count=0,
        observed_shared_score=100.0,
        observed_matched_count=10,
        theoretical_total_peptides=50,
        theoretical_unique_peptides=20,
        iterations=100,
        validation_iterations=100,
        rng=np.random.default_rng(1),
        shared_contribution_pool=np.asarray([0.1, 0.2, 0.3]),
    )

    assert result["pvalue_joint_null_fisher"] == 1.0
    assert result["fisher_statistic_observed"] == 0.0
    assert result["gate_applied"] is True
    assert result["brown_prefix_results"]["100"]["pvalue_brown"] == 1.0


@pytest.mark.parametrize(
    "override",
    [
        {"observed_unique_count": 3, "observed_matched_count": 2},
        {"theoretical_total_peptides": 0},
        {"theoretical_total_peptides": 10, "theoretical_unique_peptides": 11},
        {"iterations": 0},
        {"observed_shared_score": -1.0},
    ],
)
def test_conditional_joint_null_validates_schema(override):
    kwargs = {
        "observed_unique_count": 1,
        "observed_shared_score": 1.0,
        "observed_matched_count": 2,
        "theoretical_total_peptides": 10,
        "theoretical_unique_peptides": 4,
        "iterations": 10,
        "validation_iterations": 10,
        "rng": np.random.default_rng(1),
        "shared_contribution_pool": np.asarray([0.1, 0.2]),
    }
    kwargs.update(override)

    with pytest.raises(ValueError):
        conditional_joint_null_fisher_mc(**kwargs)


def test_experimental_joint_null_integration_keeps_production_statistic(tmp_path):
    peptide_table = tmp_path / "peptides.tsv"
    pd.DataFrame({"Sequence": ["PEPA", "SHARED"]}).to_csv(
        peptide_table,
        sep="\t",
        index=False,
    )
    pd.DataFrame({"Peptide": ["PEPA", "SHARED", "OTHERAA"]}).to_csv(
        tmp_path / "g1.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame({"Peptide": ["PEPB", "SHARED", "OTHERBB"]}).to_csv(
        tmp_path / "g2.tsv",
        sep="\t",
        index=False,
    )

    scorer = GenomePresenceScorer(num_workers=1)
    scorer.knockoff_mc_iterations = 20
    scorer.knockoff_stage2_mc_iterations = None
    scorer.experimental_joint_null_enabled = True
    scorer.experimental_joint_null_iterations = 40
    scorer.experimental_joint_null_validation_iterations = 30
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
            ("g1", {"PEPA", "SHARED"}, 3),
            ("g2", {"SHARED"}, 3),
        ],
        compute_coverage=False,
        unique_pvalue_mode="alpha-upper-bound",
        presence_combination_method="bonferroni-min",
        return_full_table=True,
    )

    assert scorer.run_stats["experimental_joint_null_enabled"] is True
    assert scorer.run_stats["theoretical_opportunity_diagnostics_available"] is True
    assert result["presence_combination_method"].eq("bonferroni-min").all()
    assert result["pvalue"].tolist() == pytest.approx(
        result["pvalue_combined_bonferroni"].tolist()
    )
    assert {
        "pvalue_joint_null_unique_component",
        "pvalue_joint_null_shared_component",
        "pvalue_combined_joint_null_fisher",
        "qvalue_combined_joint_null_fisher",
        "joint_null_statistic_spearman",
        "joint_null_validation_ks_uniform_statistic",
        "pvalue_combined_joint_null_brown_b40",
        "joint_null_brown_b40_null_fisher_mean",
        "joint_null_brown_b40_brown_scale",
        "joint_null_brown_b40_brown_df",
    }.issubset(result.columns)
    by_genome = result.set_index("genome_id")
    assert by_genome.at["g2", "pvalue_combined_joint_null_fisher"] == 1.0
    assert bool(by_genome.at["g2", "joint_null_gate_applied"]) is True
    assert by_genome.at["g2", "pvalue_combined_joint_null_brown_b40"] == 1.0


def test_joint_null_reporting_builds_all_methods_and_truth_counts():
    candidates = pd.DataFrame(
        {
            "analysis_unit_id": ["u", "u"],
            "genome_id": ["expected", "extra"],
            "num_peptides_matched": [2, 1],
            "num_peptides_unique": [1, 0],
            "pvalue_combined_bonferroni": [0.001, 1.0],
            "pvalue_unique": [0.002, 1.0],
            "pvalue_combined_fisher": [0.0005, 0.0001],
            "pvalue_combined_joint_null_fisher": [0.001, 1.0],
        }
    )

    comparison = build_joint_null_method_comparison(
        candidates,
        family_id="benchmark",
        expected_truth_by_genome={"expected": True, "extra": False},
    )
    assert set(comparison["method"]) == {
        "current_bonferroni",
        "unique_only",
        "standard_fisher_independence_unvalidated",
        "zero_unique_gated_standard_fisher",
        "empirical_joint_null_fisher",
    }
    gated_extra = comparison[
        (comparison["method"] == "zero_unique_gated_standard_fisher")
        & (comparison["genome_id"] == "extra")
    ].iloc[0]
    assert gated_extra["pvalue_raw"] == 1.0

    summary = summarize_joint_null_calls(comparison)
    current_001 = summary[
        (summary["method"] == "current_bonferroni")
        & (summary["q_threshold"] == 0.01)
    ].iloc[0]
    assert current_001["call_count"] == 1
    assert current_001["candidate_denominator"] == 2
    assert current_001["expected_recovery_count"] == 1
    assert current_001["additional_call_count"] == 0

    changes = compare_calls_to_current(comparison)
    standard_gain = changes[
        (changes["method"] == "standard_fisher_independence_unvalidated")
        & (changes["genome_id"] == "extra")
        & (changes["q_threshold"] == 0.01)
    ]
    assert len(standard_gain) == 1
    assert standard_gain.iloc[0]["change_relative_to_current"] == "gained"


def test_joint_null_reporting_does_not_invent_missing_truth():
    candidates = pd.DataFrame(
        {
            "analysis_unit_id": ["u"],
            "genome_id": ["g"],
            "num_peptides_matched": [1],
            "num_peptides_unique": [1],
            "pvalue_combined_bonferroni": [0.001],
            "pvalue_unique": [0.001],
            "pvalue_combined_fisher": [0.001],
            "pvalue_combined_joint_null_fisher": [0.001],
        }
    )
    comparison = build_joint_null_method_comparison(candidates, family_id="hamster")
    summary = summarize_joint_null_calls(comparison)

    assert comparison["benchmark_truth_available"].eq(False).all()
    assert comparison["expected_genome"].isna().all()
    assert summary["expected_recovery_count"].isna().all()
    assert summary["additional_call_count"].isna().all()


def test_joint_null_shared_impact_uses_the_experimental_unique_component():
    candidates = pd.DataFrame(
        {
            "analysis_unit_id": ["u", "u"],
            "genome_id": ["a", "b"],
            "num_peptides_matched": [2, 2],
            "num_peptides_unique": [1, 1],
            "pvalue_joint_null_unique_component": [0.2, 0.01],
            "pvalue_combined_joint_null_fisher": [0.001, 0.02],
        }
    )

    impact = build_joint_null_shared_impact(
        candidates,
        family_id="benchmark",
        expected_truth_by_genome={"a": True, "b": False},
        material_rank_change=1,
    ).set_index("genome_id")

    assert bool(impact.at["a", "raw_p_improved_with_shared"]) is True
    assert bool(impact.at["a", "rank_improvement_with_shared"] >= 1) is True
    assert bool(impact.at["a", "material_rank_improvement_with_shared"]) is True
    assert bool(impact.at["a", "expected_genome"]) is True
    assert bool(impact.at["b", "raw_p_improved_with_shared"]) is False
