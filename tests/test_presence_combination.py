import pytest

from metaumbra._scoring.ranking import (
    HMP_K2_EXACT_CALIBRATION,
    _normalize_presence_combination_method,
    bonferroni_min_p_2,
    calibrated_harmonic_mean_p_2,
    combine_presence_pvalues,
    harmonic_mean_p_2,
)


def test_harmonic_mean_matches_closed_form_examples():
    assert harmonic_mean_p_2(0.05, 0.8) == pytest.approx(2 * 0.05 * 0.8 / 0.85)
    assert harmonic_mean_p_2(0.06, 0.001) == pytest.approx(2 * 0.06 * 0.001 / 0.061)
    assert harmonic_mean_p_2(0.03, 0.03) == pytest.approx(0.03)


def test_harmonic_mean_unique_evidence_gate_blocks_shared_only_calls():
    assert harmonic_mean_p_2(
        1.0,
        0.001,
        require_unique_evidence=True,
        unique_count=0,
    ) == 1.0
    assert harmonic_mean_p_2(
        1.0,
        0.001,
        require_unique_evidence=False,
        unique_count=0,
    ) == pytest.approx(2.0 * 0.001 / 1.001)


def test_calibrated_harmonic_mean_uses_exact_k2_factor_and_requires_unique_evidence():
    raw = 2.0 * 0.06 * 0.001 / 0.061
    assert HMP_K2_EXACT_CALIBRATION == 2.0
    assert calibrated_harmonic_mean_p_2(
        0.06,
        0.001,
        num_peptides_unique=1,
    ) == pytest.approx(HMP_K2_EXACT_CALIBRATION * raw)
    assert calibrated_harmonic_mean_p_2(
        1.0,
        0.001,
        num_peptides_unique=0,
    ) == 1.0


def test_configured_combiner_supports_hmp_and_conservative_sensitivity():
    assert _normalize_presence_combination_method("hmp") == "harmonic-mean"
    assert _normalize_presence_combination_method("calibrated_hmp") == "harmonic-mean-calibrated"
    assert combine_presence_pvalues(
        0.001,
        1.0,
        method="harmonic-mean",
        unique_count=1,
    ) == pytest.approx(2.0 * 0.001 / 1.001)
    assert combine_presence_pvalues(
        0.03,
        0.03,
        method="bonferroni-min",
        unique_count=1,
    ) == pytest.approx(0.06)
    assert combine_presence_pvalues(
        0.06,
        0.001,
        method="harmonic-mean-calibrated",
        unique_count=1,
    ) == pytest.approx(HMP_K2_EXACT_CALIBRATION * (2.0 * 0.06 * 0.001 / 0.061))
    assert combine_presence_pvalues(
        1.0,
        0.001,
        method="bonferroni-min",
        unique_count=0,
    ) == 1.0
    assert bonferroni_min_p_2(0.8, 0.8) == 1.0


def test_unknown_combiner_is_rejected():
    with pytest.raises(ValueError, match="presence_combination_method"):
        _normalize_presence_combination_method("cauchy")
