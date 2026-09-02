import pytest
import numpy as np

from metaumbra._scoring.ranking import (
    HMP_K2_EXACT_CALIBRATION,
    _normalize_presence_combination_method,
    bonferroni_min_p_2,
    calibrated_harmonic_mean_p_2,
    combine_presence_pvalues,
    harmonic_mean_p_2,
    simes_closed_p_2,
    simes_intersection_p_2,
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
    assert _normalize_presence_combination_method("simes_closed") == "simes-closed"
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


def test_simes_intersection_formula():
    assert simes_intersection_p_2(0.04, 0.04) == pytest.approx(0.04)
    assert simes_intersection_p_2(0.001, 0.8) == pytest.approx(0.002)
    assert simes_intersection_p_2(0.04, 0.02) == pytest.approx(0.04)


def test_simes_closed_formula_and_zero_unique_gate():
    assert simes_closed_p_2(0.04, 0.04, num_peptides_unique=1) == pytest.approx(0.04)
    assert simes_closed_p_2(0.001, 0.8, num_peptides_unique=1) == pytest.approx(0.002)
    assert simes_closed_p_2(0.04, 0.02, num_peptides_unique=1) == pytest.approx(0.04)
    assert simes_closed_p_2(0.3, 1e-6, num_peptides_unique=1) == pytest.approx(0.3)
    assert simes_closed_p_2(1.0, 1e-10, num_peptides_unique=0) == 1.0


@pytest.mark.parametrize(
    ("p_unique", "p_shared", "expected"),
    [
        (1e-300, 1.0, 2e-300),
        (0.999999, 0.999998, 0.999999),
        (0.2, 0.8, 0.4),
        (0.8, 0.2, 0.8),
        (0.3, 0.3, 0.3),
    ],
)
def test_simes_closed_edge_cases(p_unique, p_shared, expected):
    assert simes_closed_p_2(p_unique, p_shared, num_peptides_unique=1) == pytest.approx(expected)
    assert simes_closed_p_2(p_unique, 1e-300, num_peptides_unique=0) == 1.0


def test_simes_closed_properties():
    rng = np.random.default_rng(1)
    p_unique = rng.random(10_000)
    p_shared = rng.random(10_000)
    combined = np.asarray(
        [simes_closed_p_2(u, s, num_peptides_unique=1) for u, s in zip(p_unique, p_shared)]
    )
    assert np.all(combined >= p_unique - 1e-15)
    assert np.all(combined <= np.minimum(1.0, 2.0 * p_unique) + 1e-15)
    stronger_shared = np.maximum(0.0, p_shared - 0.1)
    combined_stronger = np.asarray(
        [simes_closed_p_2(u, s, num_peptides_unique=1) for u, s in zip(p_unique, stronger_shared)]
    )
    assert np.all(combined_stronger <= combined + 1e-15)
    assert np.allclose(
        [simes_closed_p_2(u, u / 2.0, num_peptides_unique=1) for u in p_unique],
        p_unique,
    )
    eligible = p_unique <= 0.5
    assert np.allclose(
        [simes_closed_p_2(u, min(1.0, 2.0 * u), num_peptides_unique=1) for u in p_unique[eligible]],
        2.0 * p_unique[eligible],
    )


def test_unknown_combiner_is_rejected():
    with pytest.raises(ValueError, match="presence_combination_method"):
        _normalize_presence_combination_method("cauchy")
