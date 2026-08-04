import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "results" / "revision_2026-08" / "analyses" / "hmp_fisher_validation" / "run_hmp_fisher_validation.py"
SPEC = importlib.util.spec_from_file_location("hmp_fisher_validation", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def test_recompute_preserves_source_table_bh_families_and_hmp_gate():
    frame = pd.DataFrame({
        "source_table_sha256": ["a", "a", "b", "b"],
        "analysis_unit_id": ["unit"] * 4,
        "genome_id": ["g1", "g2", "g3", "g4"],
        "num_peptides_matched": [1] * 4,
        "num_peptides_unique": [1, 0, 1, 1],
        "pvalue_unique": [0.01, 1.0, 0.01, 1.0],
        "pvalue_shared": [1.0, 0.001, 1.0, 1.0],
        "pvalue": [AUDIT.fisher_p_2(0.01, 1.0), AUDIT.fisher_p_2(1.0, 0.001), AUDIT.fisher_p_2(0.01, 1.0), 1.0],
        "qvalue": [1.0] * 4,
    })
    result = AUDIT.recompute_methods(frame)
    assert result["qvalue_unique_only"].tolist() == [0.02, 1.0, 0.02, 1.0]
    assert result.loc[1, "pvalue_harmonic"] == 1.0
    assert result.loc[1, "pvalue_harmonic_calibrated"] == 1.0


def test_same_draw_simulation_includes_all_methods():
    result = AUDIT.simulate_combiner_type1(iterations=500, seed=7)
    assert set(result["method"]) == set(AUDIT.METHODS)
    assert result.shape[0] == len(AUDIT.METHODS) * len(AUDIT.SIMULATION_RHOS) * len(AUDIT.SIMULATION_ALPHAS)
    assert np.isfinite(result["empirical_type1"]).all()


def test_individual_source_aggregation_sums_calls_without_merging_bh_families():
    source_rows = pd.DataFrame(
        {
            "source_id": ["individual-a", "individual-b"],
            "scenario": ["Mix24X individual", "Mix24X individual"],
            "method": ["harmonic-mean", "harmonic-mean"],
            "q_cutoff": [0.01, 0.01],
            "analysis_unit_count": [1, 1],
            "candidate_rows": [10, 10],
            "calls": [2, 1],
            "expected_denominator": [1, 1],
            "expected_calls": [1, 1],
            "additional_calls": [1, 0],
            "zero_unique_calls": [0, 0],
            "one_unique_calls": [1, 1],
            "lost_vs_fisher": [0, 0],
            "rescued_vs_fisher": [0, 0],
        }
    )
    aggregated = AUDIT.aggregate_scenarios(source_rows)
    assert aggregated.loc[0, "source_table_count"] == 2
    assert aggregated.loc[0, "calls"] == 3
    assert aggregated.loc[0, "additional_calls"] == 1
