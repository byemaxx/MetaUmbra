import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "results"
    / "revision_2026-08"
    / "analyses"
    / "k2_combiner_final_validation"
    / "run_k2_combiner_final_validation.py"
)
SPEC = importlib.util.spec_from_file_location("k2_combiner_final_validation", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def test_k2_recompute_gates_bonferroni_and_preserves_its_ordering_against_hmp():
    frame = pd.DataFrame(
        {
            "source_table_sha256": ["a", "a", "a"],
            "analysis_unit_id": ["unit"] * 3,
            "genome_id": ["g1", "g2", "g3"],
            "num_peptides_matched": [1, 1, 1],
            "num_peptides_unique": [1, 0, 1],
            "pvalue_unique": [0.06, 1.0, 0.03],
            "pvalue_shared": [0.001, 0.001, 0.03],
            "pvalue": [AUDIT.fisher_p_2(0.06, 0.001), AUDIT.fisher_p_2(1.0, 0.001), AUDIT.fisher_p_2(0.03, 0.03)],
            "qvalue": [1.0, 1.0, 1.0],
        }
    )
    result = AUDIT.recompute_methods(frame)
    assert result.loc[1, "pvalue_bonferroni"] == 1.0
    assert result.loc[1, "pvalue_harmonic_calibrated"] == 1.0
    assert np.all(
        result["pvalue_bonferroni"]
        <= result["pvalue_harmonic_calibrated"] + 1e-15
    )
    assert result.loc[0, "pvalue_bonferroni"] == 0.002
