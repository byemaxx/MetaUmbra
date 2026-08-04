import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "results"
    / "revision_2026-08"
    / "analyses"
    / "hmp_sensitivity"
    / "run_hmp_sensitivity.py"
)
SPEC = importlib.util.spec_from_file_location("hmp_sensitivity_audit", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def test_recompute_methods_keeps_bh_families_separate_by_source_table():
    frame = pd.DataFrame(
        {
            "analysis_unit_id": ["__global__"] * 4,
            "source_table": ["source-a", "source-a", "source-b", "source-b"],
            "genome_id": ["g1", "g2", "g3", "g4"],
            "num_peptides_matched": [1, 1, 1, 1],
            "num_peptides_unique": [1, 1, 1, 1],
            "pvalue_unique": [0.01, 1.0, 0.01, 1.0],
            "pvalue_shared": [1.0, 1.0, 1.0, 1.0],
            "pvalue": [0.05605170185988092, 1.0, 0.05605170185988092, 1.0],
        }
    )

    result = AUDIT.recompute_methods(frame)

    assert result["qvalue_unique_only"].tolist() == pytest.approx([0.02, 1.0, 0.02, 1.0])


def test_recompute_methods_applies_the_hmp_unique_evidence_gate():
    frame = pd.DataFrame(
        {
            "analysis_unit_id": ["unit", "unit"],
            "source_table": ["source", "source"],
            "genome_id": ["shared-only", "unique-supported"],
            "num_peptides_matched": [1, 1],
            "num_peptides_unique": [0, 1],
            "pvalue_unique": [1.0, 0.001],
            "pvalue_shared": [0.001, 1.0],
            "pvalue": [0.014815510557964274, 0.007907755278982137],
        }
    )

    result = AUDIT.recompute_methods(frame)

    assert result.loc[0, "pvalue_harmonic"] == 1.0
    assert result.loc[1, "pvalue_harmonic"] == pytest.approx(2.0 * 0.001 / 1.001)
