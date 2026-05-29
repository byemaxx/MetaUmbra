"""Pooled peptide-set scoring helpers."""

import pandas as pd

from .ranking import bh_qvalues, qvalues_to_presence_scores


def finalize_pooled_presence_columns(
    out: pd.DataFrame,
    target_mask: pd.Series,
) -> pd.DataFrame:
    """Finalize pooled q-values, presence scores, and pass flags in-place."""
    all_p = out.loc[target_mask, "p_presence"].to_numpy(dtype=float)
    out.loc[target_mask, "q_presence"] = bh_qvalues(all_p)
    qvals = pd.to_numeric(out["q_presence"], errors="coerce")
    valid = qvals.notna()
    out.loc[valid, "presence_score"] = qvalues_to_presence_scores(qvals.loc[valid].to_numpy(dtype=float))

    out["pass_q_0_01"] = (out["q_presence"] <= 0.01) & (out["_genomes_with_any_match"])
    out["pass_q_0_05"] = (out["q_presence"] <= 0.05) & (out["_genomes_with_any_match"])
    return out
