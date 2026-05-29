"""Shared statistical utilities for genome-presence scoring."""

from typing import Dict, List, Optional, Tuple

import numpy as np


MIN_PVALUE = 1e-300
DEFAULT_UNIQUE_PVALUE_MODE = "empirical-background"
DEFAULT_UNIQUE_COUNT_POWER = 1.0
DEFAULT_UNIQUE_PEPTIDE_ERROR_SOURCE = "global-alpha"
UNIQUE_PVALUE_CANONICAL_MODES = (
    "empirical-background",
    "hypergeometric-opportunity",
    "alpha-upper-bound",
)
UNIQUE_PEPTIDE_ERROR_SOURCES = (
    "global-alpha",
    "peptide-error-column",
)


def _clip_pvalue(p: float) -> float:
    return float(np.clip(float(p), MIN_PVALUE, 1.0))


def _normalize_unique_pvalue_mode(mode: Optional[str]) -> str:
    normalized = str(mode or DEFAULT_UNIQUE_PVALUE_MODE).strip().lower()
    if normalized not in UNIQUE_PVALUE_CANONICAL_MODES:
        choices = "', '".join(UNIQUE_PVALUE_CANONICAL_MODES)
        raise ValueError(f"unique_pvalue_mode must be one of '{choices}'.")
    return normalized


def _normalize_unique_peptide_error_source(source: Optional[str]) -> str:
    normalized = str(source or DEFAULT_UNIQUE_PEPTIDE_ERROR_SOURCE).strip().lower()
    if normalized not in UNIQUE_PEPTIDE_ERROR_SOURCES:
        choices = "', '".join(UNIQUE_PEPTIDE_ERROR_SOURCES)
        raise ValueError(f"unique_peptide_error_source must be one of '{choices}'.")
    return normalized


def _effective_unique_count(
    u_raw: int,
    power: float = DEFAULT_UNIQUE_COUNT_POWER,
) -> float:
    """Convert a raw unique peptide count into a tempered effective count."""
    u_raw = int(max(u_raw, 0))
    if u_raw == 0:
        return 0.0

    power = float(power)
    if not np.isfinite(power) or not (0 < power <= 1):
        raise ValueError("unique_count_power must be in the interval (0, 1].")

    u_eff = min(float(u_raw), float(u_raw) ** power)

    return float(max(u_eff, 0.0))


def _tempered_unique_error_product_pvalue(
    unique_peptides: List[str],
    alpha: float,
    peptide_error_upper_by_peptide: Dict[str, float],
    error_source: str,
    unique_count_power: float,
) -> Tuple[float, float, str]:
    u_raw = int(len(unique_peptides))
    if u_raw <= 0:
        return 1.0, 0.0, "none"

    error_source = _normalize_unique_peptide_error_source(error_source)
    u_eff = _effective_unique_count(
        u_raw=u_raw,
        power=unique_count_power,
    )
    temper = float(u_eff) / float(max(u_raw, 1))

    if error_source == "global-alpha":
        log_eps_sum = float(u_raw) * float(np.log(alpha))
    else:
        missing = [peptide for peptide in unique_peptides if peptide not in peptide_error_upper_by_peptide]
        if missing:
            preview = ", ".join(sorted(missing)[:10])
            suffix = " ..." if len(missing) > 10 else ""
            raise ValueError(
                "unique_peptide_error_source='peptide-error-column' requires an error value for every unique peptide. "
                f"Missing {len(missing)} peptide(s): {preview}{suffix}"
            )
        errs = np.asarray([peptide_error_upper_by_peptide[peptide] for peptide in unique_peptides], dtype=float)
        errs = np.clip(errs, 1e-12, 1.0)
        log_eps_sum = float(np.sum(np.log(errs)))

    log_p = float(log_eps_sum) * temper
    p_unique = float(np.exp(max(log_p, np.log(MIN_PVALUE))))
    return _clip_pvalue(p_unique), float(u_eff), error_source
