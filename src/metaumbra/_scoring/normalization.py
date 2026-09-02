"""Peptide-sequence normalization policies used by scoring and caches."""

from __future__ import annotations

from typing import Iterable


DEFAULT_PEPTIDE_NORMALIZATION_POLICY = "il-equivalent"
PEPTIDE_NORMALIZATION_POLICIES = ("il-equivalent", "exact")


def normalize_peptide_policy(policy: str | None) -> str:
    normalized = str(policy or DEFAULT_PEPTIDE_NORMALIZATION_POLICY).strip().lower()
    if normalized not in PEPTIDE_NORMALIZATION_POLICIES:
        choices = "', '".join(PEPTIDE_NORMALIZATION_POLICIES)
        raise ValueError(f"peptide_normalization_policy must be one of '{choices}'.")
    return normalized


def normalize_peptide_sequence(sequence: object, policy: str | None = None) -> str:
    """Normalize one peptide consistently for observed/theoretical matching.

    ``il-equivalent`` maps both isoleucine and leucine to ``J`` because they are
    isobaric in conventional tandem mass spectrometry.  Q/K and AD/W are not
    collapsed: they are not exact isobaric substitutions under the relevant
    high-resolution acquisition conditions.
    """
    normalized_policy = normalize_peptide_policy(policy)
    peptide = str(sequence).strip().upper()
    if normalized_policy == "il-equivalent":
        peptide = peptide.replace("I", "J").replace("L", "J")
    return peptide


def normalize_peptide_collection(
    sequences: Iterable[object], policy: str | None = None
) -> set[str]:
    return {
        normalized
        for sequence in sequences
        if (normalized := normalize_peptide_sequence(sequence, policy))
    }

