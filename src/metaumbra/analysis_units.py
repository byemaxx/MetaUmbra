from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import pandas as pd


UNIT_MODES = ("all-samples", "per-sample", "metadata")
GLOBAL_UNIT_ID = "__global__"


def _normalize_sample_id(value: object) -> str:
    return re.sub(r"\.raw$", "", str(value).strip(), flags=re.IGNORECASE)


@dataclass(frozen=True)
class AnalysisUnitDefinition:
    mode: str = "all-samples"
    sample_id_column: str = "Run"
    analysis_unit_column: str | None = None

    def validate(self) -> None:
        if self.mode not in UNIT_MODES:
            raise ValueError(f"unit_mode must be one of: {', '.join(UNIT_MODES)}")
        if not str(self.sample_id_column).strip():
            raise ValueError("sample_id_column must not be empty")
        if self.mode == "metadata" and not str(self.analysis_unit_column or "").strip():
            raise ValueError("analysis_unit_column is required for metadata unit mode")


def load_metadata_mapping(
    metadata_table_path: str | Path,
    *,
    sample_id_column: str,
    analysis_unit_column: str,
) -> pd.DataFrame:
    path = Path(metadata_table_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Metadata table not found: {path}")
    separator = "," if path.suffix.lower() == ".csv" else "\t"
    metadata = pd.read_csv(path, sep=separator, dtype="string")
    missing_columns = [
        column
        for column in (sample_id_column, analysis_unit_column)
        if column not in metadata.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Metadata table is missing required columns {missing_columns}; "
            f"available columns: {metadata.columns.tolist()}"
        )
    metadata = metadata.copy()
    metadata[sample_id_column] = (
        metadata[sample_id_column]
        .astype("string")
        .str.strip()
        .str.replace(r"\.raw$", "", case=False, regex=True)
    )
    metadata[analysis_unit_column] = metadata[analysis_unit_column].astype("string").str.strip()
    invalid_sample_ids = metadata[sample_id_column].isna() | (metadata[sample_id_column] == "")
    if bool(invalid_sample_ids.any()):
        raise ValueError("Metadata contains empty sample IDs")
    included = pd.Series(True, index=metadata.index)
    if "included" in metadata.columns:
        included_values = metadata["included"].astype("string").str.strip().str.lower()
        included = ~included_values.isin({"0", "false", "no", "n", "off"})
    invalid_unit_ids = metadata[analysis_unit_column].isna() | (metadata[analysis_unit_column] == "")
    if bool((included & invalid_unit_ids).any()):
        raise ValueError("Metadata contains empty analysis unit IDs for included samples")
    duplicates = metadata[metadata[sample_id_column].duplicated(keep=False)]
    if not duplicates.empty:
        duplicate_ids = sorted(duplicates[sample_id_column].astype(str).unique().tolist())
        raise ValueError(
            "Each sample must occur exactly once in metadata; duplicate sample IDs: "
            + ", ".join(duplicate_ids[:10])
        )
    return metadata


def build_sample_unit_mapping(
    sample_ids: Iterable[str],
    definition: AnalysisUnitDefinition,
    *,
    metadata_table_path: str | Path | None = None,
    metadata_sample_id_column: str = "sample_id",
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Create the one authoritative sample-to-analysis-unit assignment."""
    definition.validate()
    samples = [_normalize_sample_id(value) for value in sample_ids]
    if not samples or any(not value for value in samples):
        raise ValueError("At least one non-empty sample ID is required")
    if len(samples) != len(set(samples)):
        raise ValueError("Peptide input contains duplicate normalized sample IDs")

    if definition.mode == "all-samples":
        mapping = {sample: GLOBAL_UNIT_ID for sample in samples}
        return pd.DataFrame(
            {"sample_id": samples, "analysis_unit_id": [GLOBAL_UNIT_ID] * len(samples)}
        ), None
    if definition.mode == "per-sample":
        return pd.DataFrame(
            {"sample_id": samples, "analysis_unit_id": samples}
        ), None

    if not metadata_table_path:
        raise ValueError("metadata_table_path is required for metadata unit mode")
    metadata = load_metadata_mapping(
        metadata_table_path,
        sample_id_column=metadata_sample_id_column,
        analysis_unit_column=str(definition.analysis_unit_column),
    )
    metadata_samples = set(metadata[metadata_sample_id_column].astype(str))
    missing = sorted(set(samples) - metadata_samples)
    extra = sorted(metadata_samples - set(samples))
    if missing:
        raise ValueError(
            "Metadata has no analysis unit assignment for peptide-table samples: "
            + ", ".join(missing[:10])
        )
    if extra:
        metadata = metadata[metadata[metadata_sample_id_column].isin(samples)].copy()
    unit_column = str(definition.analysis_unit_column)
    unit_by_sample = dict(
        zip(
            metadata[metadata_sample_id_column].astype(str),
            metadata[unit_column].fillna("").astype(str),
        )
    )
    mapping = pd.DataFrame(
        {
            "sample_id": samples,
            "analysis_unit_id": [unit_by_sample[sample] for sample in samples],
        }
    )
    return mapping, metadata.rename(
        columns={metadata_sample_id_column: "sample_id", unit_column: "analysis_unit_id"}
    )
