from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from .__version__ import __version__


SCHEMA_VERSION = "metaumbra.genome_selection_manifest.v1"
GENOME_THRESHOLDS = ("q0.05", "q0.01")


def validate_genome_selection_manifest(data: dict[str, Any]) -> None:
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    generated_by = data.get("generated_by")
    if not isinstance(generated_by, dict) or generated_by.get("software") != "MetaUmbra":
        raise ValueError("generated_by.software must be 'MetaUmbra'")
    unit_definition = data.get("unit_definition")
    if not isinstance(unit_definition, dict):
        raise ValueError("unit_definition must be an object")
    units = data.get("units")
    if not isinstance(units, dict) or not units:
        raise ValueError("units must contain at least one analysis unit")
    if int(unit_definition.get("n_units", -1)) != len(units):
        raise ValueError("unit_definition.n_units does not match units")
    selection = data.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("selection must be an object")
    default_threshold = selection.get("default_genome_threshold")
    available = selection.get("available_genome_thresholds")
    if default_threshold not in GENOME_THRESHOLDS or available != list(GENOME_THRESHOLDS):
        raise ValueError("selection thresholds must be q0.05 and q0.01")

    seen_samples: dict[str, str] = {}
    for unit_id, payload in units.items():
        if not isinstance(payload, dict):
            raise ValueError(f"Unit {unit_id!r} must be an object")
        samples = payload.get("sample_ids")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"Unit {unit_id!r} must contain sample_ids")
        if int(payload.get("n_samples", -1)) != len(samples):
            raise ValueError(f"Unit {unit_id!r} has inconsistent n_samples")
        for sample in samples:
            sample = str(sample)
            if sample in seen_samples:
                raise ValueError(
                    f"Sample {sample!r} is assigned to multiple units: "
                    f"{seen_samples[sample]!r} and {unit_id!r}"
                )
            seen_samples[sample] = str(unit_id)
        q005 = payload.get("genome_ids_q005")
        q001 = payload.get("genome_ids_q001")
        if not isinstance(q005, list) or not isinstance(q001, list):
            raise ValueError(f"Unit {unit_id!r} must contain both genome threshold lists")
        if not set(map(str, q001)).issubset(set(map(str, q005))):
            raise ValueError(f"Unit {unit_id!r} q0.01 genomes are not a subset of q0.05")


def build_genome_selection_manifest(
    *,
    mapping_df: pd.DataFrame,
    unit_genome_results: pd.DataFrame,
    unit_mode: str,
    sample_id_column: str,
    analysis_unit_column: str | None,
    peptide_table_path: str,
    metadata_table_path: str | None,
    genome_digest_directories: list[str],
    artifacts: dict[str, str],
    scoring_method: str,
    warnings: list[str] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    required_mapping = {"sample_id", "analysis_unit_id"}
    required_results = {"analysis_unit_id", "genome_id", "pass_q_0_01", "pass_q_0_05"}
    if not required_mapping.issubset(mapping_df.columns):
        raise ValueError("sample mapping is missing required columns")
    if not required_results.issubset(unit_genome_results.columns):
        raise ValueError("unit genome results are missing required columns")

    units: dict[str, dict[str, Any]] = {}
    mapping_included = mapping_df
    if "included" in mapping_included.columns:
        mapping_included = mapping_included[mapping_included["included"].fillna(False).astype(bool)]
    for unit_id, group in mapping_included.groupby("analysis_unit_id", sort=False):
        unit_key = str(unit_id)
        samples = group["sample_id"].astype(str).tolist()
        result_group = unit_genome_results[
            unit_genome_results["analysis_unit_id"].astype(str) == unit_key
        ]
        q005 = result_group.loc[
            result_group["pass_q_0_05"].fillna(False).astype(bool), "genome_id"
        ].astype(str).tolist()
        q001 = result_group.loc[
            result_group["pass_q_0_01"].fillna(False).astype(bool), "genome_id"
        ].astype(str).tolist()
        units[unit_key] = {
            "sample_ids": samples,
            "n_samples": len(samples),
            "genome_ids_q005": q005,
            "genome_ids_q001": q001,
        }

    peptide_path = Path(peptide_table_path).expanduser().resolve()
    metadata_path = (
        {"path": str(Path(metadata_table_path).expanduser().resolve()), "format": "csv" if Path(metadata_table_path).suffix.lower() == ".csv" else "tsv"}
        if metadata_table_path
        else None
    )
    suffix = peptide_path.suffix.lower()
    peptide_format = "diann_parquet" if suffix in {".parquet", ".pq"} else ("csv" if suffix == ".csv" else "tsv")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": {
            "software": "MetaUmbra",
            "version": str(__version__),
            "run_id": run_id or str(uuid4()),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "unit_definition": {
            "mode": unit_mode,
            "sample_id_column": sample_id_column,
            "analysis_unit_column": analysis_unit_column if unit_mode == "metadata" else None,
            "n_units": len(units),
        },
        "selection": {
            "default_genome_threshold": "q0.05",
            "available_genome_thresholds": list(GENOME_THRESHOLDS),
            "scoring_method": scoring_method,
        },
        "inputs": {
            "peptide_table": {"path": str(peptide_path), "format": peptide_format},
            "metadata_table": metadata_path,
            "genome_digest_directories": [
                str(Path(path).expanduser().resolve()) for path in genome_digest_directories
            ],
        },
        "units": units,
        "artifacts": dict(artifacts),
        "warnings": list(warnings or []),
    }
    validate_genome_selection_manifest(manifest)
    return manifest


def write_genome_selection_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    validate_genome_selection_manifest(manifest)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_genome_selection_manifest(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_genome_selection_manifest(data)
    return data

