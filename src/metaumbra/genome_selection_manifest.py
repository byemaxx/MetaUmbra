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


def _require_object_fields(
    value: object,
    *,
    name: str,
    required: set[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{name} is missing required fields: {', '.join(missing)}")
    return value


def _require_string_list(value: object, *, name: str, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        qualifier = "a non-empty" if nonempty else "an"
        raise ValueError(f"{name} must be {qualifier} array")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must contain only strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} must not contain duplicates")
    return value


def validate_genome_selection_manifest(data: dict[str, Any]) -> None:
    required_top_level = {
        "schema_version",
        "generated_by",
        "unit_definition",
        "selection",
        "inputs",
        "units",
        "artifacts",
        "warnings",
    }
    data = _require_object_fields(
        data,
        name="manifest",
        required=required_top_level,
    )
    unexpected = sorted(set(data) - required_top_level)
    if unexpected:
        raise ValueError(f"manifest contains unexpected fields: {', '.join(unexpected)}")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    generated_by = _require_object_fields(
        data["generated_by"],
        name="generated_by",
        required={"software", "version", "run_id", "generated_at"},
    )
    if generated_by["software"] != "MetaUmbra":
        raise ValueError("generated_by.software must be 'MetaUmbra'")
    for field in ("version", "run_id", "generated_at"):
        if not isinstance(generated_by[field], str):
            raise ValueError(f"generated_by.{field} must be a string")
    try:
        generated_at = datetime.fromisoformat(generated_by["generated_at"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("generated_by.generated_at must be an RFC 3339 date-time") from exc
    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise ValueError("generated_by.generated_at must include a timezone")

    unit_definition = _require_object_fields(
        data["unit_definition"],
        name="unit_definition",
        required={"mode", "sample_id_column", "analysis_unit_column", "n_units"},
    )
    if unit_definition["mode"] not in {"all-samples", "per-sample", "metadata"}:
        raise ValueError("unit_definition.mode is invalid")
    if not isinstance(unit_definition["sample_id_column"], str):
        raise ValueError("unit_definition.sample_id_column must be a string")
    if unit_definition["analysis_unit_column"] is not None and not isinstance(
        unit_definition["analysis_unit_column"], str
    ):
        raise ValueError("unit_definition.analysis_unit_column must be a string or null")
    n_units = unit_definition["n_units"]
    if not isinstance(n_units, int) or isinstance(n_units, bool) or n_units < 1:
        raise ValueError("unit_definition.n_units must be a positive integer")

    units = data["units"]
    if not isinstance(units, dict) or not units:
        raise ValueError("units must contain at least one analysis unit")
    if n_units != len(units):
        raise ValueError("unit_definition.n_units does not match units")
    selection = _require_object_fields(
        data["selection"],
        name="selection",
        required={"default_genome_threshold", "available_genome_thresholds", "scoring_method"},
    )
    default_threshold = selection["default_genome_threshold"]
    available = selection["available_genome_thresholds"]
    if default_threshold not in GENOME_THRESHOLDS or available != list(GENOME_THRESHOLDS):
        raise ValueError("selection thresholds must be q0.05 and q0.01")
    if not isinstance(selection["scoring_method"], str):
        raise ValueError("selection.scoring_method must be a string")
    if not isinstance(data["inputs"], dict):
        raise ValueError("inputs must be an object")
    if not isinstance(data["artifacts"], dict):
        raise ValueError("artifacts must be an object")
    warnings = data["warnings"]
    if not isinstance(warnings, list) or not all(isinstance(item, str) for item in warnings):
        raise ValueError("warnings must be an array of strings")

    seen_samples: dict[str, str] = {}
    for unit_id, payload in units.items():
        if not isinstance(unit_id, str):
            raise ValueError("Unit IDs must be strings")
        payload = _require_object_fields(
            payload,
            name=f"Unit {unit_id!r}",
            required={"sample_ids", "n_samples", "genome_ids_q005", "genome_ids_q001"},
        )
        samples = _require_string_list(
            payload["sample_ids"],
            name=f"Unit {unit_id!r} sample_ids",
            nonempty=True,
        )
        n_samples = payload["n_samples"]
        if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples < 1:
            raise ValueError(f"Unit {unit_id!r} n_samples must be a positive integer")
        if n_samples != len(samples):
            raise ValueError(f"Unit {unit_id!r} has inconsistent n_samples")
        for sample in samples:
            if sample in seen_samples:
                raise ValueError(
                    f"Sample {sample!r} is assigned to multiple units: "
                    f"{seen_samples[sample]!r} and {unit_id!r}"
                )
            seen_samples[sample] = str(unit_id)
        q005 = _require_string_list(
            payload["genome_ids_q005"], name=f"Unit {unit_id!r} genome_ids_q005"
        )
        q001 = _require_string_list(
            payload["genome_ids_q001"], name=f"Unit {unit_id!r} genome_ids_q001"
        )
        if not set(q001).issubset(set(q005)):
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
