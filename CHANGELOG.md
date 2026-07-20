# Changelog

## 1.4.0 - 2026-07-15

### Added

- Added explicit `all-samples`, `per-sample`, and metadata-based analysis-unit definitions.
- Added the versioned `metaumbra.genome_selection_manifest.v1` downstream contract.
- Added unified unit-level result, cohort summary, sample mapping, schema, and diagnostic artifacts.

### Changed

- Unified global and grouped genome selection around one analysis-unit scoring engine.
- Applied a moderately more permissive empirical-background calibration to `all-samples` while retaining the conservative grouped-unit profile.
- Set grouped-unit empirical-background calibration to 5% initial exclusion with adaptive 2--20% bounds.
- Updated the CLI, GUI, and Python workflow to use the same unit-definition and scoring backend.
- Made genome processing order deterministic so repeated Monte Carlo runs produce stable results.
- Placed Sample / Unit Mapping directly below Analysis Unit Definition in the GUI.

### Removed

- Removed the separate pooled/global q-value and ranking implementation.
- Removed the experimental `--unit-specific` workflow and legacy unit-specific manifest contract.

### Fixed

- Restored `knockoff_top_n_targets` enforcement in the unified per-analysis-unit scoring worker.
- Fixed run summaries and diagnostics mixing pooled and per-unit statistics.
- Fixed coverage diagnostics so they are calculated from evidence within each analysis unit.
