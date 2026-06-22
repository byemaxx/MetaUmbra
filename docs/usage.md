# MetaUmbra Usage Guide

MetaUmbra infers statistically supported genome-level presence from metaproteomic peptide evidence. A typical analysis starts with protein FASTA files for candidate genomes, digests those FASTA files into theoretical peptide tables, and then scores an observed peptide table against the genome peptide references.

This guide covers installation, GUI usage, command-line usage, input formats, output interpretation, and common troubleshooting steps.

## Contents

- [Installation](#installation)
- [Core workflow](#core-workflow)
- [Recommended minimal workflow](#recommended-minimal-workflow)
- [Using the GUI](#using-the-gui)
- [Using the command line](#using-the-command-line)
- [Input formats](#input-formats)
- [Output files](#output-files)
- [Interpreting results](#interpreting-results)
- [Performance and reproducibility](#performance-and-reproducibility)
- [Troubleshooting](#troubleshooting)

## Installation

MetaUmbra requires Python 3.10 or newer.

For most users, install the package with all optional features, including GUI support and parquet support for DIA-NN `report.parquet` extraction:

```bash
pip install "metaumbra[all]"
```

Minimal and modular installations are also available:

```bash
pip install metaumbra              # command-line tools only
pip install "metaumbra[gui]"      # command-line tools plus GUI support
pip install "metaumbra[gui-pyqt5]" # command-line tools plus PyQt5 GUI support
pip install "metaumbra[parquet]"  # command-line tools plus parquet extraction support
```

For local development from a source checkout:

```bash
pip install -e ".[all]"
```

Check that the command-line entry point is available:

```bash
metaumbra --version
metaumbra --help
```

## Core workflow

MetaUmbra has three main workflow steps:

1. Digest candidate genome protein FASTA files into theoretical peptide TSV files.
2. Prepare observed peptides from a metaproteomics search output.
3. Score genome presence by matching observed peptides against the genome digest tables.

The command-line workflow looks like this:

```bash
metaumbra digest \
  --input-dir data/genome_fastas \
  --output-dir results/genome_digests \
  --enzyme-id 42 \
  --min-length 7 \
  --max-length 30 \
  --max-miscleavages 2

metaumbra extract-parquet \
  --input data/report.parquet \
  --output results/observed_peptides.tsv \
  --force

metaumbra score \
  --peptide-table results/observed_peptides.tsv \
  --genome-digest-dirs results/genome_digests \
  --output results/genome_presence.tsv
```

The parquet extraction step is optional. If you already have a peptide TSV, skip `extract-parquet`. If genome digest tables have not been generated yet, run `digest` before `score`; otherwise, run `score` directly.

## Recommended minimal workflow

For most analyses:

1. Put one protein FASTA file per genome in a genome FASTA directory.
2. Digest the FASTA files once to create per-genome peptide TSV files.
3. Prepare an observed peptide TSV containing at least a peptide sequence column.
4. Run genome presence scoring against the digest directory.
5. Interpret `qvalue`, `presence_score`, `num_peptides_unique`, `shared_fraction`, and `mean_degeneracy`.

Reference construction only needs to be repeated when the genome collection or digestion settings change. Scoring can be repeated with different observed peptide tables, selected genome IDs, excluded genome IDs, or q-value thresholds.

## Using the GUI

Start the graphical interface with either command:

```bash
metaumbra-gui
```

or:

```bash
metaumbra gui
```

The GUI contains two main tabs:

- `Genome Presence Scoring`: load an observed peptide table, one or more genome digest directories, optional genome lineage annotations, and write the result TSV.
- `Digest FASTA`: digest one protein FASTA file or a directory of protein FASTA files into theoretical peptide TSV files.

The scoring tab also includes `Import Parquet...`, which converts a DIA-NN-style parquet report into a MetaUmbra-compatible peptide TSV. By default it maps:

The observed peptide table field also accepts DIA-NN `report.parquet` files directly. When you click Run, the GUI auto-detects the expected columns and loads them into memory for scoring without writing a separate TSV. Use `Import Parquet...` if you need to customize the column mapping or export a peptide TSV.

For multi-sample input, enable `Sample / Unit-aware Scoring`. The `Configure Sample / Unit Mapping` button reads the selected TSV/CSV/parquet table, detects sample IDs from the configured sample column, shows `n_total_rows` and `n_valid_peptides`, and lets you assign selected samples to a shared `analysis_unit_id`. You can also import or export a mapping TSV. A separate metadata table is still supported through the metadata table field.

| Parquet column | Output TSV column |
| --- | --- |
| `Run` | `Run` |
| `Stripped.Sequence` | `Sequence` |
| `Evidence` | `Evidence` |
| `Q.Value` | `Q.Value` |

The GUI can save and load its run configuration. Use this for repeated analyses where the same digest directories, column names, and runtime settings should be reused. Advanced GUI options allow users to adjust Monte Carlo iterations, stage-2 refinement, random seed, cache behavior, genome inclusion or exclusion lists, and diagnostic artifact export.

## Using the command line

MetaUmbra exposes separate subcommands for the main tasks:

```bash
metaumbra digest --help
metaumbra score --help
metaumbra extract-parquet --help
```

### Digest FASTA files

Digest one FASTA file:

```bash
metaumbra digest \
  --input-file data/genome_A.faa \
  --output-file results/genome_digests/genome_A.tsv
```

Digest a directory of FASTA files:

```bash
metaumbra digest \
  --input-dir data/genome_fastas \
  --output-dir results/genome_digests
```

Important options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--enzyme-id` | `42` | RPG enzyme ID used for in-silico digestion. |
| `--min-length` | `7` | Minimum peptide length retained in digest output. |
| `--max-length` | `30` | Maximum peptide length retained in digest output. |
| `--max-miscleavages` | `2` | Maximum number of missed cleavages. |
| `--processes` | all CPU cores | Worker count. In directory mode, files are parallelized across workers. |
| `--full-header` | off | Keep full FASTA headers instead of truncating at the first space. |
| `--no-skip-existing` | off | Rebuild existing digest outputs in directory mode. |
| `--quiet` | off | Reduce runtime log output. |

Digest output is a tab-separated file with two columns:

| Column | Description |
| --- | --- |
| `Protein` | Protein identifier from the FASTA header. |
| `Peptide` | Theoretical peptide sequence retained after length filtering. |

In directory mode, each input FASTA file creates one TSV file named after the input file stem. For example, `Genome_001.faa` becomes `Genome_001.tsv`, and `Genome_001` is later used as the `genome_id`.

### Extract peptide TSV from parquet

Use this when your peptide evidence is in a parquet file such as a DIA-NN `report.parquet`.

```bash
metaumbra extract-parquet \
  --input data/report.parquet \
  --output results/observed_peptides.tsv \
  --force
```

Default column mapping:

| Input parquet column | Output TSV column |
| --- | --- |
| `Run` | `Run` |
| `Stripped.Sequence` | `Sequence` |
| `Evidence` | `Evidence` |
| `Q.Value` | `Q.Value` |

To customize columns, repeat `--input-column` and `--output-column` in the same order:

```bash
metaumbra extract-parquet \
  --input data/report.parquet \
  --output results/observed_peptides.tsv \
  --input-column Run \
  --input-column Stripped.Sequence \
  --input-column Evidence \
  --input-column Q.Value \
  --output-column Run \
  --output-column Sequence \
  --output-column Evidence \
  --output-column Q.Value \
  --force
```

### Score genome presence

Run scoring with default column names:

```bash
metaumbra score \
  --peptide-table results/observed_peptides.tsv \
  --genome-digest-dirs results/genome_digests \
  --output results/genome_presence.tsv
```

Use multiple digest directories with a comma- or semicolon-separated list:

```bash
metaumbra score \
  --peptide-table results/observed_peptides.tsv \
  --genome-digest-dirs "results/isolate_digests;results/mag_digests" \
  --output results/genome_presence.tsv
```

Use custom peptide-table columns:

```bash
metaumbra score \
  --peptide-table results/observed_peptides.tsv \
  --genome-digest-dirs results/genome_digests \
  --output results/genome_presence.tsv \
  --peptide-seq-col Sequence \
  --peptide-score-col score \
  --peptide-error-col Q.Value \
  --peptide-error-cutoff 0.05
```

Disable decoy filtering if your table has no decoy flag column:

```bash
metaumbra score \
  --peptide-table results/observed_peptides.tsv \
  --genome-digest-dirs results/genome_digests \
  --output results/genome_presence.tsv \
  --peptide-decoy-flag-col ""
```

Restrict or exclude genomes by genome ID:

```bash
metaumbra score \
  --peptide-table results/observed_peptides.tsv \
  --genome-digest-dirs results/genome_digests \
  --output results/genome_presence_subset.tsv \
  --selected-genome-ids "Genome_001;Genome_002" \
  --exclude-genome-ids Genome_contaminant
```

For large ID lists, put one genome ID per line in a text file:

```text
Genome_001
Genome_002
Genome_003
```

Then pass the file to the matching include or exclude option with `@`:

```bash
metaumbra score \
  --peptide-table results/observed_peptides.tsv \
  --genome-digest-dirs results/genome_digests \
  --output results/genome_presence_subset.tsv \
  --selected-genome-ids @selected_genomes.txt \
  --exclude-genome-ids @excluded_genomes.txt
```

Add lineage annotations:

```bash
metaumbra score \
  --peptide-table results/observed_peptides.tsv \
  --genome-digest-dirs results/genome_digests \
  --output results/genome_presence.tsv \
  --lineage-table data/genome_lineage.tsv \
  --lineage-genome-id-col genome_id \
  --lineage-lineage-col Lineage
```

Use a matched-peptide cache for repeated scoring over the same observed peptides and genome digest files:

```bash
metaumbra score \
  --peptide-table results/observed_peptides.tsv \
  --genome-digest-dirs results/genome_digests \
  --output results/genome_presence.tsv \
  --save-cache \
  --cache-path results/genome_presence_artifacts/matched_peptides.pkl \
  --use-cache-if-exists
```

Important scoring options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--peptide-seq-col` | `Sequence` | Observed peptide sequence column. |
| `--peptide-score-col` | `Evidence` | Peptide score column. If missing, all peptides receive score `1`. |
| `--peptide-error-col` | `Q.Value` | Peptide error, FDR, PEP, or q-value column used for filtering. |
| `--peptide-error-cutoff` | `0.05` | Keep peptides with error values less than or equal to this cutoff. |
| `--unique-pvalue-mode` | `empirical-background` | Unique evidence p-value mode. `empirical-background` estimates a sample-specific weak-genome unique peptide background threshold and accumulates only unique peptides above that threshold. `hypergeometric-opportunity` uses the observed genome-unique peptide pool and theoretical unique peptide opportunity. `alpha-upper-bound` uses `alpha^(U_raw^power)`. |
| `--unique-peptide-error-source` | `global-alpha` | Error source for `alpha-upper-bound`: use the global alpha or per-peptide values from `--peptide-error-col`. |
| `--unique-count-power` | `1.0` | Power exponent for `alpha-upper-bound` effective unique evidence count, `U_eff = U_raw^power`. |
| `--unique-empirical-background-threshold-quantile` | `0.95` | Weak-background unique-count quantile used by `empirical-background`; applies to pooled scoring and unit-aware empirical-background scoring. |
| `--unit-aware` | off | Enable per-analysis-unit genome presence scoring for long-format multi-sample peptide tables. |
| `--sample-id-col` | `Run` | Sample or run ID column used by `--unit-aware`. |
| `--intensity-col` | `Precursor.Quantity` | Intensity column used to call sample-level peptide presence for `--unit-aware`. |
| `--intensity-min-value` | `0` | Minimum intensity required for sample-level peptide presence. |
| `--intensity-min-quantile` | `0` | Optional within-sample intensity quantile cutoff for sample-level peptide presence. |
| `--metadata-table` | none | Optional TSV/CSV table mapping samples to analysis units. |
| `--metadata-sample-id-col` | `sample_id` | Sample ID column in `--metadata-table`. |
| `--metadata-analysis-unit-col` | `analysis_unit_id` | Analysis unit column in `--metadata-table`. |
| `--no-export-unit-derived-tables` | off | Deprecated compatibility option. Unit-aware diagnostic tables are now written only with `--export-diagnostics` unless this legacy switch is used by older scripts. |
| `--theoretical-opportunity-cache` | auto | Optional path for the theoretical opportunity cache used by `hypergeometric-opportunity`. |
| `--rebuild-theoretical-opportunity-cache` | off | Rebuild the theoretical opportunity cache even if it already exists. |
| `--num-workers-for-theoretical-opportunity` | same as `--num-workers` | Optional worker process count for theoretical opportunity sharding and reduction. |
| `--single-peptide-error-rate-upper-bound` | `0.05` | Global alpha used by `--unique-pvalue-mode alpha-upper-bound` when `--unique-peptide-error-source global-alpha` is selected. This is separate from peptide filtering. |
| `--peptide-decoy-flag-col` | `Reverse` | Decoy flag column. Pass an empty string to disable. |
| `--decoy-flag-value` | `+` | Value treated as a decoy marker. |
| `--num-workers` | `max(1, cpu_count - 1)` | Worker process count for genome scanning. On Windows, use 60 or fewer workers because `ProcessPoolExecutor` has a platform worker limit. |
| `--selected-genome-ids` | none | Restrict analysis to listed genome IDs. Separate multiple IDs with commas/semicolons, or pass `@file` with one ID per line. |
| `--exclude-genome-ids` | none | Exclude listed genome IDs. Separate multiple IDs with commas/semicolons, or pass `@file` with one ID per line. |
| `--lineage-table` | none | Optional TSV table for adding a `Lineage` column to results. |
| `--cache-path` | none | Explicit matched-peptide cache path. Use with `--save-cache` to write it or `--use-cache-if-exists` to reuse it. |
| `--use-cache-if-exists` | off | Reuse an existing matched-peptide cache if available. |
| `--save-cache` | off | Save matched-peptide cache output. If `--cache-path` is omitted, writes `<stem>_artifacts/matched_peptides.pkl`. |
| `--no-compute-coverage` | off | Skip cumulative coverage calculations. |
| `--export-diagnostics` | off | Write diagnostic/audit artifact tables such as `run_summary.json`, `full_internal_metrics.tsv`, knockoff summaries, pooled unit-aware results, and unit-aware QC pivots. |
| `--no-export-temp` | off | Deprecated compatibility alias for leaving diagnostics disabled. Run parameters, run log, and run status are still written under `<stem>_artifacts/`. |
| `--return-full-table` | off | For non-unit-aware scoring, write the full internal result table instead of only the concise main result. In unit-aware mode, the requested output stays concise and full unit-level audit columns are written under `<stem>_artifacts/unit_aware/`. |

Unique p-value strength is controlled by `--unique-pvalue-mode`. `--unique-peptide-error-source`, `--single-peptide-error-rate-upper-bound`, and `--unique-count-power` apply to `alpha-upper-bound`.

### Unit-aware scoring for multi-sample data

MetaUmbra can score genome presence for user-defined analysis units. An analysis unit may correspond to one sample, a technical replicate group, a donor-level group of repeated runs, or another manually defined group. Peptide presence is first determined within each raw sample using an intensity column, then sample-level peptide presence is aggregated into analysis units before genome-level scoring.

Use unit-aware scoring when the input contains multiple samples or repeated runs:

```bash
metaumbra score \
  --peptide-table results/report.parquet \
  --genome-digest-dirs results/genome_digests \
  --output results/genome_presence.tsv \
  --peptide-seq-col Stripped.Sequence \
  --unit-aware \
  --sample-id-col Run \
  --intensity-col Precursor.Quantity \
  --peptide-error-col Q.Value \
  --metadata-table results/sample_metadata.tsv
```

The peptide table for `--unit-aware` must be long-format: each row should describe one peptide observation in one raw sample or run. For example:

```text
Run	Sequence	Evidence	Q.Value	Reverse	Precursor.Quantity
run_01	PEPTIDEA	0.98	0.001		125000
run_02	PEPTIDEA	0.95	0.002		98000
run_01	PEPTIDEB	0.91	0.010		43000
run_03	PEPTIDEC	0.88	0.015		61000
```

Wide peptide-by-sample matrices are not accepted directly by `--unit-aware`; convert them to long format first, with one sample/run ID column and one intensity column.

The metadata table is optional. If omitted, each sample is its own `analysis_unit_id`. If provided, it should contain columns like:

```text
sample_id	analysis_unit_id
run_01	donor_A
run_02	donor_A
run_03	donor_B
```

For each analysis unit `u`, unit-level unique evidence uses the selected unique p-value mode. The default `empirical-background` mode estimates a weak-genome background separately within each analysis unit. It uses the same background-excess formula as pooled scoring, but with conservative per-unit auto-exclusion defaults: initial 3%, minimum 0%, maximum 15%, and candidate `q <= 0.20`. This avoids reusing a cohort-level background for per-sample or per-person calls.

With `hypergeometric-opportunity`, unit-level unique evidence uses the hypergeometric model:

```text
X_g,u ~ Hypergeometric(A_total, A_g, S_u)
p_unique_g,u = P(X_g,u >= U_g,u)
```

where `U_g,u` is the observed genome-unique peptide count for genome `g` in unit `u`, `S_u` is the observed genome-unique peptide pool size in unit `u`, `A_g` is the theoretical unique peptide opportunity for genome `g`, and `A_total` is the total theoretical unique peptide opportunity across the analyzed genome panel. BH correction is applied separately within each analysis unit.

Pooled multi-sample scoring answers a cohort-level pooled support question and should not be interpreted as per-sample genome presence. Unit-aware scoring is recommended when the input contains multiple samples or repeated runs.

In the current implementation, peptide presence within an analysis unit is defined as the union of sample-level peptide presence across samples assigned to that unit. Unit-level p-values combine per-unit shared knockoff evidence and the selected per-unit unique evidence with Fisher's method, then apply BH correction separately within each analysis unit.

When unit-aware `empirical-background` is selected, MetaUmbra also writes `unit_empirical_background_calibration.tsv` under `<stem>_artifacts/unit_aware/`. This table records each analysis unit's empirical-background iteration trace, configured threshold quantile, final exclusion fraction, iteration count, active matched genome count, and small-unit warnings. Units with fewer than 100 active matched genomes use no top-genome exclusion and one opportunity bin.

For DIA-NN long/parquet input, `Precursor.Quantity` is recommended for peptide presence filtering. `Precursor.Normalised` can be selected for normalized intensity workflows, but it is not the recommended default for presence/absence detection.

## Input formats

### Protein FASTA input

Use one protein FASTA file per genome. MetaUmbra expects protein sequences, not nucleotide sequences.

Recommended naming:

```text
data/genome_fastas/
  Genome_001.faa
  Genome_002.faa
  Genome_003.faa
```

The filename stem becomes the genome ID after digestion and scoring. For example:

```text
Genome_001.faa -> Genome_001.tsv -> genome_id = Genome_001
```

By default, FASTA headers are shortened at the first space. Use `--full-header` if the full header is required in the digest output.

### Observed peptide table

The observed peptide table must be a tab-separated file. The default expected columns are:

| Column | Required | Description |
| --- | --- | --- |
| `Sequence` | yes | Observed peptide sequence. |
| `Evidence` | no | Peptide-level score. If present, values are normalized to `[0, 1]` and the maximum score per peptide is used. If absent, all peptides are scored as `1`. |
| `Q.Value` | no | Peptide-level error or q-value used for filtering. Rows with values greater than `--peptide-error-cutoff` are removed. |
| `Reverse` | no | Decoy flag column. Rows with `Reverse == "+"` are removed by default. |

Example:

```text
Run	Sequence	Evidence	Q.Value	Reverse
sample_01	PEPTIDER	0.93	0.002	
sample_01	ACDEFGHIK	0.81	0.018	
sample_01	DECOYPEP	0.70	0.001	+
```

If your column names differ, pass matching `--peptide-seq-col`, `--peptide-score-col`, `--peptide-error-col`, and `--peptide-decoy-flag-col` values.

### Genome digest table

Genome digest TSV files are produced by `metaumbra digest`, but user-provided digest tables can also be used if they contain a `Peptide` column:

```text
Protein	Peptide
protein_001	PEPTIDER
protein_002	ACDEFGHIK
```

If a digest file has no `Peptide` column, MetaUmbra tries to use the first column as the peptide column, but this fallback requires peptide-like amino-acid strings. Supplying an explicit `Peptide` column is recommended.

### Genome lineage table

A lineage table is optional and must be a tab-separated file. You choose the genome ID and lineage columns through CLI options or GUI fields.

Example:

```text
genome_id	Lineage
Genome_001	k__Bacteria;p__Firmicutes;g__Example
Genome_002	k__Bacteria;p__Proteobacteria;g__Example
```

Genome IDs should match digest TSV filename stems.

## Output files

### Main result TSV

The default scoring output is a concise TSV table with one row per genome.

| Column | Description |
| --- | --- |
| `genome_id` | Candidate genome identifier, derived from the digest TSV filename stem. |
| `Lineage` | Optional lineage annotation, included only when a lineage table is provided. |
| `evidence_rank` | Rank by lexicographic evidence before final q-value sorting. |
| `presence_rank` | Rank after sorting by `presence_score`. |
| `num_peptides_matched` | Number of observed peptides matched to the genome digest. |
| `num_peptides_unique` | Number of matched peptides unique to that genome among the analyzed genome set. |
| `unique_empirical_excess_count` | Observed unique count above the empirical-background threshold. Included in concise output for `empirical-background`. |
| `theoretical_unique_peptides` | Theoretical peptides unique to this genome among the analyzed genome set. Included in concise output for `hypergeometric-opportunity`. |
| `pvalue_shared` | Shared-peptide knockoff p-value. |
| `pvalue_unique` | Unique-evidence p-value after applying the configured mode. |
| `pvalue` | Genome-level presence p-value in the concise output table. |
| `qvalue` | BH-adjusted genome-level q-value in the concise output table. |
| `presence_score` | Ranking score used to order genome presence calls. |
| `pass_q_0_01` | `true` when `qvalue <= 0.01`. |
| `pass_q_0_05` | `true` when `qvalue <= 0.05`. |

The concise output uses `pvalue` and `qvalue`. When `--return-full-table` is enabled for non-unit-aware scoring, the output retains the full internal table, including internal columns such as `p_presence`, `q_presence`, `weighted_evidence`, and `weighted_evidence_shared`. In unit-aware mode, the requested output remains concise; use `--export-diagnostics` or unit-aware `--return-full-table` for the full unit-level audit table.

### Unit-aware output TSVs

When `--unit-aware` is enabled, the requested `--output` path contains the main unit-level genome presence table. MetaUmbra also writes the primary unit-aware outputs:

```text
<requested output TSV>
<stem>_cohort_genome_summary.tsv
<stem>_artifacts/
  unit_aware/
    <stem>_sample_unit_mapping.tsv
    unit_call_counts.tsv
    unit_specific_genome_list_q005.tsv
    unit_specific_genome_list_q001.tsv
```

- `<requested output TSV>`: one row per `analysis_unit_id` x `genome_id`, including unit-specific `qvalue` and `pass_q_0_01` / `pass_q_0_05` flags.
- `<stem>_cohort_genome_summary.tsv`: one row per genome, summarizing recurrence across units.
- `<stem>_artifacts/unit_aware/<stem>_sample_unit_mapping.tsv`: final sample-to-analysis-unit mapping used for the run.
- `unit_call_counts.tsv`: number of significant genomes per unit.
- `unit_specific_genome_list_q005.tsv`: default balanced long-format unit-specific genome-list output for downstream workflows.
- `unit_specific_genome_list_q001.tsv`: stricter high-specificity long-format alternative.

Both unit-specific genome-list files use this schema:

```text
analysis_unit_id
genome_id
Lineage
presence_rank
qvalue
```

For downstream peptide, protein, taxonomic, or OTF annotation workflows, prefer the tool-agnostic `unit_specific_genome_list_q005.tsv` list unless a stricter high-specificity list is required. Join `<stem>_artifacts/unit_aware/<stem>_sample_unit_mapping.tsv` to `unit_specific_genome_list_q005.tsv` or `unit_specific_genome_list_q001.tsv` when sample columns need to inherit their analysis unit's inferred genome list.

The unit-level table contains one row per `analysis_unit_id` and genome. Its default schema is:

```text
analysis_unit_id
genome_id
Lineage
presence_rank
qvalue
pvalue
pass_q_0_01
pass_q_0_05
num_peptides_unique
unique_empirical_excess_count
num_peptides_matched
matched_peptide_count_shared
pvalue_unique
pvalue_shared
presence_score
n_samples_in_unit
```

`Lineage` is included only when lineage annotations are available.

The cohort summary answers how often each genome is supported across units. Its default schema is:

```text
genome_id
Lineage
n_units_tested
n_units_q_le_0_05
fraction_units_q_le_0_05
n_units_q_le_0_01
fraction_units_q_le_0_01
best_qvalue
median_qvalue
best_presence_rank
total_unique_peptides_across_units
total_matched_peptides_across_units
```

`Lineage` is included only when lineage annotations are available.

### Diagnostic artifacts

At the start of each scoring run, MetaUmbra creates an artifact directory next to the main output and writes run provenance there. This happens before peptide or genome inputs are read, so failed runs can still leave the parameters and log needed for debugging.

```text
results/
  genome_presence.tsv
  genome_presence_artifacts/
    run_parameters.json
    run.log
    run_status.json
```

Use `--export-diagnostics` to add heavier audit, debug, and figure-generation outputs:

```text
results/
  genome_presence_artifacts/
    run_summary.json
    full_internal_metrics.tsv
    pooled_genome_presence.tsv  # unit-aware only
    knockoff_pools.tsv
    degeneracy_hist.tsv
    p_shared_hist.tsv
    q_calling_curve.tsv
    shared_stratum_counts.tsv
    topN_peptide_contrib.tsv  # only when export_peptide_contrib_topN > 0
    unit_aware/
      unit_empirical_background_calibration.tsv
      unit_genome_presence_full.tsv
      unit_threshold_summary.tsv
      unit_q001_genomes.tsv
      unit_q005_genomes.tsv
      genome_union_q001.tsv
      genome_union_q005.tsv
      genome_by_unit_q001_matrix.tsv
      genome_by_unit_q005_matrix.tsv
      genome_by_unit_qvalue_matrix.tsv
```

The exact set of diagnostic tables depends on available runtime data. Diagnostics are off by default; `run_parameters.json`, `run.log`, and `run_status.json` are always written.

Common artifacts:

| File | Description |
| --- | --- |
| `run_parameters.json` | Scoring configuration captured at run start, including CLI/GUI parameters, output path, CPU model, logical CPU count, total memory, platform/architecture, Python version, and MetaUmbra version. |
| `run.log` | Runtime log stream written alongside the GUI/CLI log output. |
| `run_status.json` | Completion status and final result summary for successful scoring runs. |
| `run_summary.json` | Runtime settings, input summary, platform details, timing, and call counts. |
| `full_internal_metrics.tsv` | Complete internal metrics table used to produce the concise output. |
| `pooled_genome_presence.tsv` | Supplementary pooled peptide-set result for unit-aware runs. It is not the union of unit-level calls. |
| `knockoff_pools.tsv` | Knockoff pool diagnostics for shared peptide evidence. |
| `degeneracy_hist.tsv` | Distribution of peptide degeneracy across the analyzed genome set. |
| `p_shared_hist.tsv` | Histogram of shared-evidence knockoff p-values. |
| `q_calling_curve.tsv` | Number of called genomes over several q-value thresholds. |
| `shared_stratum_counts.tsv` | Per-genome shared peptide stratum counts. |

Diagnostic unit-aware tables are redundant with the primary outputs and are meant for audit, QC, or figure generation. `unit_q001_genomes.tsv` and `unit_q005_genomes.tsv` are filtered subsets of the main unit-level table. `genome_union_q001.tsv` and `genome_union_q005.tsv` can be recovered from `<stem>_cohort_genome_summary.tsv` using `n_units_q_le_0_01` or `n_units_q_le_0_05`. `genome_by_unit_*_matrix.tsv` files are QC/visualization pivots, not preferred downstream inputs.

`unit_aware/unit_genome_presence_full.tsv` is the full unit-level audit table written with `--export-diagnostics` or unit-aware `--return-full-table`. It preserves internal columns that are intentionally omitted from the concise default requested output, including columns such as `unique_effective_count`, `theoretical_unique_peptides`, `unit_presence_rule`, and `unit_shared_mode`.

### Matched-peptide cache

Cache files are performance infrastructure. CLI runs write them only when `--save-cache` is enabled; the GUI "Save matched-peptide cache" option is enabled by default. When cache saving is enabled, MetaUmbra writes a pickle file containing matched peptide sets and theoretical peptide counts. If `--cache-path` is not provided, the default cache is:

```text
<output_directory>/<output_stem>_artifacts/matched_peptides.pkl
```

Use `--use-cache-if-exists` for repeated analyses with the same observed peptide table and genome digest directories. When reuse is requested, startup artifact cleanup preserves the default `<stem>_artifacts/matched_peptides.pkl` so it can be loaded before genome scanning. If `--cache-path` is supplied, MetaUmbra saves or reuses that path only according to `--save-cache` and `--use-cache-if-exists`, and artifact cleanup does not delete that explicit cache path. The cache can still be combined with `--selected-genome-ids` and `--exclude-genome-ids`; filters are applied after loading the cache.

`hypergeometric-opportunity` scoring can reuse or write a theoretical opportunity cache only when `--theoretical-opportunity-cache` is supplied. `empirical-background` does not build this cache by default, removes stale default theoretical opportunity caches from the current artifact directory, and uses `total_peptide_count` for opportunity binning.

Use `--rebuild-theoretical-opportunity-cache` after changing genome digest files or when you want to force a fresh theoretical opportunity scan. New caches include digest file fingerprints so MetaUmbra can detect digest file changes before reusing the cache. Legacy caches without fingerprints are still accepted when genome IDs match, but rebuilding once enables stricter validation. Theoretical opportunity building can shard theoretical peptides across worker processes with `--num-workers-for-theoretical-opportunity`; if you do not set it, MetaUmbra reuses `--num-workers`.

## Interpreting results

MetaUmbra combines unique peptide support with weighted shared peptide evidence:

- Unique matched peptides are strong genome-specific evidence.
- Shared peptides are weighted by degeneracy. A peptide found in many candidate genomes contributes less to each genome than a peptide found in only a few genomes.
- Shared-evidence p-values are estimated using a peptide-space knockoff procedure.
- Unique-evidence p-values use a peptide-depth adjusted null model by default.
- `qvalue` is the Benjamini-Hochberg adjusted genome-level presence q-value across analyzed genomes.

For default `empirical-background` unique evidence, MetaUmbra estimates a sample-specific weak-genome background for genome-unique peptide counts. It bins weak-background genomes by `total_peptide_count`, estimates the 95th percentile weak-background unique count in each bin, and accumulates only excess unique evidence:

```text
U_background = percentile_95(U_bg in same total_peptide_count bin)
U_excess = max(0, U_g - U_background)
p_unique = alpha ^ U_excess, when U_excess > 0; otherwise p_unique = 1
```

By default, pooled background exclusion is adaptive. MetaUmbra starts by excluding the top 10% evidence-ranked genomes from the weak-background pool, computes preliminary genome-level q-values, estimates the fraction of candidate genomes with `q <= 0.20`, clamps that fraction to 10-30%, and rebuilds the final weak-background threshold. This matters for pooled or high-depth cohorts where many true present genomes can otherwise contaminate the weak-background pool and inflate `U_background`. In unit-aware mode, MetaUmbra fits this empirical background independently inside each analysis unit and uses more conservative per-unit exclusion defaults: 3% initial, 0% minimum, and 15% maximum.

This is analogous in spirit to the shared peptide knockoff, but for genome-unique evidence: the shared knockoff calibrates shared weighted evidence, while `empirical-background` calibrates unique peptide counts against weak-background genomes at the same sample depth and opportunity scale. The direct empirical ECDF tail is retained as `p_unique_empirical_tail` for diagnostics, but it is not used as `pvalue_unique` because its resolution is too coarse for genome-level BH correction across thousands of genomes. `empirical-background` does not build the theoretical opportunity cache by default. With `--export-diagnostics`, `run_summary.json` records `unique_empirical_background_opportunity_source = total_peptide_count`, along with the initial and final background exclusion fractions, candidate q threshold, iteration count, threshold quantile, and alpha.

The threshold quantile is controlled by `--unique-empirical-background-threshold-quantile`. The same configured value is used for pooled `empirical-background` scoring and per-unit `empirical-background` scoring. In unit-aware empirical-background runs, the per-unit calibration table records the value in `unit_empirical_background_threshold_quantile`.

For `hypergeometric-opportunity` unique evidence, MetaUmbra compares observed genome-unique peptides against genome-specific theoretical unique peptide opportunity:

```text
X_g ~ Hypergeometric(A_total, A_g, S)
p_unique = P(X_g >= U_g)
```

Definitions:

| Symbol | Meaning |
| --- | --- |
| `A_total` | Total theoretical genome-unique peptides across genomes. |
| `A_g` | Theoretical genome-unique peptides for genome `g`. |
| `S` | Total observed genome-unique peptides across genomes. |
| `U_g` | Observed genome-unique matched peptides for genome `g`. |

By default, unique peptide evidence contributes to the combined genome presence p-value only when at least three genome-unique peptides are observed. Genomes below this threshold can still receive support from shared peptide evidence through the shared-peptide knockoff model.

The legacy upper-bound mode, `p_unique = alpha^U`, is retained for sensitivity analysis. The `peptide-column` mode should only be used when the selected peptide error column represents a peptide-level posterior error probability or a comparable per-peptide error estimate.

Practical interpretation:

- Use `qvalue <= 0.05` as a broad presence threshold and `qvalue <= 0.01` as a stricter presence threshold.
- Prefer genomes with more unique peptides when closely related genomes have similar q-values.
- Inspect `pvalue_shared`, `pvalue_unique`, `expected_unique_null`, and `unique_depth_fold` for ambiguous calls driven mostly by shared peptides or by unusually deep unique evidence.
- Use the lineage table to summarize calls at taxonomic or custom group levels outside MetaUmbra.

The analyzed genome set matters. Peptide uniqueness and degeneracy are computed across the target genomes included in the run. Adding or removing closely related genomes can change `num_peptides_unique`, shared evidence, and q-values.

## Performance and reproducibility

### Worker counts

`metaumbra digest` and `metaumbra score` both support parallel execution.

For digestion:

- Directory mode parallelizes across FASTA files.
- Single-file mode uses streaming serial digestion for smaller files and parallel batch digestion for large files.

For scoring:

- `--num-workers` controls genome scanning parallelism.
- `--num-workers-for-theoretical-opportunity` controls only the `hypergeometric-opportunity` theoretical opportunity sharding and reduction step.
- The default is `max(1, cpu_count - 1)`.
- If `--num-workers-for-theoretical-opportunity` is omitted, MetaUmbra reuses `--num-workers`.
- On Windows, keep worker counts at 60 or fewer. Very high values can exceed the `ProcessPoolExecutor` worker limit.

### Repeated runs

For repeated scoring over the same observed peptides and genome digest files:

```bash
metaumbra score \
  --peptide-table results/observed_peptides.tsv \
  --genome-digest-dirs results/genome_digests \
  --output results/genome_presence.tsv \
  --save-cache \
  --cache-path results/genome_presence_artifacts/matched_peptides.pkl \
  --use-cache-if-exists
```

Use separate cache files when changing:

- Observed peptide table
- Peptide sequence column or peptide filtering settings
- Genome digest directories
- Digest files

### Reproducibility

The scoring workflow uses deterministic sorting and a fixed default random seed for knockoff calculations. Keep these stable for reproducible runs:

- Same MetaUmbra version
- Same input peptide table and digest files
- Same selected and excluded genome IDs
- Same worker-independent scoring options
- Same knockoff settings

The GUI stores these settings in saved configuration files.

## Troubleshooting

### `Missing peptide column 'Sequence' in peptide file`

Your observed peptide table does not contain the default sequence column. Pass the correct column name:

```bash
metaumbra score \
  --peptide-table observed.tsv \
  --genome-digest-dirs genome_digests \
  --output genome_presence.tsv \
  --peptide-seq-col "Stripped.Sequence"
```

### `Score column 'Evidence' not found; setting all scores=1`

This is a warning, not a fatal error. MetaUmbra will score each unique observed peptide equally. To use scores, provide the correct score column:

```bash
--peptide-score-col score
```

### `Error column 'Q.Value' not found; skipping peptide-level error filtering`

MetaUmbra could not find the requested peptide error column. Provide the correct column or leave filtering disabled by passing an empty error column in the GUI. On the command line, use a real column name with `--peptide-error-col`.

### Windows `ProcessPoolExecutor` worker limit

On Windows, `ProcessPoolExecutor` cannot start very high numbers of worker processes. If scoring fails when using many workers, reduce `--num-workers` to 60 or fewer. For large genome panels, 8-32 workers is usually a safer starting range.

### `No valid genome folders found`

At least one `--genome-digest-dirs` path must exist and contain digest TSV files. When passing multiple directories in one argument, separate them with commas or semicolons.

### `No genome peptide TSV files found`

The digest directory exists but contains no `.tsv` files. Run `metaumbra digest` first, or check that the path points to the directory containing per-genome peptide TSV files.

### `No genome peptide TSV files matched the Only Run Genome IDs list`

Values passed with `--selected-genome-ids` must match digest TSV filename stems exactly. For example, `Genome_001.tsv` is selected with:

```bash
--selected-genome-ids Genome_001
```

### Parquet extraction says a column is missing

Inspect the parquet schema in the original search output and pass the actual column names with repeated `--input-column` and `--output-column` options. The number of input and output columns must match.

### Output already exists during parquet extraction

Use `--force` to overwrite:

```bash
metaumbra extract-parquet \
  --input data/report.parquet \
  --output results/observed_peptides.tsv \
  --force
```
