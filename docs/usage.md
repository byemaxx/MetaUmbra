# MetaUmbra analysis-unit workflow

MetaUmbra always scores genomes per `analysis_unit`. The unit definition changes; the scoring algorithm and downstream contract do not.

```bash
# One cohort-wide unit named __global__
metaumbra score --unit-mode all-samples --peptide-table report.parquet --genome-digest-dirs digests --output results

# One unit for every DIA-NN Run
metaumbra score --unit-mode per-sample --peptide-table report.parquet --genome-digest-dirs digests --output results

# Metadata groups
metaumbra score --unit-mode metadata --metadata-table metadata.tsv \
  --metadata-sample-id-col sample_id --metadata-analysis-unit-col analysis_unit_id \
  --peptide-table report.parquet --genome-digest-dirs digests --output results
```

For `per-sample` and `metadata` modes, the peptide table must be long-format and include a sample/run column, peptide sequence, and intensity. `all-samples` also accepts peptide-only tables: when sample and intensity columns are absent, MetaUmbra creates one synthetic global sample and treats each peptide row as present. DIA-NN parquet columns are detected from common names; defaults are `Run`, `Stripped.Sequence`/`Sequence`, and `Precursor.Quantity`. Trailing `.raw` is removed consistently from peptide-table and metadata sample IDs before unit mapping.

Metadata mode is strict: each peptide-table sample must occur exactly once in metadata, with a non-empty analysis unit ID. Duplicate or missing assignments are errors.

## Scoring

Sample-level peptide presence is aggregated by union within each analysis unit. Every unit independently receives the same shared-peptide knockoff, unique-evidence model, presence score, p-value, and BH q-value calculation. `all-samples` therefore differs from grouped/per-sample analysis only in the sample-to-unit assignment and number of units; it does not call a separate pooled scoring backend.

## Results

```text
results/
├── genome_selection_manifest.json
├── unit_genome_results.tsv
├── cohort_genome_summary.tsv
├── sample_unit_mapping.tsv
└── artifacts/
    ├── run_summary.json
    ├── run_parameters.json
    ├── run_status.json
    ├── run.log
    └── diagnostics/
```

`unit_genome_results.tsv` always contains `analysis_unit_id`, `genome_id`, `qvalue`, `pvalue`, `presence_rank`, `presence_score`, `pass_q_0_01`, and `pass_q_0_05`. The all-samples unit ID is `__global__`.

`genome_selection_manifest.json` is the only downstream control interface. Its schema is `metaumbra.genome_selection_manifest.v1`; the normative schema is [genome_selection_manifest.v1.schema.json](genome_selection_manifest.v1.schema.json). Detailed scores, lineage, and diagnostics remain in TSV/JSON artifacts.
