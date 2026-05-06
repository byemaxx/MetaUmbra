# MetaUmbra
[![MetaUmbra](https://raw.githubusercontent.com/byemaxx/MetaUmbra/main/src/metaumbra/assets/baner.png)](https://github.com/byemaxx/MetaUmbra)


## Genome-level presence inference from metaproteomic peptides

MetaUmbra performs genome-level presence inference from metaproteomic peptide lists. It combines unique peptide support with weighted shared peptide evidence to identify statistically supported microbial genomes and generate interpretable presence rankings.

## Main features

- Evaluate candidate genome support from metaproteomic peptide tables
- Build genome-specific theoretical peptide references from protein FASTA files
- Support user-defined genome collections, including isolate genomes, strain panels, and MAG catalogs
- Use both unique and shared peptide evidence for genome presence inference
- Report genome-level p-values, BH-adjusted q-values, and presence scores
- Provide GUI, command-line, and Python workflow support
- Support peptide tables from common metaproteomics workflows such as DIA-NN and MaxQuant

## Workflow overview
[![MetaUmbra workflow](https://raw.githubusercontent.com/byemaxx/MetaUmbra/main/src/metaumbra/assets/workflow.png)](https://github.com/byemaxx/MetaUmbra)


## Installation

MetaUmbra requires Python 3.10 or newer.

```bash
# Install with all features (GUI, parquet support)
pip install metaumbra[all]
```
or
```bash
# Install with core features only
pip install metaumbra
```

## Usage

MetaUmbra can be used through either the graphical interface or the command line.

For a detailed walkthrough, including input formats, CLI examples, output interpretation, and troubleshooting, see the [MetaUmbra Usage Guide](docs/usage.md).

### Graphical interface

```bash
metaumbra-gui
```

The GUI supports FASTA digestion, peptide table loading, genome presence scoring, and result export.

### Command line

MetaUmbra provides separate commands for the main workflow steps:

```bash
metaumbra digest --help
metaumbra score --help
metaumbra extract-parquet --help
```

A typical workflow is:

```bash
metaumbra digest ...
metaumbra score ...
```

Use `metaumbra extract-parquet ...` to convert DIA-NN parquet reports to peptide TSV files before scoring.

## Input

MetaUmbra requires:

- Protein FASTA files, with one FASTA file per genome
- An observed peptide table containing peptide sequences

Optional inputs include peptide scores, peptide-level error values, decoy flags, and genome lineage annotations.

## Output

The main output is a TSV table containing genome-level evidence and significance values.

Key output columns include:

| Column | Description |
| --- | --- |
| `genome_id` | Candidate genome identifier |
| `num_peptides_matched` | Number of observed peptides matched to the genome |
| `num_peptides_unique` | Number of matched peptides unique to the genome |
| `shared_fraction` | Fraction of matched peptides that are shared with other genomes |
| `mean_degeneracy` | Mean number of genomes containing the matched peptides |
| `pvalue` | Genome-level p-value |
| `qvalue` | BH-adjusted genome-level q-value |
| `presence_score` | Ranking score based on q-value |

## Citation

If you use MetaUmbra, please cite:

> Wu Q, Ning Z, Zhang A, Cheng K, Figeys D. MetaUmbra: Statistically Controlled Genome-Level Presence Inference from Metaproteomic Peptides.[J]. [bioRxiv, 2026.04.29.721689.](https://doi.org/10.64898/2026.04.29.721689)

A formal citation will be added after publication.

## Contact

For questions or issues, please use the GitHub issue tracker or contact the corresponding author listed in the associated manuscript.
