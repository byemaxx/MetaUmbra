# MetaUmbra
[![MetaUmbra](https://raw.githubusercontent.com/byemaxx/MetaUmbra/main/src/metaumbra/assets/baner.png)](https://github.com/byemaxx/MetaUmbra)

[GitHub Homepage](https://github.com/byemaxx/MetaUmbra)

## Genome-level presence inference from metaproteomic peptides

MetaUmbra converts identified metaproteomic peptides into statistically supported genome presence calls. It evaluates each candidate genome using both unique and shared peptide evidence and reports genome-level p-values, BH-adjusted q-values, and presence scores.

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
pip install ".[all]"
```

## Usage

MetaUmbra can be used through either the graphical interface or the command line.

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
| `weighted_evidence` | Total degeneracy-weighted peptide evidence |
| `weighted_evidence_shared` | Weighted evidence from shared peptides |
| `p_presence` | Genome-level p-value |
| `q_presence` | BH-adjusted genome-level q-value |
| `presence_score` | Ranking score based on q-value |

## Citation

If you use MetaUmbra, please cite:

> Wu Q, Ning Z, Zhang A, Cheng K, Figeys D. MetaUmbra: Statistically Controlled Genome-Level Presence Inference from Metaproteomic Peptides.

A formal citation will be added after publication.

## Contact

For questions or issues, please use the GitHub issue tracker or contact the corresponding author listed in the associated manuscript.
