# TaxaSeeker

TaxaSeeker is a tool for inferring genome/taxa presence from an observed peptide list in Metaproteomics. It takes as input a list of observed peptides (e.g. from MaxQuant, DIANN) and a database of theoretical digest peptides for a large genome collection (e.g. UHGP).

The core idea is to score each genome by how well its *in-silico digest peptides* explain the observed peptides, while accounting for shared peptides. The main scorer implements a **peptide-space knockoff null** to estimate per-genome existence p-values/q-values without requiring a second database search.

