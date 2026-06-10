# Bulk mRNA-seq processing pipeline (GRCh38 Ensembl release 113)

This repository contains a stepwise shell-based pipeline for bulk mRNA-seq quality control, alignment, post-alignment QC, and gene-level quantification.

The workflow was designed for paired-end human RNA-seq data and uses a unified reference framework based on **GRCh38 primary assembly** and **Ensembl release 113**.

## Overview

The pipeline includes the following steps:

1. Generate splice site annotation from GTF
2. Generate BED12 annotation for RSeQC
3. Raw read QC with FastQC
4. Adapter trimming and quality filtering with fastp
5. QC aggregation with MultiQC
6. Read alignment with HISAT2
7. BAM sorting, indexing, and alignment statistics with samtools
8. Post-alignment QC with RSeQC
9. Gene-level quantification with featureCounts
10. Merging count matrices across batches

## Reference files

The pipeline uses a unified reference set:

- Genome: `GRCh38 primary assembly`
- Gene annotation: `Ensembl release 113`
- Splice sites: extracted from `Homo_sapiens.GRCh38.113.chr.gtf`
- BED12 for RSeQC: generated from the same annotation source

## Input data organization

Recommended structure:

```text
project/
├── scripts/
├── ref/
├── raw_data/
│   ├── batch1/
│   └── batch2/
├── results/
└── logs/
```

Paired-end FASTQ files should follow a naming scheme such as:

```text
sample1_R1.fastq.gz
sample1_R2.fastq.gz
```

## Library type

This pipeline assumes two sequencing batches with different strand settings:

- **batch1**: unstranded
- **batch2**: reverse-stranded

Accordingly:

- batch1 is quantified with `featureCounts -s 0`
- batch2 is quantified with `featureCounts -s 2`

For HISAT2 alignment:

- batch1 is aligned **without** `--rna-strandness`
- batch2 is aligned with `--rna-strandness RF`

## Software

Suggested software versions:

- FastQC v0.11.9
- fastp v0.23.2
- MultiQC v1.19
- HISAT2 v2.2.1
- samtools v1.21
- RSeQC v4.0
- Subread/featureCounts v2.0.8

## Running the pipeline

Run each step separately:

```bash
bash scripts/01_prepare_splice_sites.sh
bash scripts/02_prepare_bed12.sh
bash scripts/03_fastqc_raw_batch1.sh
bash scripts/04_fastqc_raw_batch2.sh
bash scripts/05_fastp_batch1.sh
bash scripts/06_fastp_batch2.sh
bash scripts/07_fastqc_clean_batch1.sh
bash scripts/08_fastqc_clean_batch2.sh
bash scripts/09_multiqc.sh
bash scripts/10_hisat2_align_batch1.sh
bash scripts/11_hisat2_align_batch2.sh
bash scripts/12_rseqc_batch1.sh
bash scripts/13_rseqc_batch2.sh
bash scripts/14_featurecounts_batch1.sh
bash scripts/15_featurecounts_batch2.sh
bash scripts/16_merge_counts_matrix.sh
```

## Quantification strategy

Gene-level quantification is performed with `featureCounts` using:

- exon-level summarization
- `gene_id` as the grouping key
- paired-end fragment counting with `--countReadPairs`
- primary alignments only
- minimum mapping quality of 10

## Reproducibility

All analysis steps are implemented as one-shell-script-per-step modules with explicit inputs and outputs. To facilitate public release and manuscript submission, local absolute paths and user-specific system information were removed from the shared scripts. The workflow can be reproduced by organizing raw data and reference files according to the documented directory structure and running each shell script sequentially.

## Notes

- This repository is intended as a transparent and reproducible processing workflow for manuscript submission and public release.
- Absolute local file paths have been removed to protect personal system information.
- Users should adapt relative paths and thread settings according to their own environment.

## Citation

If you use this workflow, please cite the corresponding tools:

- FastQC
- fastp
- MultiQC
- HISAT2
- samtools
- RSeQC
- featureCounts
