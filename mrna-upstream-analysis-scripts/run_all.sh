#!/usr/bin/env bash
set -euo pipefail

# Run all steps in sequence

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
