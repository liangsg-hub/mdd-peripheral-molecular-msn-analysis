#!/usr/bin/env bash
set -euo pipefail

# Run RSeQC for batch 2 BAM files

BAM_DIR="./results/07_align_batch2"
BED12="./results/05_ref_aux/GRCh38.113.bed12"
OUT_DIR="./results/09_rseqc_batch2"

mkdir -p "${OUT_DIR}"

for BAM in "${BAM_DIR}"/*.bam; do
    SAMPLE=$(basename "${BAM}" .bam)

    infer_experiment.py \
        -r "${BED12}" \
        -i "${BAM}" \
        > "${OUT_DIR}/${SAMPLE}.infer_experiment.txt"

    read_distribution.py \
        -r "${BED12}" \
        -i "${BAM}" \
        > "${OUT_DIR}/${SAMPLE}.read_distribution.txt"

    geneBody_coverage.py \
        -r "${BED12}" \
        -i "${BAM}" \
        -o "${OUT_DIR}/${SAMPLE}.geneBody"
done

echo "Done: RSeQC for batch2"
