#!/usr/bin/env bash
set -euo pipefail

# Gene-level quantification for batch 1
# Batch 1 is treated as unstranded

GTF="./ref/Homo_sapiens.GRCh38.113.chr.gtf"
BAM_DIR="./results/06_align_batch1"
OUT_DIR="./results/10_counts_batch1"
THREADS=16

mkdir -p "${OUT_DIR}"

featureCounts \
    -a "${GTF}" \
    -o "${OUT_DIR}/gene_counts.txt" \
    -p --countReadPairs -B \
    -t exon \
    -g gene_id \
    -s 0 \
    --primary \
    -Q 10 \
    -T "${THREADS}" \
    "${BAM_DIR}"/*.bam

echo "Done: featureCounts for batch1"
