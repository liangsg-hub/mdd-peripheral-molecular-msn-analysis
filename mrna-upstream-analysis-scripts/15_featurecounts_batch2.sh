#!/usr/bin/env bash
set -euo pipefail

# Gene-level quantification for batch 2
# Batch 2 is treated as reverse-stranded

GTF="./ref/Homo_sapiens.GRCh38.113.chr.gtf"
BAM_DIR="./results/07_align_batch2"
OUT_DIR="./results/11_counts_batch2"
THREADS=16

mkdir -p "${OUT_DIR}"

featureCounts \
    -a "${GTF}" \
    -o "${OUT_DIR}/gene_counts.txt" \
    -p --countReadPairs -B \
    -t exon \
    -g gene_id \
    -s 2 \
    --primary \
    -Q 10 \
    -T "${THREADS}" \
    "${BAM_DIR}"/*.bam

echo "Done: featureCounts for batch2"
