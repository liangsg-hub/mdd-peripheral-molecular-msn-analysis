#!/usr/bin/env bash
set -euo pipefail

# Adapter trimming and quality filtering for batch 1
# Rules:
# - sliding window size: 4
# - mean quality cutoff: 10
# - minimum read length: 40

RAW_DIR="./raw_data/batch1"
OUT_DIR="./results/02_fastp/batch1"
THREADS=8

mkdir -p "${OUT_DIR}"

for R1 in "${RAW_DIR}"/*_R1.fastq.gz; do
    SAMPLE=$(basename "${R1}" _R1.fastq.gz)
    R2="${RAW_DIR}/${SAMPLE}_R2.fastq.gz"

    fastp \
        -i "${R1}" \
        -I "${R2}" \
        -o "${OUT_DIR}/${SAMPLE}_R1.clean.fastq.gz" \
        -O "${OUT_DIR}/${SAMPLE}_R2.clean.fastq.gz" \
        --cut_right \
        --cut_right_window_size 4 \
        --cut_right_mean_quality 10 \
        --length_required 40 \
        --thread "${THREADS}" \
        --html "${OUT_DIR}/${SAMPLE}.fastp.html" \
        --json "${OUT_DIR}/${SAMPLE}.fastp.json"
done

echo "Done: fastp for batch1"
