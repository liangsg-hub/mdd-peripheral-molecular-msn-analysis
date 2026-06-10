#!/usr/bin/env bash
set -euo pipefail

# Run FastQC on raw FASTQ files for batch 2

RAW_DIR="./raw_data/batch2"
OUT_DIR="./results/01_fastqc_raw/batch2"
THREADS=8

mkdir -p "${OUT_DIR}"

fastqc -t "${THREADS}" -o "${OUT_DIR}" "${RAW_DIR}"/*.fastq.gz

echo "Done: raw FastQC for batch2"
