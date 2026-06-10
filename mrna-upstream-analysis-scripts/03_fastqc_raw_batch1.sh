#!/usr/bin/env bash
set -euo pipefail

# Run FastQC on raw FASTQ files for batch 1

RAW_DIR="./raw_data/batch1"
OUT_DIR="./results/01_fastqc_raw/batch1"
THREADS=8

mkdir -p "${OUT_DIR}"

fastqc -t "${THREADS}" -o "${OUT_DIR}" "${RAW_DIR}"/*.fastq.gz

echo "Done: raw FastQC for batch1"
