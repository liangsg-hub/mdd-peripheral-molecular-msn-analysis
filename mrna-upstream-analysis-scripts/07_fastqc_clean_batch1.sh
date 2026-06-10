#!/usr/bin/env bash
set -euo pipefail

# Run FastQC on cleaned FASTQ files for batch 1

CLEAN_DIR="./results/02_fastp/batch1"
OUT_DIR="./results/03_fastqc_clean/batch1"
THREADS=8

mkdir -p "${OUT_DIR}"

fastqc -t "${THREADS}" -o "${OUT_DIR}" "${CLEAN_DIR}"/*.fastq.gz

echo "Done: clean FastQC for batch1"
