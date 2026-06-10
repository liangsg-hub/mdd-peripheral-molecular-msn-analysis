#!/usr/bin/env bash
set -euo pipefail

# Aggregate QC reports from FastQC and fastp

INPUT_DIR="./results"
OUT_DIR="./results/04_multiqc"

mkdir -p "${OUT_DIR}"

multiqc "${INPUT_DIR}" -o "${OUT_DIR}"

echo "Done: MultiQC report generated"
