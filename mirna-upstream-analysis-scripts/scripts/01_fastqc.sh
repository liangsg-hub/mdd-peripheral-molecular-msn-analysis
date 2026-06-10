#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONFIG_FILE=${1:-"${SCRIPT_DIR}/../config.sh"}

if [ ! -s "$CONFIG_FILE" ]; then
  echo "Config file not found: $CONFIG_FILE"
  echo "Copy config.example.sh to config.sh and update paths."
  exit 1
fi

source "$CONFIG_FILE"

mkdir -p "$FASTQC_DIR"

command -v fastqc >/dev/null || { echo "fastqc not found in PATH"; exit 1; }
command -v multiqc >/dev/null || { echo "multiqc not found in PATH"; exit 1; }

files=("$RAW_DIR"/*_R1.fastq.gz)
if [ ${#files[@]} -eq 0 ]; then
  echo "No R1 FASTQ files found in $RAW_DIR"
  exit 1
fi

fastqc -t "$THREADS" -o "$FASTQC_DIR" "${files[@]}"
multiqc "$FASTQC_DIR" -o "$FASTQC_DIR/multiqc_report"

echo "FastQC and MultiQC finished."
