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

LOG_DIR="$CLEAN_DIR/logs"
mkdir -p "$CLEAN_DIR" "$LOG_DIR"

command -v fastp >/dev/null || { echo "fastp not found in PATH"; exit 1; }
command -v multiqc >/dev/null || { echo "multiqc not found in PATH"; exit 1; }

files=("$RAW_DIR"/*_R1.fastq.gz)
if [ ${#files[@]} -eq 0 ]; then
  echo "No R1 FASTQ files found in $RAW_DIR"
  exit 1
fi

for f in "${files[@]}"; do
  bn=$(basename "$f" .fastq.gz)
  echo "fastp: $bn"

  fastp \
    -i "$f" \
    -o "$CLEAN_DIR/${bn}.clean.fastq.gz" \
    --length_required 18 \
    --length_limit 30 \
    -q 20 \
    -u 30 \
    -n 0 \
    -w "$THREADS" \
    --disable_adapter_trimming \
    --trim_poly_g \
    --trim_poly_x \
    -h "$CLEAN_DIR/${bn}.fastp.html" \
    -j "$CLEAN_DIR/${bn}.fastp.json" \
    > "$LOG_DIR/${bn}.fastp.stdout.log" \
    2> "$LOG_DIR/${bn}.fastp.stderr.log"
done

multiqc "$CLEAN_DIR" -o "$CLEAN_DIR/multiqc_report"

echo "fastp filtering and MultiQC finished."
