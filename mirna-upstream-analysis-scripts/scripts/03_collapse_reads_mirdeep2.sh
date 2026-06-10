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

TMP_DIR="$COLLAPSED_DIR/tmp_fastq"
LOG_DIR="$COLLAPSED_DIR/logs"
mkdir -p "$COLLAPSED_DIR" "$TMP_DIR" "$LOG_DIR"

command -v mapper.pl >/dev/null || { echo "mapper.pl not found in PATH"; exit 1; }
command -v gunzip >/dev/null || { echo "gunzip not found in PATH"; exit 1; }

files=("$CLEAN_DIR"/*_R1.clean.fastq.gz)
if [ ${#files[@]} -eq 0 ]; then
  echo "No cleaned R1 FASTQ files found in $CLEAN_DIR"
  exit 1
fi

for f in "${files[@]}"; do
  sample=$(basename "$f" _R1.clean.fastq.gz)
  echo "mapper.pl: $sample"

  tmp_fq="$TMP_DIR/${sample}_R1.clean.fastq"
  gunzip -c "$f" > "$tmp_fq"

  mapper.pl "$tmp_fq" -e -h -j -m -s "$COLLAPSED_DIR/${sample}.collapsed.fa" \
    > "$LOG_DIR/${sample}.mapper.stdout.log" \
    2> "$LOG_DIR/${sample}.mapper.stderr.log"

  rm -f "$tmp_fq"
done

echo "Read collapsing by miRDeep2 mapper.pl finished."
