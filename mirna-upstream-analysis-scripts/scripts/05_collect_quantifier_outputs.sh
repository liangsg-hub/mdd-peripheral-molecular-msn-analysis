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

SRC_DIR="$QUANTIFIER_DIR/samples"
DST_DIR="$QUANTIFIER_LIGHT_DIR"
mkdir -p "$DST_DIR"

sample_dirs=("$SRC_DIR"/*)
if [ ${#sample_dirs[@]} -eq 0 ]; then
  echo "No quantifier sample directories found in $SRC_DIR"
  exit 1
fi

for d in "${sample_dirs[@]}"; do
  [ -d "$d" ] || continue
  sample=$(basename "$d")
  out_dir="$DST_DIR/$sample"
  result_csv="$d/miRNAs_expressed_all_samples_${sample}.csv"

  if [ ! -s "$result_csv" ]; then
    echo "Missing result file: $result_csv"
    continue
  fi

  mkdir -p "$out_dir"
  cp -f "$result_csv" "$out_dir/"
done

echo "Quantifier result files copied to $DST_DIR"
