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

SAMPLES_DIR="$QUANTIFIER_DIR/samples"
LOG_DIR="$QUANTIFIER_DIR/logs"
mkdir -p "$SAMPLES_DIR" "$LOG_DIR"

command -v quantifier.pl >/dev/null || { echo "quantifier.pl not found in PATH"; exit 1; }
test -s "$MATURE_FA" || { echo "Missing file: $MATURE_FA"; exit 1; }
test -s "$HAIRPIN_FA" || { echo "Missing file: $HAIRPIN_FA"; exit 1; }

files=("$COLLAPSED_DIR"/*.collapsed.fa)
if [ ${#files[@]} -eq 0 ]; then
  echo "No collapsed FASTA files found in $COLLAPSED_DIR"
  exit 1
fi

manifest="$QUANTIFIER_DIR/manifest.tsv"
: > "$manifest"
echo -e "sample\tcollapsed_fa\tsample_out" >> "$manifest"

for f in "${files[@]}"; do
  sample=$(basename "$f" .collapsed.fa)
  sample_out="$SAMPLES_DIR/$sample"
  mkdir -p "$sample_out"

  echo "quantifier.pl: $sample"

  pushd "$sample_out" >/dev/null
  quantifier.pl \
    -p "$HAIRPIN_FA" \
    -m "$MATURE_FA" \
    -r "$f" \
    -t "$SPECIES" \
    -y "$sample" \
    > "$LOG_DIR/${sample}.quantifier.stdout.log" \
    2> "$LOG_DIR/${sample}.quantifier.stderr.log"
  popd >/dev/null

  echo -e "${sample}\t${f}\t${sample_out}" >> "$manifest"
done

echo "miRDeep2 quantifier.pl finished."
echo "Manifest: $manifest"
