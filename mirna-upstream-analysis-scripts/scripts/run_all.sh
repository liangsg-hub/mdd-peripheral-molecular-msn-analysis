#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONFIG_FILE=${1:-"${SCRIPT_DIR}/../config.sh"}

bash "$SCRIPT_DIR/01_fastqc.sh" "$CONFIG_FILE"
bash "$SCRIPT_DIR/02_fastp_filter.sh" "$CONFIG_FILE"
bash "$SCRIPT_DIR/03_collapse_reads_mirdeep2.sh" "$CONFIG_FILE"
bash "$SCRIPT_DIR/04_quantifier_mirdeep2.sh" "$CONFIG_FILE"
bash "$SCRIPT_DIR/05_collect_quantifier_outputs.sh" "$CONFIG_FILE"

echo "All miRNA upstream processing steps finished."
