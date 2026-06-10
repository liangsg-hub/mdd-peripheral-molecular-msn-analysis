#!/usr/bin/env bash
set -euo pipefail

# Generate splice sites from Ensembl GRCh38 release 113 GTF

REF_DIR="./ref"
GTF="${REF_DIR}/Homo_sapiens.GRCh38.113.chr.gtf"
OUT_DIR="./results/05_ref_aux"

mkdir -p "${OUT_DIR}"

hisat2_extract_splice_sites.py "${GTF}" > "${OUT_DIR}/GRCh38.113_splice_sites.txt"

echo "Done: splice sites saved to ${OUT_DIR}/GRCh38.113_splice_sites.txt"
