#!/usr/bin/env bash
set -euo pipefail

# Generate BED12 gene model for RSeQC from Ensembl GRCh38 release 113 GTF

REF_DIR="./ref"
GTF="${REF_DIR}/Homo_sapiens.GRCh38.113.chr.gtf"
GTF_TO_GENEPRED="${REF_DIR}/gtfToGenePred"
GENEPRED_TO_BED="${REF_DIR}/genePredToBed"
OUT_DIR="./results/05_ref_aux"

mkdir -p "${OUT_DIR}"

"${GTF_TO_GENEPRED}" "${GTF}" "${OUT_DIR}/GRCh38.113.genePred"
"${GENEPRED_TO_BED}" "${OUT_DIR}/GRCh38.113.genePred" "${OUT_DIR}/GRCh38.113.bed12"

echo "Done: BED12 saved to ${OUT_DIR}/GRCh38.113.bed12"
