#!/usr/bin/env bash
set -euo pipefail

# Align batch 2 reads with HISAT2
# Batch 2 libraries are reverse-stranded

REF_DIR="./ref"
CLEAN_DIR="./results/02_fastp/batch2"
OUT_DIR="./results/07_align_batch2"
LOG_DIR="./logs/align_batch2"
THREADS=16

INDEX_PREFIX="${REF_DIR}/genome"
SPLICE_SITES="./results/05_ref_aux/GRCh38.113_splice_sites.txt"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

for R1 in "${CLEAN_DIR}"/*_R1.clean.fastq.gz; do
    SAMPLE=$(basename "${R1}" _R1.clean.fastq.gz)
    R2="${CLEAN_DIR}/${SAMPLE}_R2.clean.fastq.gz"

    hisat2 \
        -x "${INDEX_PREFIX}" \
        --rna-strandness RF \
        --known-splicesite-infile "${SPLICE_SITES}" \
        -1 "${R1}" \
        -2 "${R2}" \
        -p "${THREADS}" \
        2> "${LOG_DIR}/${SAMPLE}.hisat2.log" \
    | samtools sort -@ "${THREADS}" -O BAM -o "${OUT_DIR}/${SAMPLE}.bam"

    samtools index -@ "${THREADS}" "${OUT_DIR}/${SAMPLE}.bam"
    samtools flagstat -@ "${THREADS}" "${OUT_DIR}/${SAMPLE}.bam" > "${OUT_DIR}/${SAMPLE}.flagstat.txt"
    samtools stats -@ "${THREADS}" "${OUT_DIR}/${SAMPLE}.bam" > "${OUT_DIR}/${SAMPLE}.stats.txt"
done

echo "Done: HISAT2 alignment for batch2"
