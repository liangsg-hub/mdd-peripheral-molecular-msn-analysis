#!/usr/bin/env bash
# Copy this file to config.sh and edit paths before running the pipeline.

PROJECT_DIR="/path/to/mirna_project"
RAW_DIR="${PROJECT_DIR}/01raw_data"
FASTQC_DIR="${PROJECT_DIR}/01fastqc"
CLEAN_DIR="${PROJECT_DIR}/02clean_fq"
COLLAPSED_DIR="${PROJECT_DIR}/03collapsed_mirdeep2"
QUANTIFIER_DIR="${PROJECT_DIR}/04quantifier"
QUANTIFIER_LIGHT_DIR="${QUANTIFIER_DIR}/samples_light"

REF_DIR="/path/to/mirna_reference"
MATURE_FA="${REF_DIR}/human_mature_miRNA.fa"
HAIRPIN_FA="${REF_DIR}/human_hairpin.fa"

SPECIES="hsa"
THREADS=8
