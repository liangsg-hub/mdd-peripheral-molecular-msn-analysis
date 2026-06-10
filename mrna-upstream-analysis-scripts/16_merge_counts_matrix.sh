#!/usr/bin/env bash
set -euo pipefail

# Merge batch1 and batch2 count matrices by gene_id
# This script keeps Geneid and sample count columns only

BATCH1="./results/10_counts_batch1/gene_counts.txt"
BATCH2="./results/11_counts_batch2/gene_counts.txt"
OUT_DIR="./results/12_merged_counts"

mkdir -p "${OUT_DIR}"

python3 - <<'PY'
import pandas as pd

batch1 = pd.read_csv("./results/10_counts_batch1/gene_counts.txt", sep="\t", comment="#")
batch2 = pd.read_csv("./results/11_counts_batch2/gene_counts.txt", sep="\t", comment="#")

meta_cols = ["Geneid", "Chr", "Start", "End", "Strand", "Length"]
sample_cols1 = [c for c in batch1.columns if c not in meta_cols]
sample_cols2 = [c for c in batch2.columns if c not in meta_cols]

df1 = batch1[["Geneid"] + sample_cols1].copy()
df2 = batch2[["Geneid"] + sample_cols2].copy()

merged = df1.merge(df2, on="Geneid", how="outer")
merged.to_csv("./results/12_merged_counts/gene_counts_merged.tsv", sep="\t", index=False)
PY

echo "Done: merged count matrix saved to ./results/12_merged_counts/gene_counts_merged.tsv"
