#!/usr/bin/env python3
"""Bilateral subcortical spatial correspondence between the MDD-HC MSN disorder map and peripheral SRPK1-associated MSN maps using BrainSMASH nulls."""
import warnings
import os
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from brainsmash.mapgen.base import Base

# =========================
# Portable project paths
# =========================
# Set MSN_PROJECT_ROOT to the folder that contains the input data.
# Example:
#   export MSN_PROJECT_ROOT=/path/to/msn_2026
# Optional: set TIAN_S4_DISTANCE_MATRIX if the distance matrix is stored elsewhere.
# Optional: set MSN_OUTPUT_ROOT to redirect generated outputs.
PROJECT_ROOT = Path(os.environ.get("MSN_PROJECT_ROOT", "/path/to/msn_2026")).expanduser()
OUTPUT_ROOT = Path(os.environ.get("MSN_OUTPUT_ROOT", PROJECT_ROOT / "msn_results_subcort")).expanduser()
TIAN_S4_DISTANCE_MATRIX = Path(
    os.environ.get("TIAN_S4_DISTANCE_MATRIX", PROJECT_ROOT / "tian_s4_distance_matrix.csv")
).expanduser()


# =========================
# fixed x map and distance matrix
# x = disease map
# =========================
disease_file = PROJECT_ROOT / "msn_results_subcort" / "tmap_mdd_vs_hc_covs_subcort.csv"
dist_file = TIAN_S4_DISTANCE_MATRIX

# =========================
# y maps to test
# y = peripheral SRPK1 subcortical maps
#
# In the updated R workflow, srpk1_imaging_map_main.csv is the
# Batch-adjusted primary map:
#   MSN ~ SRPK1 + Batch + age + sex + EDL + eTIV
# The remaining maps are sensitivity models that also retain Batch.
# =========================
img_dir = PROJECT_ROOT / "msn_results_subcort" / "srpk1_msn_subcort_assoc_v2"

img_files = {
    "batch_main": img_dir / "srpk1_imaging_map_main.csv",
    "batch_plus_med_history": img_dir / "srpk1_imaging_map_plus_med_history.csv",
    "batch_plus_neutrophils": img_dir / "srpk1_imaging_map_plus_neutrophils.csv",
    "batch_plus_med_history_neutrophils": img_dir / "srpk1_imaging_map_plus_med_history_neutrophils.csv",
}

# =========================
# Output paths
# =========================
out_dir = OUTPUT_ROOT / "a2_sub_disease_vs_srpk1_main"
out_dir.mkdir(parents=True, exist_ok=True)

null_file = out_dir / "a2_sub_nulls_x_mdd_vs_hc_tmap_tian_s4.npy"
meta_file = out_dir / "a2_sub_null_meta.json"
result_file = out_dir / "a2_sub_disease_vs_srpk1_main_results.csv"

# =========================
# Analysis settings
# =========================
n_perm = 10000
seed = 1234

# =========================
# read fixed x and distance matrix
# =========================
disease_df = pd.read_csv(disease_file)
dist_df = pd.read_csv(dist_file, index_col=0)

disease_df["ROI"] = disease_df["ROI"].astype(str).str.strip()
dist_df.index = dist_df.index.astype(str).str.strip()
dist_df.columns = dist_df.columns.astype(str).str.strip()

# x = disease t-map
if "t_group_MDD" not in disease_df.columns:
    raise ValueError("'t_group_MDD' column not found in disease_file")

x_df = disease_df[["ROI", "t_group_MDD"]].rename(columns={"t_group_MDD": "disease_t"}).dropna().copy()

# check ROI set against distance matrix
roi_x_set = set(x_df["ROI"])
roi_d_set = set(dist_df.index)

if roi_x_set != roi_d_set:
    only_x = sorted(roi_x_set - roi_d_set)
    only_d = sorted(roi_d_set - roi_x_set)
    raise ValueError(
        "ROI set mismatch between disease map and distance matrix.\n"
        f"Only in disease map: {only_x}\n"
        f"Only in distance matrix: {only_d}"
    )

# fix to atlas order
dist_order = dist_df.index.tolist()
x_df["ROI"] = pd.Categorical(x_df["ROI"], categories=dist_order, ordered=True)
x_df = x_df.sort_values("ROI").reset_index(drop=True)

roi_order = x_df["ROI"].astype(str).tolist()
dist_df = dist_df.loc[roi_order, roi_order]

x = x_df["disease_t"].to_numpy(dtype=float)
D = dist_df.to_numpy(dtype=float)

if len(x) != 54:
    raise ValueError(f"Expected 54 ROIs for disease map, got {len(x)}")

# =========================
# generate / load shared nulls for fixed x
# =========================
gen = Base(x=x, D=D, seed=seed)

if null_file.exists():
    print("Loading existing shared nulls...")
    surrogates = np.load(null_file)
    print("Surrogates shape:", surrogates.shape)

    if surrogates.shape[1] != len(x):
        raise ValueError(
            f"Cached nulls ROI dimension {surrogates.shape[1]} does not match current x length {len(x)}"
        )
else:
    print("Generating shared nulls from disease t-map...")
    surrogates = gen(n=n_perm)
    print("Surrogates shape:", surrogates.shape)
    np.save(null_file, surrogates)

# save meta
meta = {
    "n_perm_requested": n_perm,
    "n_perm_used": int(surrogates.shape[0]),
    "seed": seed,
    "disease_file": str(disease_file),
    "dist_file": str(dist_file),
    "x_map": "MDD_vs_HC_subcortical_tmap",
    "y_map_family": "Peripheral_SRPK1_subcortical_MSN_tmaps",
    "primary_y_model": "batch_main: MSN ~ SRPK1 + Batch + age + sex + EDL + eTIV",
    "sensitivity_y_models": [
        "batch_plus_med_history",
        "batch_plus_neutrophils",
        "batch_plus_med_history_neutrophils"
    ],
    "n_roi": int(len(x)),
    "shared_null_file": str(null_file.name),
    "img_files": {k: str(v) for k, v in img_files.items()}
}
with open(meta_file, "w") as f:
    json.dump(meta, f, indent=2)

# =========================
# run all A2-sub models
# =========================
all_results = []

for model_name, img_file in img_files.items():
    print("\n==============================")
    print("Running model:", model_name)
    print("File:", img_file)

    img_df = pd.read_csv(img_file)
    img_df["ROI"] = img_df["ROI"].astype(str).str.strip()

    if "t" not in img_df.columns:
        raise ValueError(f"'t' column not found in {img_file}")

    y_df = img_df[["ROI", "t", "beta", "p", "p_fdr"]].copy()
    y_df = y_df.rename(columns={"t": "img_t"})

    # merge with fixed x
    df = x_df.merge(y_df, on="ROI", how="inner")
    df = df.dropna(subset=["disease_t", "img_t"]).copy()

    print("Merged ROI count:", len(df))
    if len(df) != 54:
        raise ValueError(f"Expected 54 ROIs after merge for model {model_name}, got {len(df)}")

    # force atlas order
    df["ROI"] = pd.Categorical(df["ROI"], categories=roi_order, ordered=True)
    df = df.sort_values("ROI").reset_index(drop=True)

    y = df["img_t"].to_numpy(dtype=float)

    # save merged input
    merged_file = out_dir / f"a2_sub_input_merged_{model_name}.csv"
    vectors_file = out_dir / f"a2_sub_ordered_vectors_{model_name}.csv"
    null_corr_file = out_dir / f"a2_sub_null_distribution_{model_name}.csv"

    df.to_csv(merged_file, index=False)

    pd.DataFrame({
        "ROI": roi_order,
        "disease_t": x,
        "img_t": y
    }).to_csv(vectors_file, index=False)

    # observed correlations
    r_obs, p_naive = pearsonr(x, y)
    rho_obs, p_spear = spearmanr(x, y)

    # correlate shared nulls with current y
    null_r = np.array(
        [pearsonr(surrogates[i, :], y)[0] for i in range(surrogates.shape[0])],
        dtype=float
    )

    p_spatial = (np.sum(np.abs(null_r) >= abs(r_obs)) + 1) / (len(null_r) + 1)

    pd.DataFrame({"null_r": null_r}).to_csv(null_corr_file, index=False)

    print(f"Observed Pearson r = {r_obs:.6f}")
    print(f"Observed Spearman rho = {rho_obs:.6f}")
    print(f"Spatial-null p = {p_spatial:.6f}")

    all_results.append({
        "comparison": f"disorder_tmap_vs_periph_SRPK1_subcort_tmap_{model_name}",
        "model": model_name,
        "img_file": str(img_file),
        "n_roi": len(df),
        "pearson_r": r_obs,
        "pearson_p_naive": p_naive,
        "spearman_rho": rho_obs,
        "spearman_p_naive": p_spear,
        "p_spatial_brainsmash": p_spatial,
        "n_perm": surrogates.shape[0],
        "null_mean": float(np.mean(null_r)),
        "null_sd": float(np.std(null_r)),
        "null_min": float(np.min(null_r)),
        "null_max": float(np.max(null_r)),
    })

# =========================
# save summary
# =========================
result_df = pd.DataFrame(all_results)
result_df.to_csv(result_file, index=False)

print("\n==============================")
print("All done.")
print("Shared null file:", null_file)
print("Meta file:", meta_file)
print("Result summary:", result_file)
print(result_df)
