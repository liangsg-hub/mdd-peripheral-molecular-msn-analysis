#!/usr/bin/env python3
"""Bilateral subcortical spatial correspondence between the MDD-HC MSN disorder map and miR-139-5p-associated MSN maps using BrainSMASH nulls."""
import warnings
warnings.filterwarnings("ignore")

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from brainsmash.mapgen.base import Base

# =========================
# user configuration
# =========================
PROJECT_ROOT = Path(
    os.environ.get("MSN_PROJECT_ROOT", "/path/to/msn_project")
).expanduser()
OUTPUT_ROOT = Path(
    os.environ.get("MSN_OUTPUT_ROOT", PROJECT_ROOT / "msn_results_subcort")
).expanduser()
TIAN_S4_DISTANCE_MATRIX = Path(
    os.environ.get(
        "TIAN_S4_DISTANCE_MATRIX",
        PROJECT_ROOT / "tian_s4_distance_matrix.csv",
    )
).expanduser()

# =========================
# fixed x map and distance matrix
# x = disease map
# =========================
disease_file = OUTPUT_ROOT / "tmap_mdd_vs_hc_covs_subcort.csv"
dist_file = TIAN_S4_DISTANCE_MATRIX

# =========================
# y maps to test
# y = miR-139-5p subcortical maps
# =========================
img_dir = OUTPUT_ROOT / "mir1395p_msn_subcort_assoc"

img_files = {
    "main": img_dir / "mir1395p_imaging_map_main.csv",
    "plus_med_history": img_dir / "mir1395p_imaging_map_plus_med_history.csv",
}

# =========================
# output
# =========================
out_dir = OUTPUT_ROOT / "a1_sub_disease_vs_mir1395p_batch"
out_dir.mkdir(parents=True, exist_ok=True)

null_file = out_dir / "a1_sub_nulls_x_mdd_vs_hc_tmap_tian_s4.npy"
meta_file = out_dir / "a1_sub_null_meta.json"
result_file = out_dir / "a1_sub_batch_results.csv"

# =========================
# settings
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

if "t_group_MDD" not in disease_df.columns:
    raise ValueError("'t_group_MDD' column not found in disease_file")

x_df = disease_df[["ROI", "t_group_MDD"]].rename(columns={"t_group_MDD": "disease_t"}).dropna().copy()

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
    "n_roi": int(len(x)),
    "shared_null_file": str(null_file.name),
    "img_files": {k: str(v) for k, v in img_files.items()}
}
with open(meta_file, "w") as f:
    json.dump(meta, f, indent=2)

# =========================
# run all A1-sub models
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

    # beta may be absent in some exports, so handle safely
    keep_cols = ["ROI", "t"]
    for col in ["beta", "p", "p_fdr", "N", "se"]:
        if col in img_df.columns:
            keep_cols.append(col)

    y_df = img_df[keep_cols].copy()
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
    merged_file = out_dir / f"a1_sub_input_merged_{model_name}.csv"
    vectors_file = out_dir / f"a1_sub_ordered_vectors_{model_name}.csv"
    null_corr_file = out_dir / f"a1_sub_null_distribution_{model_name}.csv"

    df.to_csv(merged_file, index=False)

    pd.DataFrame({
        "ROI": roi_order,
        "disease_t": x,
        "mir1395p_t": y
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