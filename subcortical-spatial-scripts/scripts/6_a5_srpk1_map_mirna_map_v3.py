#!/usr/bin/env python3
"""Bilateral subcortical spatial correspondence between peripheral SRPK1-associated and miR-139-5p-associated MSN maps using BrainSMASH nulls."""
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
# matched x/y maps
# x = SRPK1 map
# y = miR-139-5p map
# =========================
srpk1_dir = PROJECT_ROOT / "msn_results_subcort" / "srpk1_msn_subcort_assoc_v2"
mir_dir = PROJECT_ROOT / "msn_results_subcort" / "mir1395p_msn_subcort_assoc"

# In the updated R workflow, srpk1_imaging_map_main.csv is the
# Batch-adjusted primary SRPK1 map. miR-139-5p maps do not include
# RNA-seq Batch, so model names explicitly describe both sides.
matched_files = {
    "srpk1_batch_main_vs_mir1395p_main": {
        "srpk1_model_tag": "batch_main",
        "mirna_model_tag": "batch_main",
        "x_file": srpk1_dir / "srpk1_imaging_map_main.csv",
        "y_file": mir_dir / "mir1395p_imaging_map_main.csv",
    },
    "srpk1_batch_plus_med_history_vs_mir1395p_plus_med_history": {
        "srpk1_model_tag": "batch_plus_med_history",
        "mirna_model_tag": "batch_plus_med_history",
        "x_file": srpk1_dir / "srpk1_imaging_map_plus_med_history.csv",
        "y_file": mir_dir / "mir1395p_imaging_map_plus_med_history.csv",
    },
    "srpk1_plus_neutrophils_vs_mir1395p_main": {
        "srpk1_model_tag": "batch_plus_neutrophils",
        "mirna_model_tag": "batch_main",
        "x_file": srpk1_dir / "srpk1_imaging_map_plus_neutrophils.csv",
        "y_file": mir_dir / "mir1395p_imaging_map_main.csv",
    },
    "srpk1_plus_med_history_neutrophils_vs_mir1395p_plus_med_history": {
        "srpk1_model_tag": "batch_plus_med_history_neutrophils",
        "mirna_model_tag": "batch_plus_med_history",
        "x_file": srpk1_dir / "srpk1_imaging_map_plus_med_history_neutrophils.csv",
        "y_file": mir_dir / "mir1395p_imaging_map_plus_med_history.csv",
    },
}

dist_file = TIAN_S4_DISTANCE_MATRIX

# =========================
# Output paths
# =========================
out_dir = OUTPUT_ROOT / "a5_sub_srpk1_main_vs_mir1395p_v3"
out_dir.mkdir(parents=True, exist_ok=True)

result_file = out_dir / "a5_sub_srpk1_batch_main_vs_mir1395p_results_v3.csv"
meta_file = out_dir / "a5_sub_srpk1_batch_main_vs_mir1395p_meta_v3.json"

# =========================
# Analysis settings
# =========================
n_perm = 10000
seed = 1234

# =========================
# load distance matrix
# =========================
dist_df = pd.read_csv(dist_file, index_col=0)
dist_df.index = dist_df.index.astype(str).str.strip()
dist_df.columns = dist_df.columns.astype(str).str.strip()
dist_order = dist_df.index.tolist()

# =========================
# Helper functions
# =========================
def prepare_map(df, roi_col="ROI", t_col="t", keep_extra=None):
    if keep_extra is None:
        keep_extra = []
    df = df.copy()
    df[roi_col] = df[roi_col].astype(str).str.strip()
    need = [roi_col, t_col] + [c for c in keep_extra if c in df.columns]
    df = df[need]
    return df

def make_nulls_for_x(x, D, null_file, seed=1234, n_perm=10000):
    gen = Base(x=x, D=D, seed=seed)

    if null_file.exists():
        print("Loading existing nulls:", null_file.name)
        surrogates = np.load(null_file)
        print("Surrogates shape:", surrogates.shape)

        if surrogates.shape[1] != len(x):
            raise ValueError(
                f"Cached nulls ROI dimension {surrogates.shape[1]} does not match current x length {len(x)}"
            )
    else:
        print("Generating nulls:", null_file.name)
        surrogates = gen(n=n_perm)
        print("Surrogates shape:", surrogates.shape)
        np.save(null_file, surrogates)

    return surrogates

# =========================
# run matched analyses
# =========================
all_results = []
meta = {
    "n_perm_requested": n_perm,
    "seed": seed,
    "dist_file": str(dist_file),
    "x_map_family": "Peripheral_SRPK1_subcortical_MSN_tmap",
    "y_map_family": "miR1395p_subcortical_MSN_tmap",
    "primary_x_model": "batch_main: MSN ~ SRPK1 + Batch + age + sex + EDL + eTIV",
    "analysis_version": "v3",
    "note": (
        "Version 3 adds SRPK1 models adjusted for neutrophils, including "
        "batch_plus_neutrophils vs miR-139-5p batch_main and "
        "batch_plus_med_history_neutrophils vs miR-139-5p batch_plus_med_history."
    ),
    "matched_files": {}
}

for model_name, paths in matched_files.items():
    print("\n==============================")
    print("Running matched model:", model_name)

    x_file = paths["x_file"]
    y_file = paths["y_file"]
    srpk1_model_tag = paths.get("srpk1_model_tag", model_name)
    mirna_model_tag = paths.get("mirna_model_tag", model_name)

    print("x_file:", x_file)
    print("y_file:", y_file)

    x_df = pd.read_csv(x_file)
    y_df = pd.read_csv(y_file)

    x_df = prepare_map(x_df, roi_col="ROI", t_col="t", keep_extra=["beta", "p", "p_fdr"])
    y_df = prepare_map(y_df, roi_col="ROI", t_col="t", keep_extra=["p", "p_fdr"])

    x_df = x_df.rename(columns={"t": "x_t"})
    y_df = y_df.rename(columns={"t": "y_t"})

    df = x_df.merge(y_df, on="ROI", how="inner")
    df = df.dropna(subset=["x_t", "y_t"]).copy()

    print("Merged ROI count:", len(df))
    print(df.head())

    if len(df) != 54:
        raise ValueError(f"Expected 54 ROIs after merge for model {model_name}, got {len(df)}")

    # ROI set consistency with distance matrix
    roi_df_set = set(df["ROI"])
    roi_d_set = set(dist_df.index)
    if roi_df_set != roi_d_set:
        only_df = sorted(roi_df_set - roi_d_set)
        only_d = sorted(roi_d_set - roi_df_set)
        raise ValueError(
            f"ROI set mismatch for model {model_name}.\n"
            f"Only in merged df: {only_df}\n"
            f"Only in distance matrix: {only_d}"
        )

    # force atlas order
    df["ROI"] = pd.Categorical(df["ROI"], categories=dist_order, ordered=True)
    df = df.sort_values("ROI").reset_index(drop=True)

    roi_order = df["ROI"].astype(str).tolist()
    D = dist_df.loc[roi_order, roi_order].to_numpy(dtype=float)

    x = df["x_t"].to_numpy(dtype=float)
    y = df["y_t"].to_numpy(dtype=float)

    # save merged input
    merged_file = out_dir / f"a5_sub_input_merged_{model_name}_v3.csv"
    vectors_file = out_dir / f"a5_sub_ordered_vectors_{model_name}_v3.csv"
    null_corr_file = out_dir / f"a5_sub_null_distribution_{model_name}_v3.csv"
    null_file = out_dir / f"a5_sub_nulls_x_srpk1_{model_name}_v3.npy"

    df.to_csv(merged_file, index=False)

    pd.DataFrame({
        "ROI": roi_order,
        "srpk1_t": x,
        "mir1395p_t": y
    }).to_csv(vectors_file, index=False)

    # observed correlations
    r_obs, p_naive = pearsonr(x, y)
    rho_obs, p_spear = spearmanr(x, y)

    # nulls generated from x = SRPK1 map
    surrogates = make_nulls_for_x(
        x=x,
        D=D,
        null_file=null_file,
        seed=seed,
        n_perm=n_perm
    )

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
        "comparison": f"periph_SRPK1_subcort_MSN_tmap_{model_name}_spatial_correspondence",
        "model": model_name,
        "srpk1_model_tag": srpk1_model_tag,
        "mirna_model_tag": mirna_model_tag,
        "x_file": str(x_file),
        "y_file": str(y_file),
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
        "merged_file": str(merged_file.name),
        "vectors_file": str(vectors_file.name),
        "null_file": str(null_file.name),
        "null_corr_file": str(null_corr_file.name),
    })

    meta["matched_files"][model_name] = {
        "srpk1_model_tag": srpk1_model_tag,
        "mirna_model_tag": mirna_model_tag,
        "x_file": str(x_file),
        "y_file": str(y_file),
        "null_file": str(null_file.name)
    }

# =========================
# save summary
# =========================
result_df = pd.DataFrame(all_results)
result_df.to_csv(result_file, index=False)

with open(meta_file, "w") as f:
    json.dump(meta, f, indent=2)

print("\n==============================")
print("All done.")
print("Result summary:", result_file)
print("Meta file:", meta_file)
print(result_df)
