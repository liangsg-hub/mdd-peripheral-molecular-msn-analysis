#!/usr/bin/env python3
"""Left-hemisphere subcortical spatial correspondence between AHBA-derived SRPK1 expression and the MDD-HC MSN disorder map using BrainSMASH nulls."""
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
# Input paths
# =========================
expr_file = PROJECT_ROOT / "ahba_tian_subcortex_srpk1" / "tian_subcortex_SRPK1_main_named.csv"
disease_file = PROJECT_ROOT / "msn_results_subcort" / "tmap_mdd_vs_hc_covs_subcort.csv"
dist_file = TIAN_S4_DISTANCE_MATRIX

out_dir = OUTPUT_ROOT / "a3_sub_ahba_srpk1_expr_vs_disease_tmap_lh"
out_dir.mkdir(parents=True, exist_ok=True)

out_merged = out_dir / "a3_sub_input_merged_lh_v2.csv"
out_vectors = out_dir / "a3_sub_ordered_vectors_lh_v2.csv"
out_null = out_dir / "a3_sub_null_distribution_lh_v2.csv"
out_result = out_dir / "a3_sub_result_lh_v2.csv"
null_file = out_dir / "a3_sub_nulls_x_ahba_srpk1_expr_tian_s4_lh_v2.npy"
meta_file = out_dir / "a3_sub_null_meta_lh_v2.json"
roi_order_check_file = out_dir / "a3_tian_s4_lh_roi_order_check_v2.csv"


# =========================
# Analysis settings
# =========================
n_perm = 10000
seed = 1234

roi_col = "ROI"
expr_val_col = "SRPK1"
disease_val_col = "t_group_MDD"
expected_full_n_roi = 54
expected_hemi_n_roi = 27
hemisphere_to_use = "left"


# =========================
# Helper functions
# =========================
def check_file(path, label):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def normalize_roi_name(x):
    return str(x).strip().strip("'").strip('"')


def read_table(path, label, index_col=None):
    check_file(path, label)
    return pd.read_csv(path, index_col=index_col)


def check_required_columns(df, required_cols, label):
    missing = sorted(set(required_cols) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {label}: {missing}")


def prepare_value_map(df, value_col, value_name, label):
    check_required_columns(df, [roi_col, value_col], label)

    out = df[[roi_col, value_col]].rename(columns={value_col: value_name}).copy()
    out[roi_col] = out[roi_col].map(normalize_roi_name)
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce")

    if out[roi_col].duplicated().any():
        dup = out.loc[out[roi_col].duplicated(), roi_col].tolist()
        raise ValueError(f"Duplicated ROI in {label}: {dup[:20]}")

    return out


def align_to_target_order(df, value_name, target_order, label):
    input_set = set(df[roi_col])
    missing = [x for x in target_order if x not in input_set]

    if missing:
        raise ValueError(f"Target LH ROI names missing in {label}: {missing}")

    lookup = pd.DataFrame({
        roi_col: target_order,
        "atlas_order_full": np.arange(28, 28 + len(target_order)),
        "atlas_order_lh": np.arange(1, len(target_order) + 1),
        "hemisphere": "lh",
    })

    aligned = lookup.merge(
        df[[roi_col, value_name]],
        on=roi_col,
        how="left",
        validate="one_to_one"
    )

    if aligned[value_name].isna().any():
        bad = aligned.loc[aligned[value_name].isna(), roi_col].tolist()
        raise ValueError(f"Missing numeric values after LH alignment in {label}: {bad}")

    return aligned


def check_distance_matrix(dist_df):
    dist_df.index = dist_df.index.map(normalize_roi_name)
    dist_df.columns = dist_df.columns.map(normalize_roi_name)

    if dist_df.shape != (expected_full_n_roi, expected_full_n_roi):
        raise ValueError(
            f"Expected full Tian S4 distance matrix shape "
            f"({expected_full_n_roi}, {expected_full_n_roi}), got {dist_df.shape}."
        )

    row_order = dist_df.index.tolist()
    col_order = dist_df.columns.tolist()

    if row_order != col_order:
        raise ValueError("Distance matrix row and column ROI orders are not identical.")

    if len(set(row_order)) != expected_full_n_roi:
        s = pd.Series(row_order)
        dup = s[s.duplicated()].tolist()
        raise ValueError(f"Duplicated ROI names in distance matrix: {dup[:20]}")

    return dist_df, row_order


# =========================
# read data
# =========================
expr_df = read_table(expr_file, "AHBA SRPK1 expression file")
disease_df = read_table(disease_file, "disease t-map file")
dist_df = read_table(dist_file, "Tian S4 distance matrix", index_col=0)

expr_df.columns = [str(x).strip() for x in expr_df.columns]
disease_df.columns = [str(x).strip() for x in disease_df.columns]

expr_df[roi_col] = expr_df[roi_col].map(normalize_roi_name)
disease_df[roi_col] = disease_df[roi_col].map(normalize_roi_name)

check_required_columns(expr_df, [roi_col, expr_val_col], "AHBA SRPK1 expression file")
check_required_columns(disease_df, [roi_col, disease_val_col], "disease t-map file")

dist_df, full_order = check_distance_matrix(dist_df)

# Tian S4 subcortical atlas: 54 ROIs in total.
# The first 27 ROIs are right hemisphere and the last 27 ROIs are left hemisphere.
rh_order = full_order[:expected_hemi_n_roi]
lh_order = full_order[expected_hemi_n_roi:expected_full_n_roi]
target_order = lh_order

if len(target_order) != expected_hemi_n_roi:
    raise ValueError(f"Expected {expected_hemi_n_roi} LH ROIs, got {len(target_order)}.")

roi_order_check = pd.DataFrame({
    "atlas_order_full": np.arange(1, expected_full_n_roi + 1),
    "ROI": full_order,
    "hemisphere_by_order": ["rh"] * expected_hemi_n_roi + ["lh"] * expected_hemi_n_roi,
})
roi_order_check.to_csv(roi_order_check_file, index=False)

print("Full Tian S4 ROI count:", len(full_order))
print("RH ROI count:", len(rh_order))
print("LH ROI count:", len(lh_order))
print("First RH ROI:", rh_order[0])
print("Last RH ROI:", rh_order[-1])
print("First LH ROI:", lh_order[0])
print("Last LH ROI:", lh_order[-1])

# =========================
# prepare and align LH maps
# =========================
expr_map = prepare_value_map(expr_df, expr_val_col, "ahba_srpk1_expr", "AHBA SRPK1 expression map")
disease_map = prepare_value_map(disease_df, disease_val_col, "disease_tmap", "disease t-map")

expr_lh = align_to_target_order(
    expr_map,
    "ahba_srpk1_expr",
    target_order,
    "AHBA SRPK1 expression map"
)

disease_lh = align_to_target_order(
    disease_map,
    "disease_tmap",
    target_order,
    "disease t-map"
)

# Optional p-value columns from disease map
optional_cols = ["p_group_MDD", "p_fdr_group_MDD"]
available_optional_cols = [c for c in optional_cols if c in disease_df.columns]
if available_optional_cols:
    disease_extra = disease_df[[roi_col] + available_optional_cols].copy()
    disease_extra[roi_col] = disease_extra[roi_col].map(normalize_roi_name)
    disease_lh = disease_lh.merge(
        disease_extra,
        on=roi_col,
        how="left",
        validate="one_to_one"
    )

# =========================
# merge aligned LH maps
# =========================
df = expr_lh.merge(
    disease_lh,
    on=[roi_col, "atlas_order_full", "atlas_order_lh", "hemisphere"],
    how="inner",
    validate="one_to_one"
)

df = df.dropna(subset=["ahba_srpk1_expr", "disease_tmap"]).copy()

print("Merged LH ROI count:", len(df))
print(df.head())

if len(df) != expected_hemi_n_roi:
    raise ValueError(f"Expected {expected_hemi_n_roi} LH ROIs after merge, got {len(df)}.")

# =========================
# fix LH atlas order and distance matrix
# =========================
df[roi_col] = pd.Categorical(df[roi_col], categories=target_order, ordered=True)
df = df.sort_values(roi_col).reset_index(drop=True)

roi_order = df[roi_col].astype(str).tolist()
if roi_order != target_order:
    raise ValueError("Final LH ROI order does not match the target LH order.")

dist_lh = dist_df.loc[target_order, target_order]
D = dist_lh.to_numpy(dtype=float)

if D.shape != (expected_hemi_n_roi, expected_hemi_n_roi):
    raise ValueError(f"Expected LH distance matrix shape (27, 27), got {D.shape}.")

x = df["ahba_srpk1_expr"].to_numpy(dtype=float)
y = df["disease_tmap"].to_numpy(dtype=float)

# save merged and ordered vectors
df.to_csv(out_merged, index=False)

pd.DataFrame({
    "atlas_order_full": df["atlas_order_full"],
    "atlas_order_lh": df["atlas_order_lh"],
    "hemisphere": df["hemisphere"],
    "ROI": roi_order,
    "ahba_srpk1_expr": x,
    "disease_tmap": y,
}).to_csv(out_vectors, index=False)

# =========================
# observed correlations
# =========================
r_obs, p_naive = pearsonr(x, y)
rho_obs, p_spear = spearmanr(x, y)

print(f"Observed Pearson r = {r_obs:.6f}")
print(f"Observed Spearman rho = {rho_obs:.6f}")

# =========================
# BrainSMASH
# nulls generated from x = AHBA SRPK1 expression, using LH distance matrix
# =========================
gen = Base(x=x, D=D, seed=seed)

if null_file.exists():
    print("Loading existing LH nulls...")
    surrogates = np.load(null_file)
    print("Surrogates shape:", surrogates.shape)

    if surrogates.shape[1] != len(x):
        raise ValueError(
            f"Cached nulls ROI dimension {surrogates.shape[1]} does not match current LH data {len(x)}. "
            "Delete the cached null file and rerun this script."
        )
else:
    print("Generating LH nulls...")
    surrogates = gen(n=n_perm)
    print("Surrogates shape:", surrogates.shape)
    np.save(null_file, surrogates)

# correlate surrogate x with fixed y
null_r = np.array(
    [pearsonr(surrogates[i, :], y)[0] for i in range(surrogates.shape[0])],
    dtype=float
)

p_spatial = (np.sum(np.abs(null_r) >= abs(r_obs)) + 1) / (len(null_r) + 1)

print(f"Spatial-null p = {p_spatial:.6f}")

pd.DataFrame({"null_r": null_r}).to_csv(out_null, index=False)

# =========================
# summary result
# =========================
result_df = pd.DataFrame([{
    "comparison": "a3_sub_ahba_srpk1_expr_lh_vs_mdd_vs_hc_sub_tmap_lh",
    "atlas": "Tian_S4_subcortex",
    "hemisphere": hemisphere_to_use,
    "n_roi": len(df),
    "full_atlas_n_roi": expected_full_n_roi,
    "hemisphere_n_roi": expected_hemi_n_roi,
    "roi_selection": "last_27_rois_of_tian_s4_distance_matrix",
    "pearson_r": r_obs,
    "pearson_p_naive": p_naive,
    "spearman_rho": rho_obs,
    "spearman_p_naive": p_spear,
    "p_spatial_brainsmash": p_spatial,
    "n_perm": surrogates.shape[0],
    "null_mean": float(np.nanmean(null_r)),
    "null_sd": float(np.nanstd(null_r)),
    "null_min": float(np.nanmin(null_r)),
    "null_max": float(np.nanmax(null_r)),
}])

result_df.to_csv(out_result, index=False)

# =========================
# meta
# =========================
meta = {
    "analysis": "left_subcortical_AHBA_SRPK1_expression_vs_MDD_HC_MSN_tmap",
    "atlas": "Tian S4 subcortex",
    "hemisphere": hemisphere_to_use,
    "full_atlas_n_roi": expected_full_n_roi,
    "hemisphere_n_roi": expected_hemi_n_roi,
    "roi_selection": "The first 27 ROIs of the Tian S4 distance matrix are right hemisphere; the last 27 ROIs are left hemisphere. This analysis used the last 27 ROIs.",
    "n_perm_requested": n_perm,
    "n_perm_used": int(surrogates.shape[0]),
    "seed": seed,
    "expr_file": str(expr_file),
    "disease_file": str(disease_file),
    "dist_file": str(dist_file),
    "map_x": "AHBA_SRPK1_expression_left_subcortex",
    "map_y": "MDD_vs_HC_subcortical_tmap_left_subcortex",
    "n_roi_used": int(len(x)),
    "roi_order_check_file": str(roi_order_check_file.name),
    "ordered_vector_file": str(out_vectors.name),
    "null_file": str(null_file.name),
    "note": "BrainSMASH surrogate maps were generated from the LH AHBA SRPK1 expression vector and the LH 27 x 27 Tian S4 subcortical distance matrix."
}

with open(meta_file, "w") as f:
    json.dump(meta, f, indent=2)

print("Saved merged input:", out_merged)
print("Saved ordered vectors:", out_vectors)
print("Saved null distribution:", out_null)
print("Saved result:", out_result)
print("Saved meta:", meta_file)
print("Saved ROI order check:", roi_order_check_file)
print(result_df)
