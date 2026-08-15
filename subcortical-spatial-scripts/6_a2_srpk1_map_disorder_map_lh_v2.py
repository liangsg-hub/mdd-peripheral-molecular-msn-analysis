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
# x = left hemisphere disease map
# =========================
disease_file = OUTPUT_ROOT / "tmap_mdd_vs_hc_covs_subcort.csv"
dist_file = TIAN_S4_DISTANCE_MATRIX

# =========================
# y maps to test
# y = left hemisphere peripheral SRPK1 subcortical maps
#
# In the updated R workflow, srpk1_imaging_map_main.csv is the
# Batch-adjusted primary map:
#   MSN ~ SRPK1 + Batch + age + sex + EDL + eTIV
# The remaining maps are sensitivity models that also retain Batch.
# =========================
img_dir = OUTPUT_ROOT / "srpk1_msn_subcort_assoc_v2"

img_files = {
    "batch_main": img_dir / "srpk1_imaging_map_main.csv",
    "batch_plus_med_history": img_dir / "srpk1_imaging_map_plus_med_history.csv",
    "batch_plus_neutrophils": img_dir / "srpk1_imaging_map_plus_neutrophils.csv",
    "batch_plus_med_history_neutrophils": img_dir / "srpk1_imaging_map_plus_med_history_neutrophils.csv",
}

# =========================
# output
# =========================
out_dir = OUTPUT_ROOT / "a2_sub_disease_vs_srpk1_lh"
out_dir.mkdir(parents=True, exist_ok=True)

null_file = out_dir / "a2_sub_nulls_x_mdd_vs_hc_tmap_tian_s4_lh_v2.npy"
meta_file = out_dir / "a2_sub_null_meta_lh_v2.json"
result_file = out_dir / "a2_sub_disease_vs_srpk1_lh_results_v2.csv"
roi_order_check_file = out_dir / "a2_tian_s4_lh_roi_order_check_v2.csv"

merged_dir = out_dir / "input_merged"
vectors_dir = out_dir / "ordered_vectors"
null_dist_dir = out_dir / "null_distributions"
for d in [merged_dir, vectors_dir, null_dist_dir]:
    d.mkdir(parents=True, exist_ok=True)

# =========================
# settings
# =========================
n_perm = 10000
seed = 1234
expected_full_n_roi = 54
expected_hemi_n_roi = 27
hemisphere = "left"

# =========================
# helpers
# =========================
def require_columns(df, cols, file_label):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {file_label}: {missing}")


def two_sided_spatial_p(null_r, r_obs):
    return (np.sum(np.abs(null_r) >= abs(r_obs)) + 1) / (len(null_r) + 1)


def normalize_roi_column(df):
    df = df.copy()
    df["ROI"] = df["ROI"].astype(str).str.strip()
    return df


def check_distance_matrix(dist_df):
    if dist_df.shape[0] != expected_full_n_roi or dist_df.shape[1] != expected_full_n_roi:
        raise ValueError(
            f"Expected full Tian S4 distance matrix shape "
            f"{expected_full_n_roi} x {expected_full_n_roi}, got {dist_df.shape}"
        )

    row_names = dist_df.index.astype(str).str.strip().tolist()
    col_names = dist_df.columns.astype(str).str.strip().tolist()

    if row_names != col_names:
        raise ValueError("Distance matrix row and column names are not identical or not in the same order.")

    if len(set(row_names)) != expected_full_n_roi:
        duplicated = pd.Series(row_names)[pd.Series(row_names).duplicated()].tolist()
        raise ValueError(f"Duplicated ROI names in distance matrix: {duplicated[:20]}")

    return row_names


def align_to_target_order(df, value_cols, target_order, label):
    require_columns(df, ["ROI"] + value_cols, label)
    df = normalize_roi_column(df)

    if df["ROI"].duplicated().any():
        duplicated = df.loc[df["ROI"].duplicated(), "ROI"].tolist()
        raise ValueError(f"Duplicated ROI names in {label}: {duplicated[:20]}")

    missing = [roi for roi in target_order if roi not in set(df["ROI"])]
    if missing:
        raise ValueError(f"Missing left hemisphere ROIs in {label}: {missing}")

    aligned = pd.DataFrame({"ROI": target_order}).merge(
        df[["ROI"] + value_cols],
        on="ROI",
        how="left",
        validate="one_to_one"
    )

    for col in value_cols:
        aligned[col] = pd.to_numeric(aligned[col], errors="coerce")

    if aligned[value_cols].isna().any().any():
        bad_rows = aligned.loc[aligned[value_cols].isna().any(axis=1), "ROI"].tolist()
        raise ValueError(f"Missing numeric values after left hemisphere alignment in {label}: {bad_rows}")

    return aligned


# =========================
# read fixed x and distance matrix
# =========================
disease_df = pd.read_csv(disease_file)
dist_df = pd.read_csv(dist_file, index_col=0)

dist_df.index = dist_df.index.astype(str).str.strip()
dist_df.columns = dist_df.columns.astype(str).str.strip()

full_order = check_distance_matrix(dist_df)
rh_order = full_order[:expected_hemi_n_roi]
lh_order = full_order[expected_hemi_n_roi:expected_full_n_roi]
target_order = lh_order

if len(rh_order) != expected_hemi_n_roi or len(lh_order) != expected_hemi_n_roi:
    raise ValueError(
        f"Unexpected hemisphere split: rh={len(rh_order)}, lh={len(lh_order)}"
    )

print("Full Tian S4 ROI count:", len(full_order))
print("RH ROI count:", len(rh_order))
print("LH ROI count:", len(lh_order))
print("First RH ROI:", rh_order[0])
print("Last RH ROI:", rh_order[-1])
print("First LH ROI:", lh_order[0])
print("Last LH ROI:", lh_order[-1])

roi_order_check = pd.DataFrame({
    "atlas_order_full": np.arange(1, expected_full_n_roi + 1),
    "ROI": full_order,
    "hemisphere_by_order": ["right"] * expected_hemi_n_roi + ["left"] * expected_hemi_n_roi,
    "used_in_lh_analysis": [False] * expected_hemi_n_roi + [True] * expected_hemi_n_roi,
})
roi_order_check.to_csv(roi_order_check_file, index=False)

# x = left hemisphere disease t-map
require_columns(disease_df, ["ROI", "t_group_MDD"], "disease_file")
disease_df = normalize_roi_column(disease_df)
x_all = disease_df[["ROI", "t_group_MDD"]].rename(columns={"t_group_MDD": "disease_t"}).dropna().copy()
x_df = align_to_target_order(x_all, ["disease_t"], target_order, "disease map")

roi_order = x_df["ROI"].astype(str).tolist()
dist_lh = dist_df.loc[roi_order, roi_order]

x = x_df["disease_t"].to_numpy(dtype=float)
D = dist_lh.to_numpy(dtype=float)

if len(x) != expected_hemi_n_roi:
    raise ValueError(f"Expected {expected_hemi_n_roi} left hemisphere ROIs for disease map, got {len(x)}")

if D.shape != (expected_hemi_n_roi, expected_hemi_n_roi):
    raise ValueError(f"Expected left hemisphere distance matrix shape 27 x 27, got {D.shape}")

# =========================
# generate or load shared nulls for fixed x
# =========================
gen = Base(x=x, D=D, seed=seed)

if null_file.exists():
    print("Loading existing shared left hemisphere nulls")
    surrogates = np.load(null_file)
    print("Surrogates shape:", surrogates.shape)

    if surrogates.shape[1] != len(x):
        raise ValueError(
            f"Cached nulls ROI dimension {surrogates.shape[1]} does not match current x length {len(x)}"
        )
    if surrogates.shape[0] != n_perm:
        print(f"Warning: cached n_perm={surrogates.shape[0]}, current n_perm={n_perm}")
else:
    print("Generating shared left hemisphere nulls from disease t-map")
    surrogates = gen(n=n_perm)
    print("Surrogates shape:", surrogates.shape)
    np.save(null_file, surrogates)

# =========================
# run all A2-sub left hemisphere models
# =========================
all_results = []

for model_name, img_file in img_files.items():
    print("\n==============================")
    print("Running model:", model_name)
    print("File:", img_file)

    if not img_file.exists():
        raise FileNotFoundError(f"Input map not found: {img_file}")

    img_df = pd.read_csv(img_file)
    require_columns(img_df, ["ROI", "t"], str(img_file))
    img_df = normalize_roi_column(img_df)

    keep_cols = ["ROI", "t"]
    for col in ["beta", "p", "p_fdr", "N", "se"]:
        if col in img_df.columns:
            keep_cols.append(col)

    y_all = img_df[keep_cols].copy().rename(columns={"t": "img_t"})
    y_df = align_to_target_order(y_all, ["img_t"], target_order, f"SRPK1 map {model_name}")

    # Merge with fixed x after both maps are aligned to the same left hemisphere order
    df = x_df.merge(y_df, on="ROI", how="inner", validate="one_to_one")
    df = df.dropna(subset=["disease_t", "img_t"]).copy()

    print("Merged LH ROI count:", len(df))
    if len(df) != expected_hemi_n_roi:
        raise ValueError(
            f"Expected {expected_hemi_n_roi} left hemisphere ROIs after merge for model {model_name}, got {len(df)}"
        )

    df["ROI"] = pd.Categorical(df["ROI"], categories=roi_order, ordered=True)
    df = df.sort_values("ROI").reset_index(drop=True)

    y = df["img_t"].to_numpy(dtype=float)

    merged_file = merged_dir / f"a2_sub_input_merged_lh_{model_name}_v2.csv"
    vectors_file = vectors_dir / f"a2_sub_ordered_vectors_lh_{model_name}_v2.csv"
    null_corr_file = null_dist_dir / f"a2_sub_null_distribution_lh_{model_name}_v2.csv"

    df.to_csv(merged_file, index=False)

    pd.DataFrame({
        "ROI": roi_order,
        "disease_t": x,
        "srpk1_t": y
    }).to_csv(vectors_file, index=False)

    r_obs, p_naive = pearsonr(x, y)
    rho_obs, p_spear = spearmanr(x, y)

    null_r = np.array(
        [pearsonr(surrogates[i, :], y)[0] for i in range(surrogates.shape[0])],
        dtype=float
    )

    p_spatial = two_sided_spatial_p(null_r, r_obs)

    pd.DataFrame({"null_r": null_r}).to_csv(null_corr_file, index=False)

    print(f"Observed Pearson r = {r_obs:.6f}")
    print(f"Observed Spearman rho = {rho_obs:.6f}")
    print(f"Spatial-null p = {p_spatial:.6f}")

    all_results.append({
        "comparison": f"disorder_tmap_lh_vs_periph_SRPK1_subcort_tmap_lh_{model_name}",
        "model": model_name,
        "img_file": str(img_file),
        "hemisphere": hemisphere,
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
        "merged_file": str(merged_file.relative_to(out_dir)),
        "vectors_file": str(vectors_file.relative_to(out_dir)),
        "null_corr_file": str(null_corr_file.relative_to(out_dir)),
    })

# =========================
# save summary and metadata
# =========================
result_df = pd.DataFrame(all_results)
result_df.to_csv(result_file, index=False)

meta = {
    "analysis": "A2 subcortical left hemisphere spatial correlation",
    "hemisphere": hemisphere,
    "hemisphere_selection": "Tian S4 full order: first 27 ROIs are right hemisphere, last 27 ROIs are left hemisphere",
    "n_perm_requested": n_perm,
    "n_perm_used": int(surrogates.shape[0]),
    "seed": seed,
    "disease_file": str(disease_file),
    "dist_file": str(dist_file),
    "x_map": "MDD_vs_HC_subcortical_tmap_left_hemisphere",
    "y_map_family": "Peripheral_SRPK1_subcortical_MSN_tmaps_left_hemisphere",
    "primary_y_model": "batch_main: MSN ~ SRPK1 + Batch + age + sex + EDL + eTIV",
    "sensitivity_y_models": [
        "batch_plus_med_history",
        "batch_plus_neutrophils",
        "batch_plus_med_history_neutrophils"
    ],
    "expected_full_n_roi": expected_full_n_roi,
    "expected_hemi_n_roi": expected_hemi_n_roi,
    "n_roi_used": int(len(x)),
    "right_roi_order": rh_order,
    "left_roi_order": roi_order,
    "roi_order_check_file": str(roi_order_check_file.name),
    "shared_null_file": str(null_file.name),
    "img_files": {k: str(v) for k, v in img_files.items()},
    "result_file": str(result_file.name),
    "output_subdirectories": {
        "input_merged": str(merged_dir.name),
        "ordered_vectors": str(vectors_dir.name),
        "null_distributions": str(null_dist_dir.name)
    }
}

with open(meta_file, "w") as f:
    json.dump(meta, f, indent=2)

print("\n==============================")
print("All done.")
print("Shared null file:", null_file)
print("Meta file:", meta_file)
print("Result summary:", result_file)
print("ROI order check:", roi_order_check_file)
print(result_df)
