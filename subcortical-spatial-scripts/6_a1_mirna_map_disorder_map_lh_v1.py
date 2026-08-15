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
# y = left hemisphere miR-139-5p subcortical maps
# =========================
img_dir = OUTPUT_ROOT / "mir1395p_msn_subcort_assoc"

img_files = {
    "main": img_dir / "mir1395p_imaging_map_main.csv",
    "plus_med_history": img_dir / "mir1395p_imaging_map_plus_med_history.csv",
}

# =========================
# output
# =========================
out_dir = OUTPUT_ROOT / "a1_sub_disease_vs_mir1395p_lh"
out_dir.mkdir(parents=True, exist_ok=True)

null_file = out_dir / "a1_sub_nulls_x_mdd_vs_hc_tmap_tian_s4_lh_v1.npy"
meta_file = out_dir / "a1_sub_null_meta_lh_v1.json"
result_file = out_dir / "a1_sub_lh_results_v1.csv"
roi_order_check_file = out_dir / "a1_tian_s4_lh_roi_order_check_v1.csv"

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
expected_n_lh = 27
expected_n_rh = 27

# Tian S4 order rule used here:
# first 27 ROIs are right hemisphere
# last 27 ROIs are left hemisphere

# =========================
# helpers
# =========================
def check_file(path, label):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def normalize_roi_name(x):
    return str(x).strip().strip("'").strip('"')


def require_columns(df, cols, file_label):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {file_label}: {missing}")


def two_sided_spatial_p(null_r, r_obs):
    return (np.sum(np.abs(null_r) >= abs(r_obs)) + 1) / (len(null_r) + 1)


def read_tian_s4_lh_order(dist):
    if dist.shape != (expected_full_n_roi, expected_full_n_roi):
        raise ValueError(
            f"Expected Tian S4 distance matrix shape "
            f"{expected_full_n_roi} x {expected_full_n_roi}, got {dist.shape}"
        )

    if list(dist.index) != list(dist.columns):
        raise ValueError("Distance matrix row names and column names are not identical.")

    full_order = [normalize_roi_name(x) for x in dist.index.tolist()]

    if len(full_order) != expected_full_n_roi:
        raise ValueError(f"Expected {expected_full_n_roi} ROI names, got {len(full_order)}")

    if len(set(full_order)) != expected_full_n_roi:
        dup = pd.Series(full_order)
        dup = dup[dup.duplicated()].tolist()
        raise ValueError(f"Duplicated ROI names in distance matrix: {dup[:20]}")

    rh_order = full_order[:expected_n_rh]
    lh_order = full_order[expected_n_rh:expected_full_n_roi]

    if len(rh_order) != expected_n_rh or len(lh_order) != expected_n_lh:
        raise ValueError(
            f"Unexpected hemisphere split: RH={len(rh_order)}, LH={len(lh_order)}"
        )

    return full_order, rh_order, lh_order


def prepare_lh_map(df, value_col, value_name, target_order, file_label):
    require_columns(df, ["ROI", value_col], file_label)

    out = df[["ROI", value_col]].copy()
    out.columns = ["ROI", value_name]
    out["ROI"] = out["ROI"].map(normalize_roi_name)
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce")

    if out["ROI"].duplicated().any():
        dup = out.loc[out["ROI"].duplicated(), "ROI"].tolist()
        raise ValueError(f"Duplicated ROI in {file_label}: {dup[:20]}")

    target_set = set(target_order)
    input_set = set(out["ROI"])

    missing = [roi for roi in target_order if roi not in input_set]
    extra = sorted(input_set - target_set)

    if missing:
        raise ValueError(f"LH ROI names missing in {file_label}: {missing}")

    lookup = pd.DataFrame({
        "ROI": target_order,
        "atlas_order_lh": np.arange(1, len(target_order) + 1),
        "hemisphere": "lh",
    })

    aligned = lookup.merge(
        out[out["ROI"].isin(target_set)],
        on="ROI",
        how="left",
        validate="one_to_one"
    )

    if aligned[value_name].isna().any():
        bad = aligned.loc[aligned[value_name].isna(), "ROI"].tolist()
        raise ValueError(f"Missing numeric values after LH alignment in {file_label}: {bad}")

    return aligned, extra


# =========================
# read fixed x and distance matrix
# =========================
check_file(disease_file, "disease map")
check_file(dist_file, "Tian S4 distance matrix")

disease_df = pd.read_csv(disease_file)
dist_df = pd.read_csv(dist_file, index_col=0)

disease_df.columns = [str(x).strip() for x in disease_df.columns]
dist_df.index = dist_df.index.map(normalize_roi_name)
dist_df.columns = dist_df.columns.map(normalize_roi_name)

full_order, rh_order, lh_order = read_tian_s4_lh_order(dist_df)
target_order = lh_order

roi_order_check = pd.DataFrame({
    "full_atlas_order": np.arange(1, expected_full_n_roi + 1),
    "ROI": full_order,
    "hemisphere_by_order": ["rh"] * expected_n_rh + ["lh"] * expected_n_lh,
})
roi_order_check.to_csv(roi_order_check_file, index=False)

print("Full Tian S4 ROI count:", len(full_order))
print("RH ROI count:", len(rh_order))
print("LH ROI count:", len(lh_order))
print("First RH ROI:", rh_order[0])
print("Last RH ROI:", rh_order[-1])
print("First LH ROI:", lh_order[0])
print("Last LH ROI:", lh_order[-1])

x_df, disease_extra = prepare_lh_map(
    disease_df,
    "t_group_MDD",
    "disease_t",
    target_order,
    "disease map"
)

print("Ignored extra disease ROIs outside LH order:", len(disease_extra))

roi_order = x_df["ROI"].astype(str).tolist()

dist_lh = dist_df.loc[roi_order, roi_order]
D = dist_lh.to_numpy(dtype=float)

x = x_df["disease_t"].to_numpy(dtype=float)

if len(x) != expected_n_lh:
    raise ValueError(f"Expected {expected_n_lh} LH ROIs for disease map, got {len(x)}")

if D.shape != (expected_n_lh, expected_n_lh):
    raise ValueError(f"Expected LH distance matrix shape {expected_n_lh} x {expected_n_lh}, got {D.shape}")

# =========================
# generate or load shared nulls for fixed x
# =========================
gen = Base(x=x, D=D, seed=seed)

if null_file.exists():
    print("Loading existing shared LH nulls...")
    surrogates = np.load(null_file)
    print("Surrogates shape:", surrogates.shape)

    if surrogates.shape[1] != len(x):
        raise ValueError(
            f"Cached nulls ROI dimension {surrogates.shape[1]} does not match current x length {len(x)}"
        )
    if surrogates.shape[0] != n_perm:
        print(f"Warning: cached n_perm={surrogates.shape[0]}, current n_perm={n_perm}")
else:
    print("Generating shared LH nulls from disease t-map...")
    surrogates = gen(n=n_perm)
    print("Surrogates shape:", surrogates.shape)
    np.save(null_file, surrogates)

# =========================
# run all A1-sub LH models
# =========================
all_results = []

for model_name, img_file in img_files.items():
    print("\n==============================")
    print("Running model:", model_name)
    print("File:", img_file)

    check_file(img_file, f"miR-139-5p map {model_name}")

    img_df = pd.read_csv(img_file)
    img_df.columns = [str(x).strip() for x in img_df.columns]

    y_df, img_extra = prepare_lh_map(
        img_df,
        "t",
        "mir1395p_t",
        target_order,
        f"miR-139-5p map {model_name}"
    )

    print("Ignored extra miR-139-5p ROIs outside LH order:", len(img_extra))

    # Keep optional statistics from the imaging map if present
    optional_cols = []
    for col in ["beta", "p", "p_fdr", "N", "se"]:
        if col in img_df.columns:
            optional_cols.append(col)

    if optional_cols:
        img_optional = img_df[["ROI"] + optional_cols].copy()
        img_optional["ROI"] = img_optional["ROI"].map(normalize_roi_name)
        img_optional = img_optional[img_optional["ROI"].isin(target_order)].copy()
        y_df = y_df.merge(img_optional, on="ROI", how="left", validate="one_to_one")

    df = x_df.merge(
        y_df,
        on=["ROI", "atlas_order_lh", "hemisphere"],
        how="inner",
        validate="one_to_one"
    )

    df = df.dropna(subset=["disease_t", "mir1395p_t"]).copy()

    print("Merged LH ROI count:", len(df))
    if len(df) != expected_n_lh:
        raise ValueError(
            f"Expected {expected_n_lh} LH ROIs after merge for model {model_name}, got {len(df)}"
        )

    df["ROI"] = pd.Categorical(df["ROI"], categories=roi_order, ordered=True)
    df = df.sort_values("ROI").reset_index(drop=True)

    y = df["mir1395p_t"].to_numpy(dtype=float)

    merged_file = merged_dir / f"a1_sub_input_merged_lh_{model_name}_v1.csv"
    vectors_file = vectors_dir / f"a1_sub_ordered_vectors_lh_{model_name}_v1.csv"
    null_corr_file = null_dist_dir / f"a1_sub_null_distribution_lh_{model_name}_v1.csv"

    df.to_csv(merged_file, index=False)

    pd.DataFrame({
        "ROI": roi_order,
        "hemisphere": "lh",
        "disease_t": x,
        "mir1395p_t": y
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
        "comparison": f"subcort_MDD_HC_disorder_tmap_lh_vs_miR1395p_subcort_tmap_lh_{model_name}",
        "model": model_name,
        "img_file": str(img_file),
        "hemisphere": "left",
        "atlas": "Tian S4 subcortex",
        "roi_selection": "last 27 ROIs of Tian S4 distance matrix",
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
    "analysis": "A1 left subcortical spatial correlation between MDD-HC disorder map and miR-139-5p-associated MSN maps",
    "n_perm_requested": n_perm,
    "n_perm_used": int(surrogates.shape[0]),
    "seed": seed,
    "disease_file": str(disease_file),
    "dist_file": str(dist_file),
    "x_map": "MDD_vs_HC_subcortical_tmap_left_hemisphere",
    "y_map_family": "miR1395p_subcortical_MSN_tmaps_left_hemisphere",
    "atlas": "Tian S4 subcortex",
    "total_roi_full_atlas": expected_full_n_roi,
    "n_roi_used": int(len(x)),
    "hemisphere": "left",
    "roi_selection": "last 27 ROIs of Tian S4 distance matrix",
    "roi_order_check_file": str(roi_order_check_file.name),
    "roi_order": roi_order,
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
