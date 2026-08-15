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
# x = left subcortical SRPK1 expression map
# Tian S4 subcortical atlas has 54 ROIs:
#   first 27 ROIs  = right hemisphere
#   last 27 ROIs   = left hemisphere
# =========================
expr_file = PROJECT_ROOT / "ahba_tian_subcortex_srpk1/tian_subcortex_SRPK1_main_named.csv"
dist_file = TIAN_S4_DISTANCE_MATRIX

# =========================
# y maps to test
# y = left peripheral SRPK1-related subcortical MSN maps
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
out_dir = OUTPUT_ROOT / "a4_sub_srpk1_expr_vs_periph_srpk1_batch_models_lh"
out_dir.mkdir(parents=True, exist_ok=True)

result_file = out_dir / "a4_sub_srpk1_expr_lh_vs_periph_srpk1_lh_all_models_results_v5.csv"
meta_file = out_dir / "a4_sub_srpk1_expr_lh_vs_periph_srpk1_lh_all_models_meta_v5.json"
null_file = out_dir / "a4_sub_nulls_x_srpk1_expr_tian_s4_lh_v5.npy"
roi_order_check_file = out_dir / "a4_tian_s4_lh_roi_order_check_v5.csv"

# Optional organized outputs
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
expected_n_rh = 27
expected_n_lh = 27
expected_n_roi = expected_n_lh

# =========================
# helpers
# =========================
def require_columns(df, cols, file_label):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {file_label}: {missing}")


def normalize_roi_name(x):
    return str(x).strip().strip("'").strip('"')


def two_sided_spatial_p(null_r, r_obs):
    return (np.sum(np.abs(null_r) >= abs(r_obs)) + 1) / (len(null_r) + 1)


def check_unique_roi(df, label):
    if df["ROI"].duplicated().any():
        dup = df.loc[df["ROI"].duplicated(), "ROI"].tolist()
        raise ValueError(f"Duplicated ROI names in {label}: {dup[:20]}")


def subset_to_target_order(df, value_cols, target_order, label, allow_full_or_target=True):
    check_unique_roi(df, label)

    target_set = set(target_order)
    input_set = set(df["ROI"])

    if not target_set.issubset(input_set):
        missing = [x for x in target_order if x not in input_set]
        raise ValueError(f"Missing left-hemisphere ROIs in {label}: {missing[:30]}")

    if not allow_full_or_target:
        extra = sorted(input_set - target_set)
        if extra:
            raise ValueError(f"Extra ROI names in {label}: {extra[:30]}")

    out = df.loc[df["ROI"].isin(target_set), ["ROI"] + value_cols].copy()
    out["ROI"] = pd.Categorical(out["ROI"], categories=target_order, ordered=True)
    out = out.sort_values("ROI").reset_index(drop=True)
    out["ROI"] = out["ROI"].astype(str)

    if len(out) != len(target_order):
        raise ValueError(f"Expected {len(target_order)} left-hemisphere ROIs in {label}, got {len(out)}")

    for col in value_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if out[col].isna().any():
            bad = out.loc[out[col].isna(), "ROI"].tolist()
            raise ValueError(f"Missing numeric values in {label}, column {col}: {bad[:30]}")

    return out

# =========================
# read fixed x and distance matrix
# =========================
expr_df = pd.read_csv(expr_file)
dist_df = pd.read_csv(dist_file, index_col=0)

require_columns(expr_df, ["ROI", "SRPK1"], "expr_file")

expr_df.columns = [str(c).strip() for c in expr_df.columns]
expr_df["ROI"] = expr_df["ROI"].map(normalize_roi_name)
dist_df.index = dist_df.index.map(normalize_roi_name)
dist_df.columns = dist_df.columns.map(normalize_roi_name)

# Check full Tian S4 distance matrix
if dist_df.shape != (expected_full_n_roi, expected_full_n_roi):
    raise ValueError(
        f"Expected full Tian S4 distance matrix shape "
        f"({expected_full_n_roi}, {expected_full_n_roi}), got {dist_df.shape}"
    )

if list(dist_df.index) != list(dist_df.columns):
    raise ValueError("Distance matrix row and column ROI orders are not identical.")

full_order = dist_df.index.tolist()
rh_order = full_order[:expected_n_rh]
lh_order = full_order[expected_n_rh:expected_full_n_roi]
target_order = lh_order

if len(rh_order) != expected_n_rh or len(lh_order) != expected_n_lh:
    raise ValueError(
        f"Unexpected Tian S4 hemisphere split: RH={len(rh_order)}, LH={len(lh_order)}"
    )

roi_order_check = pd.DataFrame({
    "atlas_order_full": np.arange(1, expected_full_n_roi + 1),
    "ROI": full_order,
    "hemisphere_by_position": ["right"] * expected_n_rh + ["left"] * expected_n_lh,
    "used_in_analysis": [False] * expected_n_rh + [True] * expected_n_lh,
})
roi_order_check.to_csv(roi_order_check_file, index=False)

print("Full Tian S4 ROI count:", len(full_order))
print("RH ROI count:", len(rh_order))
print("LH ROI count:", len(lh_order))
print("First LH ROI:", lh_order[0])
print("Last LH ROI:", lh_order[-1])

# Prepare left-hemisphere x map
x_df_all = expr_df[["ROI", "SRPK1"]].rename(columns={"SRPK1": "expr_srpk1"}).copy()
x_df = subset_to_target_order(
    x_df_all,
    ["expr_srpk1"],
    target_order,
    "SRPK1 expression map",
    allow_full_or_target=True,
)

roi_order = x_df["ROI"].astype(str).tolist()
dist_lh = dist_df.loc[roi_order, roi_order]

x = x_df["expr_srpk1"].to_numpy(dtype=float)
D = dist_lh.to_numpy(dtype=float)

if len(x) != expected_n_roi:
    raise ValueError(f"Expected {expected_n_roi} left-hemisphere ROIs for expression map, got {len(x)}")

if D.shape != (expected_n_roi, expected_n_roi):
    raise ValueError(f"Expected LH distance matrix shape ({expected_n_roi}, {expected_n_roi}), got {D.shape}")

# =========================
# generate or load shared nulls for fixed left-hemisphere x
# =========================
gen = Base(x=x, D=D, seed=seed)

if null_file.exists():
    print("Loading existing shared LH nulls for SRPK1 expression map")
    surrogates = np.load(null_file)
    print("Surrogates shape:", surrogates.shape)

    if surrogates.shape[1] != len(x):
        raise ValueError(
            f"Cached nulls ROI dimension {surrogates.shape[1]} does not match current x length {len(x)}"
        )
    if surrogates.shape[0] != n_perm:
        print(f"Warning: cached n_perm={surrogates.shape[0]}, current n_perm={n_perm}")
else:
    print("Generating shared LH nulls from SRPK1 expression map")
    surrogates = gen(n=n_perm)
    print("Surrogates shape:", surrogates.shape)
    np.save(null_file, surrogates)

# =========================
# run all A4-sub LH models
# =========================
all_results = []

for model_name, img_file in img_files.items():
    print("\n==============================")
    print("Running model:", model_name)
    print("File:", img_file)

    if not img_file.exists():
        raise FileNotFoundError(f"Input map not found: {img_file}")

    img_df = pd.read_csv(img_file)
    img_df.columns = [str(c).strip() for c in img_df.columns]
    require_columns(img_df, ["ROI", "t"], str(img_file))

    img_df["ROI"] = img_df["ROI"].map(normalize_roi_name)

    keep_cols = ["ROI", "t"]
    for extra_col in ["beta", "p", "p_fdr"]:
        if extra_col in img_df.columns:
            keep_cols.append(extra_col)

    y_df_all = img_df[keep_cols].copy().rename(columns={"t": "img_t"})
    value_cols = [c for c in y_df_all.columns if c != "ROI"]

    y_df = subset_to_target_order(
        y_df_all,
        value_cols,
        target_order,
        f"peripheral SRPK1 subcortical MSN map {model_name}",
        allow_full_or_target=True,
    )

    # Merge with fixed x map in LH atlas order
    df = x_df.merge(y_df, on="ROI", how="inner", validate="one_to_one")
    df = df.dropna(subset=["expr_srpk1", "img_t"]).copy()

    print("Merged LH ROI count:", len(df))
    if len(df) != expected_n_roi:
        raise ValueError(f"Expected {expected_n_roi} LH ROIs after merge for model {model_name}, got {len(df)}")

    df["ROI"] = pd.Categorical(df["ROI"], categories=roi_order, ordered=True)
    df = df.sort_values("ROI").reset_index(drop=True)
    df["ROI"] = df["ROI"].astype(str)

    y = df["img_t"].to_numpy(dtype=float)

    # Save aligned input files
    merged_file = merged_dir / f"a4_sub_input_merged_lh_{model_name}_v5.csv"
    vectors_file = vectors_dir / f"a4_sub_ordered_vectors_lh_{model_name}_v5.csv"
    null_corr_file = null_dist_dir / f"a4_sub_null_distribution_lh_{model_name}_v5.csv"

    df.to_csv(merged_file, index=False)

    pd.DataFrame({
        "ROI": roi_order,
        "hemisphere": "left",
        "expr_srpk1": x,
        "img_t": y,
    }).to_csv(vectors_file, index=False)

    # Observed correlations
    r_obs, p_naive = pearsonr(x, y)
    rho_obs, p_spear = spearmanr(x, y)

    # Correlate shared LH nulls with current LH y map
    null_r = np.array(
        [pearsonr(surrogates[i, :], y)[0] for i in range(surrogates.shape[0])],
        dtype=float,
    )

    p_spatial = two_sided_spatial_p(null_r, r_obs)

    pd.DataFrame({"null_r": null_r}).to_csv(null_corr_file, index=False)

    print(f"Observed Pearson r = {r_obs:.6f}")
    print(f"Observed Spearman rho = {rho_obs:.6f}")
    print(f"Spatial-null p = {p_spatial:.6f}")

    all_results.append({
        "comparison": f"subcort_SRPK1_expr_LH_vs_peripheral_SRPK1_subcort_tmap_LH_{model_name}",
        "model": model_name,
        "hemisphere": "left",
        "roi_selection": "last 27 ROIs of Tian S4 distance matrix",
        "expr_file": str(expr_file),
        "img_file": str(img_file),
        "dist_file": str(dist_file),
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
        "shared_null_file": str(null_file.name),
        "null_corr_file": str(null_corr_file.relative_to(out_dir)),
    })

# =========================
# save summary and metadata
# =========================
result_df = pd.DataFrame(all_results)
result_df.to_csv(result_file, index=False)

meta = {
    "analysis": "A4 left subcortical AHBA SRPK1 expression vs peripheral SRPK1-associated subcortical MSN t-maps",
    "n_perm_requested": n_perm,
    "n_perm_used": int(surrogates.shape[0]),
    "seed": seed,
    "expr_file": str(expr_file),
    "dist_file": str(dist_file),
    "map_x": "Left_subcortical_SRPK1_expression",
    "map_y_family": "Left_peripheral_SRPK1_subcortical_MSN_tmaps",
    "primary_y_model": "batch_main: MSN ~ SRPK1 + Batch + age + sex + EDL + eTIV",
    "sensitivity_y_models": [
        "batch_plus_med_history",
        "batch_plus_neutrophils",
        "batch_plus_med_history_neutrophils",
    ],
    "atlas": "Tian S4 subcortex",
    "full_atlas_n_roi": expected_full_n_roi,
    "hemisphere": "left",
    "n_roi_used": int(len(x)),
    "hemisphere_split_rule": "first 27 ROIs are right hemisphere; last 27 ROIs are left hemisphere",
    "rh_roi_order": rh_order,
    "lh_roi_order": roi_order,
    "roi_order_check_file": str(roi_order_check_file.name),
    "shared_null_file": str(null_file.name),
    "img_files": {k: str(v) for k, v in img_files.items()},
    "result_file": str(result_file.name),
    "output_subdirectories": {
        "input_merged": str(merged_dir.name),
        "ordered_vectors": str(vectors_dir.name),
        "null_distributions": str(null_dist_dir.name),
    },
}

with open(meta_file, "w") as f:
    json.dump(meta, f, indent=2)

print("\n==============================")
print("All done.")
print("Shared LH null file:", null_file)
print("ROI order check file:", roi_order_check_file)
print("Meta file:", meta_file)
print("Result summary:", result_file)
print(result_df)
