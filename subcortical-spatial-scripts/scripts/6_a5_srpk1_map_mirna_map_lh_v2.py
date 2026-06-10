#!/usr/bin/env python3
"""Left-hemisphere subcortical spatial correspondence between peripheral SRPK1-associated and miR-139-5p-associated MSN maps using BrainSMASH nulls."""
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
# x = left-hemisphere SRPK1-associated subcortical MSN map
# y = left-hemisphere miR-139-5p-associated subcortical MSN map
#
# Tian S4 subcortical atlas has 54 ROIs in total:
#   first 27 ROIs  = right hemisphere
#   last 27 ROIs   = left hemisphere
# =========================
srpk1_dir = PROJECT_ROOT / "msn_results_subcort" / "srpk1_msn_subcort_assoc_v2"
mir_dir = PROJECT_ROOT / "msn_results_subcort" / "mir1395p_msn_subcort_assoc"

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
out_dir = OUTPUT_ROOT / "a5_sub_srpk1_vs_mir1395p_lh_v2"
out_dir.mkdir(parents=True, exist_ok=True)

result_file = out_dir / "a5_sub_srpk1_vs_mir1395p_lh_results_v2.csv"
meta_file = out_dir / "a5_sub_srpk1_vs_mir1395p_lh_meta_v2.json"
roi_order_check_file = out_dir / "a5_tian_s4_lh_roi_order_check_v2.csv"

merged_dir = out_dir / "input_merged"
vectors_dir = out_dir / "ordered_vectors"
null_dist_dir = out_dir / "null_distributions"
nulls_dir = out_dir / "nulls"
for d in [merged_dir, vectors_dir, null_dist_dir, nulls_dir]:
    d.mkdir(parents=True, exist_ok=True)

# =========================
# Analysis settings
# =========================
n_perm = 10000
seed = 1234
expected_full_n_roi = 54
expected_hemi_n_roi = 27
hemi = "lh"

# =========================
# Helper functions
# =========================
def check_file(path, label):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def require_columns(df, cols, file_label):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {file_label}: {missing}")


def normalize_roi_name(x):
    return str(x).strip().strip("'").strip('"')


def prepare_map(path, value_name, file_label, keep_extra=None):
    if keep_extra is None:
        keep_extra = []

    check_file(path, file_label)
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    require_columns(df, ["ROI", "t"], file_label)

    df["ROI"] = df["ROI"].map(normalize_roi_name)
    df["t"] = pd.to_numeric(df["t"], errors="coerce")

    keep_cols = ["ROI", "t"]
    for c in keep_extra:
        if c in df.columns:
            keep_cols.append(c)

    df = df[keep_cols].rename(columns={"t": value_name}).copy()

    if df["ROI"].duplicated().any():
        dup = df.loc[df["ROI"].duplicated(), "ROI"].tolist()
        raise ValueError(f"Duplicated ROI names in {file_label}: {dup[:20]}")

    return df


def subset_and_order_map(df, value_name, target_order, file_label):
    target_set = set(target_order)
    input_set = set(df["ROI"])

    missing = [roi for roi in target_order if roi not in input_set]
    if missing:
        raise ValueError(f"Missing LH ROIs in {file_label}: {missing}")

    extra = sorted(input_set - target_set)
    if extra:
        print(f"Ignoring {len(extra)} non-LH ROIs in {file_label}.")

    out = df[df["ROI"].isin(target_set)].copy()
    out["ROI"] = pd.Categorical(out["ROI"], categories=target_order, ordered=True)
    out = out.sort_values("ROI").reset_index(drop=True)
    out["ROI"] = out["ROI"].astype(str)

    if len(out) != len(target_order):
        raise ValueError(
            f"Expected {len(target_order)} LH ROIs in {file_label}, got {len(out)}."
        )

    if out[value_name].isna().any():
        bad = out.loc[out[value_name].isna(), "ROI"].tolist()
        raise ValueError(f"Missing numeric values in {file_label}: {bad}")

    return out


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
        if surrogates.shape[0] != n_perm:
            print(f"Warning: cached n_perm={surrogates.shape[0]}, current n_perm={n_perm}")
    else:
        print("Generating nulls:", null_file.name)
        surrogates = gen(n=n_perm)
        print("Surrogates shape:", surrogates.shape)
        np.save(null_file, surrogates)

    return surrogates


def two_sided_spatial_p(null_r, r_obs):
    return (np.sum(np.abs(null_r) >= abs(r_obs)) + 1) / (len(null_r) + 1)

# =========================
# load distance matrix and define LH order
# =========================
check_file(dist_file, "Tian S4 distance matrix")
dist_df = pd.read_csv(dist_file, index_col=0)
dist_df.index = dist_df.index.map(normalize_roi_name)
dist_df.columns = [normalize_roi_name(c) for c in dist_df.columns]

if dist_df.shape != (expected_full_n_roi, expected_full_n_roi):
    raise ValueError(f"Expected 54 x 54 distance matrix, got {dist_df.shape}")

if dist_df.index.tolist() != dist_df.columns.tolist():
    raise ValueError("Distance matrix row and column names are not identical or not in the same order.")

full_order = dist_df.index.tolist()
rh_order = full_order[:expected_hemi_n_roi]
lh_order = full_order[expected_hemi_n_roi:expected_full_n_roi]
target_order = lh_order

if len(rh_order) != expected_hemi_n_roi or len(lh_order) != expected_hemi_n_roi:
    raise ValueError(
        f"Unexpected Tian S4 hemisphere split: RH={len(rh_order)}, LH={len(lh_order)}"
    )

print("Full Tian S4 ROI count:", len(full_order))
print("RH ROI count:", len(rh_order))
print("LH ROI count:", len(lh_order))
print("First LH ROI:", lh_order[0])
print("Last LH ROI:", lh_order[-1])

roi_check = pd.DataFrame({
    "atlas_order": np.arange(1, expected_full_n_roi + 1),
    "ROI": full_order,
    "hemisphere_by_order": ["rh"] * expected_hemi_n_roi + ["lh"] * expected_hemi_n_roi,
    "used_in_lh_analysis": [False] * expected_hemi_n_roi + [True] * expected_hemi_n_roi,
})
roi_check.to_csv(roi_order_check_file, index=False)

D_lh = dist_df.loc[target_order, target_order].to_numpy(dtype=float)
if D_lh.shape != (expected_hemi_n_roi, expected_hemi_n_roi):
    raise ValueError(f"Expected LH distance matrix shape 27 x 27, got {D_lh.shape}")

# =========================
# run matched analyses
# =========================
all_results = []
meta = {
    "analysis": "subcortical_A5_SRPK1_MSN_tmap_vs_miR1395p_MSN_tmap_LH",
    "n_perm_requested": n_perm,
    "seed": seed,
    "dist_file": str(dist_file),
    "roi_order_check_file": str(roi_order_check_file),
    "atlas": "Tian S4 subcortex",
    "total_roi_full_atlas": expected_full_n_roi,
    "hemisphere": "left",
    "n_roi_used": expected_hemi_n_roi,
    "roi_selection": "last 27 ROIs of Tian S4 distance matrix are treated as left hemisphere",
    "rh_order": rh_order,
    "lh_order": lh_order,
    "x_map_family": "Peripheral_SRPK1_subcortical_MSN_tmap_LH",
    "y_map_family": "miR1395p_subcortical_MSN_tmap_LH",
    "primary_x_model": "batch_main: MSN ~ SRPK1 + Batch + age + sex + EDL + eTIV",
    "analysis_version": "v2",
    "note": (
        "This LH subcortical A5 analysis includes primary, medication-adjusted, "
        "neutrophil-adjusted, and medication-plus-neutrophil-adjusted SRPK1 models. "
        "The miR-139-5p side uses the matched primary or medication-adjusted map."
    ),
    "matched_files": {},
    "output_subdirectories": {
        "input_merged": str(merged_dir.name),
        "ordered_vectors": str(vectors_dir.name),
        "null_distributions": str(null_dist_dir.name),
        "nulls": str(nulls_dir.name),
    },
}

for model_name, paths in matched_files.items():
    print("\n==============================")
    print("Running matched LH model:", model_name)

    x_file = paths["x_file"]
    y_file = paths["y_file"]
    srpk1_model_tag = paths["srpk1_model_tag"]
    mirna_model_tag = paths["mirna_model_tag"]

    print("x_file:", x_file)
    print("y_file:", y_file)

    x_df_raw = prepare_map(
        x_file,
        value_name="srpk1_t",
        file_label=f"SRPK1 map {model_name}",
        keep_extra=["beta", "p", "p_fdr", "N", "se"],
    )
    y_df_raw = prepare_map(
        y_file,
        value_name="mir1395p_t",
        file_label=f"miR-139-5p map {model_name}",
        keep_extra=["beta", "p", "p_fdr", "N", "se"],
    )

    x_df = subset_and_order_map(
        x_df_raw,
        value_name="srpk1_t",
        target_order=target_order,
        file_label=f"SRPK1 map {model_name}",
    )
    y_df = subset_and_order_map(
        y_df_raw,
        value_name="mir1395p_t",
        target_order=target_order,
        file_label=f"miR-139-5p map {model_name}",
    )

    df = x_df.merge(
        y_df,
        on="ROI",
        how="inner",
        validate="one_to_one",
        suffixes=("_srpk1", "_mir1395p"),
    )
    df = df.dropna(subset=["srpk1_t", "mir1395p_t"]).copy()

    print("Merged LH ROI count:", len(df))
    print(df.head())

    if len(df) != expected_hemi_n_roi:
        raise ValueError(
            f"Expected {expected_hemi_n_roi} LH ROIs after merge for model {model_name}, got {len(df)}"
        )

    df["ROI"] = pd.Categorical(df["ROI"], categories=target_order, ordered=True)
    df = df.sort_values("ROI").reset_index(drop=True)
    df["ROI"] = df["ROI"].astype(str)

    roi_order = df["ROI"].tolist()
    if roi_order != target_order:
        raise ValueError("Merged ROI order does not match LH target order.")

    x = df["srpk1_t"].to_numpy(dtype=float)
    y = df["mir1395p_t"].to_numpy(dtype=float)

    merged_file = merged_dir / f"a5_sub_input_merged_lh_{model_name}_v2.csv"
    vectors_file = vectors_dir / f"a5_sub_ordered_vectors_lh_{model_name}_v2.csv"
    null_corr_file = null_dist_dir / f"a5_sub_null_distribution_lh_{model_name}_v2.csv"
    null_file = nulls_dir / f"a5_sub_nulls_x_srpk1_lh_{model_name}_v2.npy"

    df.to_csv(merged_file, index=False)

    pd.DataFrame({
        "ROI": roi_order,
        "srpk1_t": x,
        "mir1395p_t": y,
    }).to_csv(vectors_file, index=False)

    r_obs, p_naive = pearsonr(x, y)
    rho_obs, p_spear = spearmanr(x, y)

    surrogates = make_nulls_for_x(
        x=x,
        D=D_lh,
        null_file=null_file,
        seed=seed,
        n_perm=n_perm,
    )

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
        "comparison": f"periph_SRPK1_subcort_MSN_tmap_LH_vs_miR1395p_subcort_MSN_tmap_LH_{model_name}",
        "model": model_name,
        "srpk1_model_tag": srpk1_model_tag,
        "mirna_model_tag": mirna_model_tag,
        "hemisphere": hemi,
        "x_file": str(x_file),
        "y_file": str(y_file),
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
        "null_file": str(null_file.relative_to(out_dir)),
        "null_corr_file": str(null_corr_file.relative_to(out_dir)),
    })

    meta["matched_files"][model_name] = {
        "srpk1_model_tag": srpk1_model_tag,
        "mirna_model_tag": mirna_model_tag,
        "x_file": str(x_file),
        "y_file": str(y_file),
        "null_file": str(null_file.relative_to(out_dir)),
    }

# =========================
# save summary
# =========================
result_df = pd.DataFrame(all_results)
result_df.to_csv(result_file, index=False)

meta["n_perm_used"] = int(all_results[0]["n_perm"]) if all_results else 0
meta["result_file"] = str(result_file.name)

with open(meta_file, "w") as f:
    json.dump(meta, f, indent=2)

print("\n==============================")
print("All done.")
print("ROI order check:", roi_order_check_file)
print("Result summary:", result_file)
print("Meta file:", meta_file)
print(result_df)
