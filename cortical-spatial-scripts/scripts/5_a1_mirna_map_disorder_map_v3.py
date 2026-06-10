#!/usr/bin/env python3
"""
Compare bilateral miR-139-5p-associated cortical MSN maps with the MDD-related cortical MSN map using spatial null models.

Set MSN_PROJECT_ROOT to the project data directory before running.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
import os
from scipy.stats import zscore
from neuromaps import stats


# =========================
# user configuration
# =========================
# Set this environment variable to the root directory containing the input data.
# Example: export MSN_PROJECT_ROOT=/path/to/msn_2026
PROJECT_ROOT = Path(os.environ.get("MSN_PROJECT_ROOT", "/path/to/msn_2026")).expanduser()


# =========================
# input and output paths
# =========================
mirna_dir = PROJECT_ROOT / "msn_results_cort/mir1395p_msn_assoc"

mirna_maps = {
    "batch_main": mirna_dir / "mir1395p_imaging_map_main.csv",
    "batch_plus_med_history": mirna_dir / "mir1395p_imaging_map_plus_med_history.csv",
}

disease_model_tag = "mdd_hc_covariates"
disease_map_file = PROJECT_ROOT / "msn_results_cort/tmap_mdd_vs_hc_covariates.csv"

roi_order_file = PROJECT_ROOT / "dk308_roi_order.txt"

# precomputed Burt nulls for DK-308
burt_null_file = PROJECT_ROOT / "burt2020_dk308_perm1w_2mm.npy"

out_dir = PROJECT_ROOT / "msn_results_cort/a1_mir1395p_vs_disease"
out_dir.mkdir(parents=True, exist_ok=True)

result_file = out_dir / "a1_mir1395p_vs_disease_batch_models_burt_results_v3.csv"
meta_file = out_dir / "a1_mir1395p_vs_disease_batch_models_meta_v3.json"
roi_order_check_file = out_dir / "a1_dk308_roi_order_check_v3.csv"


# =========================
# settings
# =========================
roi_col = "ROI"
mirna_val_col = "t"
disease_val_col = "t_group_MDD"

expected_n_roi = 308
expected_n_lh = 152
expected_n_rh = 156
use_zscore = False


# =========================
# helpers
# =========================
def check_file(path, label):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def normalize_roi_name(x):
    return str(x).strip().strip("'").strip('"')


def read_roi_order(path):
    check_file(path, "DK-308 ROI order file")
    text = path.read_text(encoding="utf-8", errors="ignore")

    names = re.findall(r"'([^']+)'", text)

    if len(names) == 0:
        names = []
        for line in text.splitlines():
            line = line.strip()
            if line == "" or line.lower() == "roi":
                continue
            parts = [p.strip() for p in line.split(",") if p.strip() != ""]
            names.extend(parts)

    names = [normalize_roi_name(x) for x in names if normalize_roi_name(x) != ""]

    if len(names) != expected_n_roi:
        raise ValueError(
            f"Expected {expected_n_roi} ROI names in ROI order file, got {len(names)}."
        )

    if len(set(names)) != expected_n_roi:
        s = pd.Series(names)
        dup = s[s.duplicated()].tolist()
        raise ValueError(f"Duplicated ROI names in ROI order file: {dup[:20]}")

    n_lh = sum(x.startswith("lh_") for x in names)
    n_rh = sum(x.startswith("rh_") for x in names)
    n_other = len(names) - n_lh - n_rh

    if n_lh != expected_n_lh or n_rh != expected_n_rh or n_other != 0:
        raise ValueError(
            f"Unexpected ROI order hemisphere counts: lh={n_lh}, rh={n_rh}, other={n_other}."
        )

    return names


def prepare_map(path, value_col, value_name, label):
    check_file(path, label)

    df = pd.read_csv(path)
    df.columns = [str(x).strip() for x in df.columns]

    missing_cols = sorted(set([roi_col, value_col]) - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing columns in {label}: {missing_cols}")

    df = df[[roi_col, value_col]].rename(columns={value_col: value_name}).copy()
    df[roi_col] = df[roi_col].map(normalize_roi_name)
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")

    if len(df) != expected_n_roi:
        raise ValueError(f"Expected {expected_n_roi} rows in {label}, got {len(df)}")

    if df[roi_col].duplicated().any():
        dup = df.loc[df[roi_col].duplicated(), roi_col].tolist()
        raise ValueError(f"Duplicated ROI in {label}: {dup[:20]}")

    return df


def align_map_to_order(df, value_name, standard_roi_order, label):
    standard_set = set(standard_roi_order)
    input_set = set(df[roi_col])

    missing = [x for x in standard_roi_order if x not in input_set]
    extra = sorted(input_set - standard_set)

    if len(missing) > 0:
        raise ValueError(f"ROI names missing in {label}: {missing[:30]}")

    if len(extra) > 0:
        raise ValueError(f"Extra ROI names in {label}: {extra[:30]}")

    lookup = pd.DataFrame({
        roi_col: standard_roi_order,
        "atlas_order": np.arange(1, expected_n_roi + 1),
        "hemisphere": ["lh" if x.startswith("lh_") else "rh" for x in standard_roi_order],
    })

    aligned = lookup.merge(
        df[[roi_col, value_name]],
        on=roi_col,
        how="left",
        validate="one_to_one",
    )

    if aligned[value_name].isna().any():
        bad = aligned.loc[aligned[value_name].isna(), roi_col].tolist()
        raise ValueError(f"Missing numeric values after alignment in {label}: {bad[:30]}")

    n_lh = int((aligned["hemisphere"] == "lh").sum())
    n_rh = int((aligned["hemisphere"] == "rh").sum())

    if n_lh != expected_n_lh or n_rh != expected_n_rh:
        raise ValueError(
            f"Unexpected hemisphere counts after alignment in {label}: lh={n_lh}, rh={n_rh}."
        )

    return aligned


def maybe_zscore(x, y):
    if not use_zscore:
        return x, y

    x = zscore(x, nan_policy="omit")
    y = zscore(y, nan_policy="omit")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return x, y


# =========================
# read ROI order, fixed disease map and nulls
# =========================
standard_roi_order = read_roi_order(roi_order_file)

roi_order_check = pd.DataFrame({
    "atlas_order": np.arange(1, expected_n_roi + 1),
    "ROI": standard_roi_order,
    "hemisphere": ["lh" if x.startswith("lh_") else "rh" for x in standard_roi_order],
})
roi_order_check.to_csv(roi_order_check_file, index=False)

disease_df_raw = prepare_map(
    disease_map_file,
    disease_val_col,
    "disease_t",
    "disease map"
)

disease_df = align_map_to_order(
    disease_df_raw,
    "disease_t",
    standard_roi_order,
    "disease map"
)

check_file(burt_null_file, "Burt null file")
nulls = np.load(burt_null_file)
print("Null shape:", nulls.shape)

if nulls.shape[0] != expected_n_roi:
    raise ValueError(
        f"Null first dimension ({nulls.shape[0]}) does not match expected ROI count ({expected_n_roi})."
    )


# =========================
# run miR-139-5p models
# =========================
all_results = []
meta = {
    "analysis": "cortical_A1_miR1395p_MSN_tmap_vs_MDD_HC_MSN_tmap",
    "disease_model_tag": disease_model_tag,
    "disease_map_file": str(disease_map_file),
    "roi_order_file": str(roi_order_file),
    "roi_order_check_file": str(roi_order_check_file),
    "burt_null_file": str(burt_null_file),
    "expected_n_roi": expected_n_roi,
    "expected_n_lh": expected_n_lh,
    "expected_n_rh": expected_n_rh,
    "use_zscore": use_zscore,
    "mirna_maps": {k: str(v) for k, v in mirna_maps.items()},
    "note": (
        "All ROI-level maps are aligned to dk308_roi_order.txt before extracting x/y vectors. "
        "This ensures x, y and the DK-308 Burt nulls share the same ROI order. "
        "The batch_main and batch_plus_med_history miR-139-5p MSN maps are analyzed separately."
    ),
}

for mirna_model_tag, mirna_map_file in mirna_maps.items():
    print("\n==============================")
    print("Running miR-139-5p model:", mirna_model_tag)
    print("miR-139-5p file:", mirna_map_file)

    mirna_df_raw = prepare_map(
        mirna_map_file,
        mirna_val_col,
        "mir1395p_t",
        "miR-139-5p map"
    )

    mirna_df = align_map_to_order(
        mirna_df_raw,
        "mir1395p_t",
        standard_roi_order,
        "miR-139-5p map"
    )

    df = mirna_df[[roi_col, "atlas_order", "hemisphere", "mir1395p_t"]].merge(
        disease_df[[roi_col, "atlas_order", "hemisphere", "disease_t"]],
        on=[roi_col, "atlas_order", "hemisphere"],
        how="inner",
        validate="one_to_one"
    )

    print("Aligned merged ROI count:", len(df))
    print("Aligned first ROI:", df[roi_col].iloc[0])
    print("Aligned last LH ROI:", df[roi_col].iloc[expected_n_lh - 1])
    print("Aligned first RH ROI:", df[roi_col].iloc[expected_n_lh])
    print("Aligned last ROI:", df[roi_col].iloc[-1])
    print(df.head())

    if len(df) != expected_n_roi:
        raise ValueError(f"Expected {expected_n_roi} ROIs after merge for {mirna_model_tag}, got {len(df)}")

    df = df.dropna(subset=["mir1395p_t", "disease_t"]).copy()
    print("ROI count after dropna:", len(df))

    if len(df) != expected_n_roi:
        raise ValueError(
            f"ROI count after dropna is not {expected_n_roi} for {mirna_model_tag}."
        )

    x_raw = df["mir1395p_t"].to_numpy(dtype=float)
    y_raw = df["disease_t"].to_numpy(dtype=float)
    x, y = maybe_zscore(x_raw, y_raw)

    df["mir1395p_t_used"] = x
    df["disease_t_used"] = y

    if nulls.shape[0] != len(df):
        raise ValueError(
            f"Null first dimension ({nulls.shape[0]}) does not match number of ROIs ({len(df)})."
        )

    merged_file = out_dir / f"a1_input_merged_maps_mir1395p_{mirna_model_tag}_vs_{disease_model_tag}_aligned_v3.csv"
    null_corr_file = out_dir / f"a1_null_distribution_mir1395p_{mirna_model_tag}_vs_{disease_model_tag}_aligned_v3.csv"
    df.to_csv(merged_file, index=False)

    r_obs, p_burt = stats.compare_images(x, y, nulls=nulls)
    r_naive = np.corrcoef(x, y)[0, 1]

    null_corrs = np.array(
        [np.corrcoef(nulls[:, i], y)[0, 1] for i in range(nulls.shape[1])],
        dtype=float,
    )
    pd.DataFrame({"null_r": null_corrs}).to_csv(null_corr_file, index=False)

    print(f"Observed r = {r_obs:.6f}")
    print(f"Burt-null p = {p_burt:.6f}")

    all_results.append({
        "comparison": "Peripheral_miR1395p_MSN_tmap_vs_MDD_HC_MSN_tmap",
        "mirna_model_tag": mirna_model_tag,
        "disease_model_tag": disease_model_tag,
        "mirna_map_file": str(mirna_map_file),
        "disease_map_file": str(disease_map_file),
        "roi_order_file": str(roi_order_file),
        "n_roi": len(df),
        "n_lh": int((df["hemisphere"] == "lh").sum()),
        "n_rh": int((df["hemisphere"] == "rh").sum()),
        "aligned_first_roi": df[roi_col].iloc[0],
        "aligned_last_lh_roi": df[roi_col].iloc[expected_n_lh - 1],
        "aligned_first_rh_roi": df[roi_col].iloc[expected_n_lh],
        "aligned_last_roi": df[roi_col].iloc[-1],
        "use_zscore": use_zscore,
        "r_obs": r_obs,
        "r_naive": r_naive,
        "p_burt": p_burt,
        "n_perm": nulls.shape[1],
        "null_mean": float(np.nanmean(null_corrs)),
        "null_sd": float(np.nanstd(null_corrs)),
        "null_min": float(np.nanmin(null_corrs)),
        "null_max": float(np.nanmax(null_corrs)),
        "merged_file": str(merged_file.name),
        "null_corr_file": str(null_corr_file.name),
    })

# =========================
# save summary
# =========================
result_df = pd.DataFrame(all_results)
result_df.to_csv(result_file, index=False)

with open(meta_file, "w") as f:
    json.dump(meta, f, indent=2)

print("\n==============================")
print("All done.")
print("Saved result summary to:", result_file)
print("Saved meta to:", meta_file)
print("Saved ROI order check to:", roi_order_check_file)
print(result_df)
