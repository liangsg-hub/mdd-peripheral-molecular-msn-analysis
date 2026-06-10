#!/usr/bin/env python3
"""
Compare the left-hemisphere AHBA-derived SRPK1 expression map with the MDD-related cortical MSN map using spatial null models.

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
ahba_map_file = PROJECT_ROOT / "ahba_dk308_srpk1/dk308_lh_SRPK1_expression_main.csv"
disease_map_file = PROJECT_ROOT / "msn_results_cort/tmap_mdd_vs_hc_covariates.csv"
dk308_info_file = PROJECT_ROOT / "dk308_lh_info.csv"

roi_order_file = PROJECT_ROOT / "dk308_roi_order.txt"

# LEFT hemisphere Burt nulls
burt_null_file = PROJECT_ROOT / "burt2020_dk308_lh_perm10000_2mm.npy"

out_dir = PROJECT_ROOT / "msn_results_cort/a3_ahba_srpk1_disorder_map"
out_dir.mkdir(parents=True, exist_ok=True)

out_merged = out_dir / "a3_input_merged_maps_lh_aligned_v3.csv"
out_result = out_dir / "a3_ahba_srpk1_vs_disease_burt_result_lh_v3.csv"
out_null = out_dir / "a3_null_distribution_lh_v3.csv"
out_meta = out_dir / "a3_ahba_srpk1_vs_disease_meta_lh_v3.json"
roi_order_check_file = out_dir / "a3_dk308_lh_roi_order_check_v3.csv"


# =========================
# settings
# =========================
roi_col = "ROI"
ahba_val_col = "SRPK1"
disease_val_col = "t_group_MDD"
use_zscore = False

expected_n_roi = 152
expected_full_n_roi = 308
expected_n_lh = 152
expected_n_rh = 156


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

    if len(names) != expected_full_n_roi:
        raise ValueError(
            f"Expected {expected_full_n_roi} ROI names in ROI order file, got {len(names)}."
        )

    if len(set(names)) != expected_full_n_roi:
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

    lh_names = [x for x in names if x.startswith("lh_")]
    if len(lh_names) != expected_n_roi:
        raise ValueError(f"Expected {expected_n_roi} LH ROI names, got {len(lh_names)}.")

    return names, lh_names


def prepare_disease_map(path):
    check_file(path, "disease map")
    df = pd.read_csv(path)
    df.columns = [str(x).strip() for x in df.columns]

    missing_cols = sorted(set([roi_col, disease_val_col]) - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing columns in disease map: {missing_cols}")

    df = df[[roi_col, disease_val_col]].rename(columns={disease_val_col: "disease_t"}).copy()
    df[roi_col] = df[roi_col].map(normalize_roi_name)
    df["disease_t"] = pd.to_numeric(df["disease_t"], errors="coerce")

    if df[roi_col].duplicated().any():
        dup = df.loc[df[roi_col].duplicated(), roi_col].tolist()
        raise ValueError(f"Duplicated ROI in disease map: {dup[:20]}")

    return df


def prepare_ahba_map(ahba_path, info_path):
    check_file(ahba_path, "AHBA map")
    check_file(info_path, "DK308 LH info file")

    ahba_df = pd.read_csv(ahba_path)
    info_df = pd.read_csv(info_path)

    ahba_df.columns = [str(x).strip() for x in ahba_df.columns]
    info_df.columns = [str(x).strip() for x in info_df.columns]

    missing_ahba_cols = sorted(set([roi_col, ahba_val_col]) - set(ahba_df.columns))
    missing_info_cols = sorted(set(["id", "label"]) - set(info_df.columns))

    if missing_ahba_cols:
        raise ValueError(f"Missing columns in AHBA map: {missing_ahba_cols}")
    if missing_info_cols:
        raise ValueError(f"Missing columns in DK308 LH info file: {missing_info_cols}")

    ahba_df = ahba_df[[roi_col, ahba_val_col]].rename(columns={ahba_val_col: "ahba_srpk1"}).copy()

    info_df["id"] = pd.to_numeric(info_df["id"], errors="coerce")
    info_df["label"] = info_df["label"].astype(str).str.strip()

    def make_lh_roi(label):
        label = str(label).strip()
        if label.startswith("lh_"):
            return label
        return "lh_" + label

    info_df["ROI_name"] = info_df["label"].map(make_lh_roi)

    ahba_df[roi_col] = pd.to_numeric(ahba_df[roi_col], errors="coerce")

    ahba_df = ahba_df.merge(
        info_df[["id", "ROI_name"]],
        left_on=roi_col,
        right_on="id",
        how="left",
        validate="many_to_one"
    )

    if ahba_df["ROI_name"].isna().any():
        bad = ahba_df.loc[ahba_df["ROI_name"].isna(), roi_col].tolist()
        raise ValueError(f"AHBA ROI ids without DK308 LH mapping: {bad[:20]}")

    ahba_df[roi_col] = ahba_df["ROI_name"].map(normalize_roi_name)
    ahba_df["ahba_srpk1"] = pd.to_numeric(ahba_df["ahba_srpk1"], errors="coerce")
    ahba_df = ahba_df.drop(columns=["id", "ROI_name"])

    if ahba_df[roi_col].duplicated().any():
        dup = ahba_df.loc[ahba_df[roi_col].duplicated(), roi_col].tolist()
        raise ValueError(f"Duplicated ROI in AHBA map after mapping: {dup[:20]}")

    return ahba_df


def align_map_to_order(df, value_name, target_roi_order, label, allow_extra=False):
    target_set = set(target_roi_order)
    input_set = set(df[roi_col])

    missing = [x for x in target_roi_order if x not in input_set]
    extra = sorted(input_set - target_set)

    if len(missing) > 0:
        raise ValueError(f"ROI names missing in {label}: {missing[:30]}")

    if len(extra) > 0 and not allow_extra:
        raise ValueError(f"Extra ROI names in {label}: {extra[:30]}")

    lookup = pd.DataFrame({
        roi_col: target_roi_order,
        "atlas_order": np.arange(1, len(target_roi_order) + 1),
        "hemisphere": ["lh" if x.startswith("lh_") else "rh" for x in target_roi_order],
    })

    aligned = lookup.merge(
        df[[roi_col, value_name]],
        on=roi_col,
        how="left",
        validate="one_to_one"
    )

    if aligned[value_name].isna().any():
        bad = aligned.loc[aligned[value_name].isna(), roi_col].tolist()
        raise ValueError(f"Missing numeric values after alignment in {label}: {bad[:30]}")

    n_lh = int((aligned["hemisphere"] == "lh").sum())
    n_rh = int((aligned["hemisphere"] == "rh").sum())

    if n_lh != expected_n_roi or n_rh != 0:
        raise ValueError(
            f"Unexpected hemisphere counts after LH alignment in {label}: lh={n_lh}, rh={n_rh}."
        )

    return aligned, extra


def maybe_zscore(x, y):
    if not use_zscore:
        return x, y

    x = zscore(x, nan_policy="omit")
    y = zscore(y, nan_policy="omit")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return x, y


# =========================
# read ROI order and maps
# =========================
full_roi_order, lh_roi_order = read_roi_order(roi_order_file)

roi_order_check = pd.DataFrame({
    "atlas_order": np.arange(1, expected_n_roi + 1),
    "ROI": lh_roi_order,
    "hemisphere": "lh",
})
roi_order_check.to_csv(roi_order_check_file, index=False)

ahba_df_raw = prepare_ahba_map(ahba_map_file, dk308_info_file)
disease_df_raw = prepare_disease_map(disease_map_file)

print("Raw AHBA ROI count:", len(ahba_df_raw))
print("Raw disease ROI count:", len(disease_df_raw))

ahba_df, ahba_extra = align_map_to_order(
    ahba_df_raw,
    "ahba_srpk1",
    lh_roi_order,
    "AHBA map",
    allow_extra=False
)

disease_df, disease_extra = align_map_to_order(
    disease_df_raw,
    "disease_t",
    lh_roi_order,
    "disease map",
    allow_extra=True
)

print("Ignored extra disease ROIs outside LH order:", len(disease_extra))


# =========================
# merge aligned maps
# =========================
df = ahba_df[[roi_col, "atlas_order", "hemisphere", "ahba_srpk1"]].merge(
    disease_df[[roi_col, "atlas_order", "hemisphere", "disease_t"]],
    on=[roi_col, "atlas_order", "hemisphere"],
    how="inner",
    validate="one_to_one"
)

print("Aligned merged ROI count:", len(df))
print("Aligned first ROI:", df[roi_col].iloc[0])
print("Aligned last ROI:", df[roi_col].iloc[-1])
print(df.head())

if len(df) != expected_n_roi:
    raise ValueError(f"Expected {expected_n_roi} LH ROIs after merge, got {len(df)}")

df = df.dropna(subset=["ahba_srpk1", "disease_t"]).copy()
print("ROI count after dropna:", len(df))

if len(df) != expected_n_roi:
    raise ValueError(f"ROI count after dropna is not {expected_n_roi}.")


# =========================
# vectors
# =========================
x_raw = df["ahba_srpk1"].to_numpy(dtype=float)
y_raw = df["disease_t"].to_numpy(dtype=float)

x, y = maybe_zscore(x_raw, y_raw)

df["ahba_srpk1_used"] = x
df["disease_t_used"] = y

df.to_csv(out_merged, index=False)


# =========================
# load LH Burt nulls
# =========================
check_file(burt_null_file, "LH Burt null file")
nulls = np.load(burt_null_file)
print("Null shape:", nulls.shape)

if nulls.shape[0] != len(df):
    raise ValueError(
        f"Null first dimension ({nulls.shape[0]}) does not match number of LH ROIs ({len(df)})."
    )


# =========================
# spatial correlation
# nulls correspond to first map x
# =========================
r_obs, p_burt = stats.compare_images(x, y, nulls=nulls)

print(f"Observed r = {r_obs:.6f}")
print(f"Burt-null p = {p_burt:.6f}")


# =========================
# naive Pearson and null summary
# =========================
r_naive = np.corrcoef(x, y)[0, 1]

null_corrs = np.array(
    [np.corrcoef(nulls[:, i], y)[0, 1] for i in range(nulls.shape[1])],
    dtype=float,
)

pd.DataFrame({"null_r": null_corrs}).to_csv(out_null, index=False)

result_df = pd.DataFrame([{
    "comparison": "AHBA_SRPK1_LH_vs_MDD_vs_HC_MSN_tmap_LH",
    "n_roi": len(df),
    "n_lh": int((df["hemisphere"] == "lh").sum()),
    "roi_order_file": str(roi_order_file),
    "aligned_first_roi": df[roi_col].iloc[0],
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
    "merged_file": str(out_merged.name),
    "null_corr_file": str(out_null.name),
}])

result_df.to_csv(out_result, index=False)

meta = {
    "analysis": "cortical_a3_AHBA_SRPK1_LH_vs_MDD_HC_MSN_tmap_LH",
    "ahba_map_file": str(ahba_map_file),
    "disease_map_file": str(disease_map_file),
    "dk308_info_file": str(dk308_info_file),
    "roi_order_file": str(roi_order_file),
    "roi_order_check_file": str(roi_order_check_file),
    "burt_null_file": str(burt_null_file),
    "expected_n_roi": expected_n_roi,
    "use_zscore": use_zscore,
    "note": (
        "AHBA numeric ROI ids are mapped to DK-308 LH ROI names using dk308_lh_info.csv. "
        "Both AHBA and disease maps are then aligned to the LH subset of dk308_roi_order.txt "
        "before extracting x/y vectors. This ensures x, y and the LH Burt nulls share the same ROI order."
    ),
}

with open(out_meta, "w") as f:
    json.dump(meta, f, indent=2)

print("Saved merged input to:", out_merged)
print("Saved result to:", out_result)
print("Saved null distribution to:", out_null)
print("Saved meta to:", out_meta)
print("Saved ROI order check to:", roi_order_check_file)
print("Result summary:")
print(result_df)
