#!/usr/bin/env python3
"""
Compare the left-hemisphere AHBA-derived SRPK1 expression map with peripheral SRPK1-associated cortical MSN maps using spatial null models.

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
dk308_info_file = PROJECT_ROOT / "dk308_lh_info.csv"

roi_order_file = PROJECT_ROOT / "dk308_roi_order.txt"

peripheral_dir = PROJECT_ROOT / "msn_results_cort/srpk1_msn_assoc_v2"
peripheral_maps = {
    "batch_main": peripheral_dir / "srpk1_imaging_map_main.csv",
    "batch_plus_med_history": peripheral_dir / "srpk1_imaging_map_plus_med_history.csv",
    "batch_plus_neutrophils": peripheral_dir / "srpk1_imaging_map_plus_neutrophils.csv",
    "batch_plus_med_history_neutrophils": peripheral_dir / "srpk1_imaging_map_plus_med_history_neutrophils.csv",
}

# LEFT-hemisphere Burt nulls, shape should be (152, n_perm)
burt_null_file = PROJECT_ROOT / "burt2020_dk308_lh_perm10000_2mm.npy"

out_dir = PROJECT_ROOT / "msn_results_cort/a4_ahba_srpk1_vs_peripheral_batch"
out_dir.mkdir(parents=True, exist_ok=True)

result_file = out_dir / "a4_ahba_srpk1_vs_periph_srpk1_batch_models_burt_results_v5.csv"
meta_file = out_dir / "a4_ahba_srpk1_vs_periph_srpk1_batch_models_meta_v5.json"
roi_order_check_file = out_dir / "a4_dk308_lh_roi_order_check_v5.csv"


# =========================
# settings
# =========================
roi_col = "ROI"
ahba_val_col = "SRPK1"
peripheral_val_col = "t"
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
        validate="many_to_one",
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


def prepare_peripheral_map(path, label):
    check_file(path, label)

    df = pd.read_csv(path)
    df.columns = [str(x).strip() for x in df.columns]

    missing_cols = sorted(set([roi_col, peripheral_val_col]) - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing columns in {label}: {missing_cols}")

    df = df[[roi_col, peripheral_val_col]].rename(columns={peripheral_val_col: "peripheral_t"}).copy()
    df[roi_col] = df[roi_col].map(normalize_roi_name)
    df["peripheral_t"] = pd.to_numeric(df["peripheral_t"], errors="coerce")

    if df[roi_col].duplicated().any():
        dup = df.loc[df[roi_col].duplicated(), roi_col].tolist()
        raise ValueError(f"Duplicated ROI in {label}: {dup[:20]}")

    return df


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
        validate="one_to_one",
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
# read ROI order, AHBA map and nulls
# =========================
full_roi_order, lh_roi_order = read_roi_order(roi_order_file)

roi_order_check = pd.DataFrame({
    "atlas_order": np.arange(1, expected_n_roi + 1),
    "ROI": lh_roi_order,
    "hemisphere": "lh",
})
roi_order_check.to_csv(roi_order_check_file, index=False)

ahba_df_raw = prepare_ahba_map(ahba_map_file, dk308_info_file)

ahba_df, ahba_extra = align_map_to_order(
    ahba_df_raw,
    "ahba_srpk1",
    lh_roi_order,
    "AHBA map",
    allow_extra=False,
)

check_file(burt_null_file, "LH Burt null file")
nulls = np.load(burt_null_file)
print("Null shape:", nulls.shape)

if nulls.shape[0] != expected_n_roi:
    raise ValueError(
        f"Null first dimension ({nulls.shape[0]}) does not match expected LH ROI count ({expected_n_roi})."
    )


# =========================
# run all peripheral SRPK1 models
# =========================
all_results = []
meta = {
    "analysis": "cortical_a4_AHBA_SRPK1_LH_vs_peripheral_SRPK1_MSN_tmap_LH",
    "ahba_map_file": str(ahba_map_file),
    "dk308_info_file": str(dk308_info_file),
    "roi_order_file": str(roi_order_file),
    "roi_order_check_file": str(roi_order_check_file),
    "burt_null_file": str(burt_null_file),
    "expected_n_roi": expected_n_roi,
    "use_zscore": use_zscore,
    "peripheral_maps": {k: str(v) for k, v in peripheral_maps.items()},
    "note": (
        "AHBA numeric ROI ids are mapped to DK-308 LH ROI names using dk308_lh_info.csv. "
        "AHBA and peripheral SRPK1 MSN maps are aligned to the LH subset of dk308_roi_order.txt "
        "before extracting x/y vectors. This ensures x, y and the LH Burt nulls share the same ROI order. "
        "Peripheral batch_main corresponds to y ~ SRPK1 + Batch + age + sex + EDL + eTIV."
    ),
}

for peripheral_model_tag, peripheral_map_file in peripheral_maps.items():
    print("\n==============================")
    print("Running peripheral model:", peripheral_model_tag)
    print("Peripheral file:", peripheral_map_file)

    peripheral_df_raw = prepare_peripheral_map(peripheral_map_file, "peripheral SRPK1 map")

    peripheral_df, peripheral_extra = align_map_to_order(
        peripheral_df_raw,
        "peripheral_t",
        lh_roi_order,
        "peripheral SRPK1 map",
        allow_extra=True,
    )

    print("Ignored extra peripheral ROIs outside LH order:", len(peripheral_extra))

    df = ahba_df[[roi_col, "atlas_order", "hemisphere", "ahba_srpk1"]].merge(
        peripheral_df[[roi_col, "atlas_order", "hemisphere", "peripheral_t"]],
        on=[roi_col, "atlas_order", "hemisphere"],
        how="inner",
        validate="one_to_one",
    )

    print("Aligned merged ROI count:", len(df))
    print("Aligned first ROI:", df[roi_col].iloc[0])
    print("Aligned last ROI:", df[roi_col].iloc[-1])
    print(df.head())

    if len(df) != expected_n_roi:
        raise ValueError(f"Expected {expected_n_roi} LH ROIs after merge for {peripheral_model_tag}, got {len(df)}")

    df = df.dropna(subset=["ahba_srpk1", "peripheral_t"]).copy()
    print("ROI count after dropna:", len(df))

    if len(df) != expected_n_roi:
        raise ValueError(
            f"ROI count after dropna is not {expected_n_roi} for {peripheral_model_tag}."
        )

    x_raw = df["ahba_srpk1"].to_numpy(dtype=float)
    y_raw = df["peripheral_t"].to_numpy(dtype=float)
    x, y = maybe_zscore(x_raw, y_raw)

    df["ahba_srpk1_used"] = x
    df["peripheral_t_used"] = y

    if nulls.shape[0] != len(df):
        raise ValueError(
            f"Null first dimension ({nulls.shape[0]}) does not match number of LH ROIs ({len(df)})."
        )

    merged_file = out_dir / f"a4_input_merged_maps_lh_srpk1_{peripheral_model_tag}_aligned_v5.csv"
    null_corr_file = out_dir / f"a4_null_distribution_lh_srpk1_{peripheral_model_tag}_aligned_v5.csv"
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
        "comparison": "AHBA_SRPK1_LH_vs_Peripheral_SRPK1_MSN_tmap_LH",
        "peripheral_model_tag": peripheral_model_tag,
        "ahba_map_file": str(ahba_map_file),
        "peripheral_map_file": str(peripheral_map_file),
        "roi_order_file": str(roi_order_file),
        "n_roi": len(df),
        "n_lh": int((df["hemisphere"] == "lh").sum()),
        "aligned_first_roi": df[roi_col].iloc[0],
        "aligned_last_roi": df[roi_col].iloc[-1],
        "ignored_extra_peripheral_roi_count": len(peripheral_extra),
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
