#!/usr/bin/env python3
"""
Compare the left-cortical MDD-HC MSN t map with miR-139-5p-associated
left-cortical MSN t maps using Map 1-specific Burt2020 spatial null maps.

Map 1 = MDD-HC left-cortical MSN t map
Map 2 = miR-139-5p-associated left-cortical MSN t map

Set MSN_PROJECT_ROOT to the project data directory before running.
Set DK308_LH_PARCELLATION to the left-hemisphere DK-308 volumetric
parcellation in MNI152 2 mm space if it is not stored at the default path.
"""

import warnings
warnings.filterwarnings("ignore")

import hashlib
import json
import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import zscore
from neuromaps import nulls as neuromaps_nulls
from neuromaps import stats


# =========================
# user configuration
# =========================
# Example:
#   export MSN_PROJECT_ROOT=/path/to/msn_project
#   export DK308_LH_PARCELLATION=/path/to/dk308_parcellation_lh_2mm.nii.gz
# Optional:
#   export BURT_N_PROC=4
PROJECT_ROOT = Path(
    os.environ.get("MSN_PROJECT_ROOT", "/path/to/msn_project")
).expanduser()

DK308_LH_PARCELLATION = Path(
    os.environ.get(
        "DK308_LH_PARCELLATION",
        PROJECT_ROOT / "dk308_parcellation_lh_2mm.nii.gz",
    )
).expanduser()


# =========================
# input and output paths
# =========================
mirna_dir = PROJECT_ROOT / "msn_results_cort/mir1395p_msn_assoc"

mirna_maps = {
    "main": mirna_dir / "mir1395p_imaging_map_main.csv",
    "plus_med_history": mirna_dir / "mir1395p_imaging_map_plus_med_history.csv",
}

disease_model_tag = "mdd_hc_covariates"
disease_map_file = PROJECT_ROOT / "msn_results_cort/tmap_mdd_vs_hc_covariates.csv"
roi_order_file = PROJECT_ROOT / "dk308_roi_order.txt"

out_dir = PROJECT_ROOT / "msn_results_cort/a1_disease_vs_mir1395p_lh_v2"
out_dir.mkdir(parents=True, exist_ok=True)

result_file = out_dir / "a1_disease_vs_mir1395p_burt_results_lh_v2.csv"
meta_file = out_dir / "a1_disease_vs_mir1395p_meta_lh_v2.json"
roi_order_check_file = out_dir / "a1_dk308_lh_roi_order_check_v2.csv"

# Map 1-specific Burt2020 nulls and cache metadata
burt_null_file = out_dir / (
    "a1_nulls_x_mdd_vs_hc_tmap_burt2020_dk308_lh_perm10000_v2.npy"
)
burt_null_meta_file = out_dir / (
    "a1_nulls_x_mdd_vs_hc_tmap_burt2020_dk308_lh_perm10000_v2.json"
)


# =========================
# settings
# =========================
roi_col = "ROI"
mirna_val_col = "t"
disease_val_col = "t_group_MDD"

expected_full_n_roi = 308
expected_n_roi = 152
expected_n_lh = 152
expected_n_rh = 156

n_perm = 10000
seed = 1234
n_proc = int(os.environ.get("BURT_N_PROC", "1"))
burt_atlas = "MNI152"
burt_density = "2mm"
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

    if len(names) != expected_full_n_roi:
        raise ValueError(
            f"Expected {expected_full_n_roi} ROI names in ROI order file, "
            f"got {len(names)}."
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
            f"Unexpected ROI order hemisphere counts: "
            f"lh={n_lh}, rh={n_rh}, other={n_other}."
        )

    lh_names = [x for x in names if x.startswith("lh_")]
    if len(lh_names) != expected_n_roi:
        raise ValueError(
            f"Expected {expected_n_roi} LH ROI names, got {len(lh_names)}."
        )

    return names, lh_names


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

    if len(df) not in [expected_full_n_roi, expected_n_roi]:
        raise ValueError(
            f"Expected {expected_full_n_roi} or {expected_n_roi} rows in {label}, "
            f"got {len(df)}."
        )

    if df[roi_col].duplicated().any():
        dup = df.loc[df[roi_col].duplicated(), roi_col].tolist()
        raise ValueError(f"Duplicated ROI in {label}: {dup[:20]}")

    return df


def align_map_to_lh_order(df, value_name, lh_roi_order, label, allow_extra=True):
    target_set = set(lh_roi_order)
    input_set = set(df[roi_col])

    missing = [x for x in lh_roi_order if x not in input_set]
    extra = sorted(input_set - target_set)

    if missing:
        raise ValueError(f"LH ROI names missing in {label}: {missing[:30]}")

    if extra and not allow_extra:
        raise ValueError(f"Extra ROI names in {label}: {extra[:30]}")

    lookup = pd.DataFrame({
        roi_col: lh_roi_order,
        "atlas_order": np.arange(1, len(lh_roi_order) + 1),
        "hemisphere": "lh",
    })

    aligned = lookup.merge(
        df[[roi_col, value_name]],
        on=roi_col,
        how="left",
        validate="one_to_one",
    )

    if aligned[value_name].isna().any():
        bad = aligned.loc[aligned[value_name].isna(), roi_col].tolist()
        raise ValueError(
            f"Missing numeric values after LH alignment in {label}: {bad[:30]}"
        )

    if len(aligned) != expected_n_roi:
        raise ValueError(
            f"Expected {expected_n_roi} LH ROIs after alignment in {label}, "
            f"got {len(aligned)}."
        )

    return aligned, extra


def transform_map(x):
    x = np.asarray(x, dtype=float)
    if not use_zscore:
        return x

    x = zscore(x, nan_policy="omit")
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def array_sha256(x):
    arr = np.ascontiguousarray(np.asarray(x))
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def file_sha256(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def check_dk308_lh_parcellation(path):
    check_file(path, "DK-308 LH MNI152 2 mm parcellation")

    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    finite = data[np.isfinite(data)]
    labels = np.unique(finite)
    labels = labels[labels != 0]

    if not np.allclose(labels, np.round(labels)):
        raise ValueError(
            "DK-308 LH parcellation contains non-integer parcel labels."
        )

    labels = np.sort(np.round(labels).astype(int))

    if len(labels) != expected_n_roi:
        raise ValueError(
            f"Expected {expected_n_roi} nonzero parcel labels in DK-308 LH "
            f"parcellation, got {len(labels)}."
        )

    expected_labels = np.arange(1, expected_n_roi + 1)
    if not np.array_equal(labels, expected_labels):
        raise ValueError(
            "DK-308 LH parcellation labels must be consecutive integers 1-152 "
            "so that the parcellated null-map order matches the LH ROI order."
        )

    return labels


def build_null_cache_signature(x, roi_order, parcellation_file):
    return {
        "map1_vector_sha256": array_sha256(x),
        "roi_order_sha256": array_sha256(np.asarray(roi_order, dtype="U")),
        "parcellation_sha256": file_sha256(parcellation_file),
        "parcellation_file": str(parcellation_file),
        "atlas": burt_atlas,
        "density": burt_density,
        "n_perm": n_perm,
        "seed": seed,
        "n_proc": n_proc,
        "use_zscore": use_zscore,
        "expected_n_roi": expected_n_roi,
        "hemisphere": "left",
    }


def generate_or_load_burt_nulls(x, roi_order):
    check_dk308_lh_parcellation(DK308_LH_PARCELLATION)
    signature = build_null_cache_signature(
        x,
        roi_order,
        DK308_LH_PARCELLATION,
    )

    cache_is_valid = False
    if burt_null_file.exists() and burt_null_meta_file.exists():
        try:
            with open(burt_null_meta_file, "r", encoding="utf-8") as f:
                cached_meta = json.load(f)
            cache_is_valid = all(
                cached_meta.get(key) == value
                for key, value in signature.items()
            )
        except (OSError, ValueError, json.JSONDecodeError):
            cache_is_valid = False

    if cache_is_valid:
        print("Loading cached Map 1-specific LH Burt2020 nulls...")
        nulls = np.load(burt_null_file)
    else:
        if burt_null_file.exists() or burt_null_meta_file.exists():
            print(
                "Cached LH Burt2020 nulls do not match the current Map 1 "
                "or settings; regenerating..."
            )
        else:
            print("Generating Map 1-specific LH Burt2020 nulls...")

        nulls = neuromaps_nulls.burt2020(
            data=x,
            atlas=burt_atlas,
            density=burt_density,
            parcellation=str(DK308_LH_PARCELLATION),
            n_perm=n_perm,
            seed=seed,
            n_proc=n_proc,
        )

        np.save(burt_null_file, nulls)
        with open(burt_null_meta_file, "w", encoding="utf-8") as f:
            json.dump(signature, f, indent=2)

    print("LH Burt2020 null shape:", nulls.shape)

    if nulls.shape != (expected_n_roi, n_perm):
        raise ValueError(
            f"Expected LH Burt2020 null shape ({expected_n_roi}, {n_perm}), "
            f"got {nulls.shape}."
        )

    if not np.isfinite(nulls).all():
        raise ValueError("LH Burt2020 null array contains NaN or infinite values.")

    return nulls, signature


# =========================
# read ROI order and fixed Map 1
# =========================
full_roi_order, lh_roi_order = read_roi_order(roi_order_file)

roi_order_check = pd.DataFrame({
    "atlas_order": np.arange(1, expected_n_roi + 1),
    "ROI": lh_roi_order,
    "hemisphere": "lh",
})
roi_order_check.to_csv(roi_order_check_file, index=False)

disease_df_raw = prepare_map(
    disease_map_file,
    disease_val_col,
    "disease_t",
    "disease map",
)

disease_df, disease_extra = align_map_to_lh_order(
    disease_df_raw,
    "disease_t",
    lh_roi_order,
    "disease map",
    allow_extra=True,
)

print("Raw disease ROI count:", len(disease_df_raw))
print("Ignored extra disease ROIs outside LH order:", len(disease_extra))

# Map 1 = MDD-HC left-cortical MSN t map
x_raw = disease_df["disease_t"].to_numpy(dtype=float)
x = transform_map(x_raw)
disease_df["disease_t_used"] = x

if np.std(x) == 0:
    raise ValueError("Map 1 has zero variance; spatial correlation cannot be computed.")

# Generate or load null maps derived specifically from the current Map 1
nulls, null_signature = generate_or_load_burt_nulls(x, lh_roi_order)


# =========================
# run Map 2 miR-139-5p models
# =========================
all_results = []
meta = {
    "analysis": "cortical_A1_MDD_HC_MSN_tmap_vs_miR1395p_MSN_tmap_LH",
    "analysis_version": "v2",
    "map_1": "MDD_vs_HC_left_cortical_MSN_tmap",
    "map_2_family": "miR1395p_associated_left_cortical_MSN_tmaps",
    "disease_model_tag": disease_model_tag,
    "disease_map_file": str(disease_map_file),
    "roi_order_file": str(roi_order_file),
    "roi_order_check_file": str(roi_order_check_file),
    "dk308_lh_parcellation_file": str(DK308_LH_PARCELLATION),
    "burt_null_file": str(burt_null_file),
    "burt_null_meta_file": str(burt_null_meta_file),
    "burt_null_signature": null_signature,
    "burt_atlas": burt_atlas,
    "burt_density": burt_density,
    "n_perm_requested": n_perm,
    "n_perm_used": int(nulls.shape[1]),
    "seed": seed,
    "n_proc": n_proc,
    "expected_full_n_roi": expected_full_n_roi,
    "expected_n_roi": expected_n_roi,
    "expected_n_lh": expected_n_lh,
    "expected_n_rh": 0,
    "use_zscore": use_zscore,
    "mirna_maps": {k: str(v) for k, v in mirna_maps.items()},
    "note": (
        "Map 1 is the MDD-HC left-cortical MSN t map. Burt2020 null maps "
        "are generated dynamically from Map 1 using the 152-label DK-308 "
        "left-hemisphere parcellation in MNI152 2 mm space and reused for "
        "both miR-139-5p Map 2 models. Bilateral input maps are restricted "
        "to the 152 left-hemisphere ROIs before analysis."
    ),
}

for mirna_model_tag, mirna_map_file in mirna_maps.items():
    print("\n==============================")
    print("Running miR-139-5p model:", mirna_model_tag)
    print("Map 1 file:", disease_map_file)
    print("Map 2 file:", mirna_map_file)

    mirna_df_raw = prepare_map(
        mirna_map_file,
        mirna_val_col,
        "mir1395p_t",
        "miR-139-5p map",
    )

    mirna_df, mirna_extra = align_map_to_lh_order(
        mirna_df_raw,
        "mir1395p_t",
        lh_roi_order,
        "miR-139-5p map",
        allow_extra=True,
    )

    print("Raw miR-139-5p ROI count:", len(mirna_df_raw))
    print("Ignored extra miR-139-5p ROIs outside LH order:", len(mirna_extra))

    df = disease_df[[
        roi_col,
        "atlas_order",
        "hemisphere",
        "disease_t",
        "disease_t_used",
    ]].merge(
        mirna_df[[roi_col, "atlas_order", "hemisphere", "mir1395p_t"]],
        on=[roi_col, "atlas_order", "hemisphere"],
        how="inner",
        validate="one_to_one",
    )

    print("Aligned merged LH ROI count:", len(df))
    print("Aligned first LH ROI:", df[roi_col].iloc[0])
    print("Aligned last LH ROI:", df[roi_col].iloc[-1])
    print(df.head())

    if len(df) != expected_n_roi:
        raise ValueError(
            f"Expected {expected_n_roi} LH ROIs after merge for "
            f"{mirna_model_tag}, got {len(df)}."
        )

    df = df.dropna(subset=["disease_t_used", "mir1395p_t"]).copy()
    print("LH ROI count after dropna:", len(df))

    if len(df) != expected_n_roi:
        raise ValueError(
            f"LH ROI count after dropna is not {expected_n_roi} for "
            f"{mirna_model_tag}."
        )

    # Map 1 remains fixed; Map 2 changes by miRNA model
    y_raw = df["mir1395p_t"].to_numpy(dtype=float)
    y = transform_map(y_raw)
    df["mir1395p_t_used"] = y

    if np.std(y) == 0:
        raise ValueError(
            f"Map 2 has zero variance for miRNA model {mirna_model_tag}."
        )

    if nulls.shape[0] != len(df):
        raise ValueError(
            f"Null first dimension ({nulls.shape[0]}) does not match "
            f"number of LH ROIs ({len(df)})."
        )

    merged_file = out_dir / (
        f"a1_input_merged_maps_{disease_model_tag}_vs_mir1395p_"
        f"{mirna_model_tag}_aligned_lh_v2.csv"
    )
    null_corr_file = out_dir / (
        f"a1_null_distribution_{disease_model_tag}_vs_mir1395p_"
        f"{mirna_model_tag}_aligned_lh_v2.csv"
    )
    df.to_csv(merged_file, index=False)

    r_obs, p_burt = stats.compare_images(x, y, nulls=nulls)
    r_naive = np.corrcoef(x, y)[0, 1]

    null_corrs = np.array(
        [np.corrcoef(nulls[:, i], y)[0, 1] for i in range(nulls.shape[1])],
        dtype=float,
    )

    if not np.isfinite(null_corrs).all():
        raise ValueError(
            f"Null correlation distribution contains non-finite values for "
            f"{mirna_model_tag}."
        )

    pd.DataFrame({"null_r": null_corrs}).to_csv(null_corr_file, index=False)

    print(f"Observed r = {r_obs:.6f}")
    print(f"Burt-null p = {p_burt:.6f}")

    all_results.append({
        "comparison": "MDD_HC_MSN_tmap_vs_Peripheral_miR1395p_MSN_tmap_LH",
        "map_1": "MDD_vs_HC_left_cortical_MSN_tmap",
        "map_2": "miR1395p_associated_left_cortical_MSN_tmap",
        "disease_model_tag": disease_model_tag,
        "mirna_model_tag": mirna_model_tag,
        "map_1_file": str(disease_map_file),
        "map_2_file": str(mirna_map_file),
        "roi_order_file": str(roi_order_file),
        "burt_null_file": str(burt_null_file),
        "n_roi": len(df),
        "n_lh": int((df["hemisphere"] == "lh").sum()),
        "n_rh": int((df["hemisphere"] == "rh").sum()),
        "aligned_first_lh_roi": df[roi_col].iloc[0],
        "aligned_last_lh_roi": df[roi_col].iloc[-1],
        "use_zscore": use_zscore,
        "r_obs": r_obs,
        "r_naive": r_naive,
        "p_burt": p_burt,
        "n_perm": nulls.shape[1],
        "null_mean": float(np.mean(null_corrs)),
        "null_sd": float(np.std(null_corrs)),
        "null_min": float(np.min(null_corrs)),
        "null_max": float(np.max(null_corrs)),
        "merged_file": str(merged_file.name),
        "null_corr_file": str(null_corr_file.name),
    })


# =========================
# save summary
# =========================
result_df = pd.DataFrame(all_results)
result_df.to_csv(result_file, index=False)

with open(meta_file, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("\n==============================")
print("All done.")
print("Saved Map 1-specific LH Burt nulls to:", burt_null_file)
print("Saved null metadata to:", burt_null_meta_file)
print("Saved result summary to:", result_file)
print("Saved analysis metadata to:", meta_file)
print("Saved ROI order check to:", roi_order_check_file)
print(result_df)