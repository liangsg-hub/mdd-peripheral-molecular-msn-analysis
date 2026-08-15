#!/usr/bin/env python3
"""
Compare the MDD-HC left-cortical MSN t map with the AHBA-derived
left-cortical SRPK1 expression map using Map 1-specific Burt2020 null maps.

Map 1 = MDD-HC left-cortical MSN t map
Map 2 = AHBA-derived left-cortical SRPK1 expression map

Set MSN_PROJECT_ROOT to the project data directory before running.
Set DK308_LH_PARCELLATION if the left-hemisphere DK-308 parcellation is
not stored at the default path.
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
ahba_map_file = (
    PROJECT_ROOT
    / "ahba_dk308_srpk1"
    / "dk308_lh_SRPK1_expression_main.csv"
)
disease_map_file = (
    PROJECT_ROOT
    / "msn_results_cort"
    / "tmap_mdd_vs_hc_covariates.csv"
)
dk308_info_file = PROJECT_ROOT / "dk308_lh_info.csv"
roi_order_file = PROJECT_ROOT / "dk308_roi_order.txt"

out_dir = (
    PROJECT_ROOT
    / "msn_results_cort"
    / "a3_disease_vs_ahba_srpk1_lh_v5"
)
out_dir.mkdir(parents=True, exist_ok=True)

out_merged = out_dir / "a3_input_merged_maps_lh_aligned_v4.csv"
out_result = out_dir / "a3_disease_vs_ahba_srpk1_burt_result_lh_v4.csv"
out_null = out_dir / "a3_null_distribution_lh_v4.csv"
out_meta = out_dir / "a3_disease_vs_ahba_srpk1_meta_lh_v4.json"
roi_order_check_file = out_dir / "a3_dk308_lh_roi_order_check_v4.csv"
info_order_check_file = out_dir / "a3_dk308_lh_info_order_check_v4.csv"

# Map 1-specific Burt2020 nulls and cache metadata
burt_null_file = out_dir / (
    "a3_nulls_x_mdd_vs_hc_tmap_burt2020_dk308_lh_perm10000_v4.npy"
)
burt_null_meta_file = out_dir / (
    "a3_nulls_x_mdd_vs_hc_tmap_burt2020_dk308_lh_perm10000_v4.json"
)


# =========================
# settings
# =========================
roi_col = "ROI"
ahba_val_col = "SRPK1"
disease_val_col = "t_group_MDD"

expected_n_roi = 152
expected_full_n_roi = 308
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

    names = [
        normalize_roi_name(x)
        for x in names
        if normalize_roi_name(x) != ""
    ]

    if len(names) != expected_full_n_roi:
        raise ValueError(
            f"Expected {expected_full_n_roi} ROI names in ROI order file, "
            f"got {len(names)}."
        )

    if len(set(names)) != expected_full_n_roi:
        s = pd.Series(names)
        dup = s[s.duplicated()].tolist()
        raise ValueError(
            f"Duplicated ROI names in ROI order file: {dup[:20]}"
        )

    n_lh = sum(x.startswith("lh_") for x in names)
    n_rh = sum(x.startswith("rh_") for x in names)
    n_other = len(names) - n_lh - n_rh

    if (
        n_lh != expected_n_lh
        or n_rh != expected_n_rh
        or n_other != 0
    ):
        raise ValueError(
            "Unexpected ROI order hemisphere counts: "
            f"lh={n_lh}, rh={n_rh}, other={n_other}."
        )

    lh_names = [x for x in names if x.startswith("lh_")]
    if len(lh_names) != expected_n_roi:
        raise ValueError(
            f"Expected {expected_n_roi} LH ROI names, "
            f"got {len(lh_names)}."
        )

    return names, lh_names


def prepare_disease_map(path):
    check_file(path, "disease map")

    df = pd.read_csv(path)
    df.columns = [str(x).strip() for x in df.columns]

    missing_cols = sorted(
        set([roi_col, disease_val_col]) - set(df.columns)
    )
    if missing_cols:
        raise ValueError(
            f"Missing columns in disease map: {missing_cols}"
        )

    df = df[[roi_col, disease_val_col]].rename(
        columns={disease_val_col: "disease_t"}
    ).copy()

    df[roi_col] = df[roi_col].map(normalize_roi_name)
    df["disease_t"] = pd.to_numeric(
        df["disease_t"],
        errors="coerce",
    )

    if len(df) not in [expected_n_roi, expected_full_n_roi]:
        raise ValueError(
            f"Expected {expected_n_roi} LH rows or "
            f"{expected_full_n_roi} bilateral rows in disease map, "
            f"got {len(df)}."
        )

    if df[roi_col].duplicated().any():
        dup = df.loc[df[roi_col].duplicated(), roi_col].tolist()
        raise ValueError(
            f"Duplicated ROI in disease map: {dup[:20]}"
        )

    return df


def prepare_ahba_map(ahba_path, info_path, lh_roi_order):
    check_file(ahba_path, "AHBA map")
    check_file(info_path, "DK-308 LH info file")

    ahba_df = pd.read_csv(ahba_path)
    info_df = pd.read_csv(info_path)

    ahba_df.columns = [str(x).strip() for x in ahba_df.columns]
    info_df.columns = [str(x).strip() for x in info_df.columns]

    missing_ahba_cols = sorted(
        set([roi_col, ahba_val_col]) - set(ahba_df.columns)
    )
    missing_info_cols = sorted(
        set(["id", "label"]) - set(info_df.columns)
    )

    if missing_ahba_cols:
        raise ValueError(
            f"Missing columns in AHBA map: {missing_ahba_cols}"
        )
    if missing_info_cols:
        raise ValueError(
            f"Missing columns in DK-308 LH info file: {missing_info_cols}"
        )

    info_df["id"] = pd.to_numeric(
        info_df["id"],
        errors="coerce",
    )
    if info_df["id"].isna().any():
        bad = info_df.loc[info_df["id"].isna()].index.tolist()
        raise ValueError(
            f"Non-numeric IDs in DK-308 LH info file at rows: {bad[:20]}"
        )

    if not np.allclose(info_df["id"], np.round(info_df["id"])):
        raise ValueError(
            "DK-308 LH info IDs must be integers."
        )

    info_df["id"] = np.round(info_df["id"]).astype(int)
    info_df["label"] = info_df["label"].astype(str).str.strip()

    if info_df["id"].duplicated().any():
        dup = info_df.loc[
            info_df["id"].duplicated(),
            "id",
        ].tolist()
        raise ValueError(
            f"Duplicated IDs in DK-308 LH info file: {dup[:20]}"
        )

    def make_lh_roi(label):
        label = normalize_roi_name(label)
        if label.startswith("lh_"):
            return label
        return "lh_" + label

    info_df["ROI_name"] = info_df["label"].map(make_lh_roi)

    if info_df["ROI_name"].duplicated().any():
        dup = info_df.loc[
            info_df["ROI_name"].duplicated(),
            "ROI_name",
        ].tolist()
        raise ValueError(
            f"Duplicated ROI names in DK-308 LH info file: {dup[:20]}"
        )

    expected_ids = list(range(1, expected_n_roi + 1))
    observed_ids = sorted(info_df["id"].tolist())

    if observed_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(observed_ids))
        extra = sorted(set(observed_ids) - set(expected_ids))
        raise ValueError(
            "DK-308 LH info IDs must be exactly 1-152. "
            f"Missing: {missing[:20]}; extra: {extra[:20]}"
        )

    info_order = (
        info_df.sort_values("id")["ROI_name"]
        .map(normalize_roi_name)
        .tolist()
    )

    info_order_check = pd.DataFrame({
        "label_id": expected_ids,
        "ROI_from_info": info_order,
        "ROI_from_dk308_order": lh_roi_order,
        "order_match": np.asarray(info_order) == np.asarray(lh_roi_order),
    })
    info_order_check.to_csv(info_order_check_file, index=False)

    if info_order != lh_roi_order:
        mismatches = info_order_check.loc[
            ~info_order_check["order_match"]
        ].head(20)
        raise ValueError(
            "DK-308 LH info order does not match the LH ROI order used by "
            "dk308_roi_order.txt and labels 1-152 of the LH parcellation. "
            f"First mismatches:\n{mismatches.to_string(index=False)}"
        )

    ahba_df = ahba_df[[roi_col, ahba_val_col]].rename(
        columns={ahba_val_col: "ahba_srpk1"}
    ).copy()

    ahba_df[roi_col] = pd.to_numeric(
        ahba_df[roi_col],
        errors="coerce",
    )
    if ahba_df[roi_col].isna().any():
        bad = ahba_df.loc[
            ahba_df[roi_col].isna()
        ].index.tolist()
        raise ValueError(
            f"Non-numeric AHBA ROI IDs at rows: {bad[:20]}"
        )

    if not np.allclose(
        ahba_df[roi_col],
        np.round(ahba_df[roi_col]),
    ):
        raise ValueError(
            "AHBA ROI IDs must be integers."
        )

    ahba_df[roi_col] = np.round(
        ahba_df[roi_col]
    ).astype(int)

    ahba_df = ahba_df.merge(
        info_df[["id", "ROI_name"]],
        left_on=roi_col,
        right_on="id",
        how="left",
        validate="many_to_one",
    )

    if ahba_df["ROI_name"].isna().any():
        bad = ahba_df.loc[
            ahba_df["ROI_name"].isna(),
            roi_col,
        ].tolist()
        raise ValueError(
            f"AHBA ROI IDs without DK-308 LH mapping: {bad[:20]}"
        )

    ahba_df[roi_col] = ahba_df["ROI_name"].map(
        normalize_roi_name
    )
    ahba_df["ahba_srpk1"] = pd.to_numeric(
        ahba_df["ahba_srpk1"],
        errors="coerce",
    )
    ahba_df = ahba_df.drop(columns=["id", "ROI_name"])

    if len(ahba_df) != expected_n_roi:
        raise ValueError(
            f"Expected {expected_n_roi} AHBA ROI rows, "
            f"got {len(ahba_df)}."
        )

    if ahba_df[roi_col].duplicated().any():
        dup = ahba_df.loc[
            ahba_df[roi_col].duplicated(),
            roi_col,
        ].tolist()
        raise ValueError(
            f"Duplicated ROI in AHBA map after mapping: {dup[:20]}"
        )

    return ahba_df


def align_map_to_lh_order(
    df,
    value_name,
    target_roi_order,
    label,
    allow_extra=False,
):
    target_set = set(target_roi_order)
    input_set = set(df[roi_col])

    missing = [
        x for x in target_roi_order
        if x not in input_set
    ]
    extra = sorted(input_set - target_set)

    if missing:
        raise ValueError(
            f"ROI names missing in {label}: {missing[:30]}"
        )

    if extra and not allow_extra:
        raise ValueError(
            f"Extra ROI names in {label}: {extra[:30]}"
        )

    lookup = pd.DataFrame({
        roi_col: target_roi_order,
        "atlas_order": np.arange(
            1,
            len(target_roi_order) + 1,
        ),
        "hemisphere": "lh",
    })

    aligned = lookup.merge(
        df[[roi_col, value_name]],
        on=roi_col,
        how="left",
        validate="one_to_one",
    )

    if aligned[value_name].isna().any():
        bad = aligned.loc[
            aligned[value_name].isna(),
            roi_col,
        ].tolist()
        raise ValueError(
            f"Missing numeric values after alignment in {label}: "
            f"{bad[:30]}"
        )

    if len(aligned) != expected_n_roi:
        raise ValueError(
            f"Expected {expected_n_roi} LH ROIs after alignment "
            f"in {label}, got {len(aligned)}."
        )

    return aligned, extra


def transform_map(x):
    x = np.asarray(x, dtype=float)

    if not use_zscore:
        return x

    x = zscore(x, nan_policy="omit")
    return np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


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
    check_file(
        path,
        "DK-308 LH MNI152 2 mm parcellation",
    )

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
            f"Expected {expected_n_roi} nonzero parcel labels in "
            f"DK-308 LH parcellation, got {len(labels)}."
        )

    expected_labels = np.arange(1, expected_n_roi + 1)

    if not np.array_equal(labels, expected_labels):
        raise ValueError(
            "DK-308 LH parcellation labels must be consecutive integers "
            "1-152 so that the parcellated null-map order matches the "
            "LH ROI order."
        )

    return labels


def build_null_cache_signature(
    x,
    roi_order,
    parcellation_file,
):
    return {
        "map1_vector_sha256": array_sha256(x),
        "roi_order_sha256": array_sha256(
            np.asarray(roi_order, dtype="U")
        ),
        "parcellation_sha256": file_sha256(
            parcellation_file
        ),
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
    check_dk308_lh_parcellation(
        DK308_LH_PARCELLATION
    )

    signature = build_null_cache_signature(
        x,
        roi_order,
        DK308_LH_PARCELLATION,
    )

    cache_is_valid = False

    if (
        burt_null_file.exists()
        and burt_null_meta_file.exists()
    ):
        try:
            with open(
                burt_null_meta_file,
                "r",
                encoding="utf-8",
            ) as f:
                cached_meta = json.load(f)

            cache_is_valid = all(
                cached_meta.get(key) == value
                for key, value in signature.items()
            )
        except (
            OSError,
            ValueError,
            json.JSONDecodeError,
        ):
            cache_is_valid = False

    if cache_is_valid:
        print(
            "Loading cached Map 1-specific "
            "LH Burt2020 nulls..."
        )
        null_maps = np.load(burt_null_file)
    else:
        if (
            burt_null_file.exists()
            or burt_null_meta_file.exists()
        ):
            print(
                "Cached LH Burt2020 nulls do not match "
                "the current Map 1 or settings; regenerating..."
            )
        else:
            print(
                "Generating Map 1-specific LH Burt2020 nulls..."
            )

        null_maps = neuromaps_nulls.burt2020(
            data=x,
            atlas=burt_atlas,
            density=burt_density,
            parcellation=str(DK308_LH_PARCELLATION),
            n_perm=n_perm,
            seed=seed,
            n_proc=n_proc,
        )

        np.save(burt_null_file, null_maps)

        with open(
            burt_null_meta_file,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(signature, f, indent=2)

    print("LH Burt2020 null shape:", null_maps.shape)

    if null_maps.shape != (expected_n_roi, n_perm):
        raise ValueError(
            "Expected LH Burt2020 null shape "
            f"({expected_n_roi}, {n_perm}), "
            f"got {null_maps.shape}."
        )

    if not np.isfinite(null_maps).all():
        raise ValueError(
            "LH Burt2020 null array contains NaN or infinite values."
        )

    return null_maps, signature


# =========================
# read ROI order and maps
# =========================
full_roi_order, lh_roi_order = read_roi_order(
    roi_order_file
)

roi_order_check = pd.DataFrame({
    "atlas_order": np.arange(
        1,
        expected_n_roi + 1,
    ),
    "ROI": lh_roi_order,
    "hemisphere": "lh",
})
roi_order_check.to_csv(
    roi_order_check_file,
    index=False,
)

ahba_df_raw = prepare_ahba_map(
    ahba_map_file,
    dk308_info_file,
    lh_roi_order,
)
disease_df_raw = prepare_disease_map(
    disease_map_file
)

print("Raw AHBA ROI count:", len(ahba_df_raw))
print("Raw disease ROI count:", len(disease_df_raw))

ahba_df, ahba_extra = align_map_to_lh_order(
    ahba_df_raw,
    "ahba_srpk1",
    lh_roi_order,
    "AHBA map",
    allow_extra=False,
)

disease_df, disease_extra = align_map_to_lh_order(
    disease_df_raw,
    "disease_t",
    lh_roi_order,
    "disease map",
    allow_extra=True,
)

print(
    "Ignored extra disease ROIs outside LH order:",
    len(disease_extra),
)


# =========================
# merge aligned maps
# =========================
df = disease_df[[
    roi_col,
    "atlas_order",
    "hemisphere",
    "disease_t",
]].merge(
    ahba_df[[
        roi_col,
        "atlas_order",
        "hemisphere",
        "ahba_srpk1",
    ]],
    on=[
        roi_col,
        "atlas_order",
        "hemisphere",
    ],
    how="inner",
    validate="one_to_one",
)

print("Aligned merged ROI count:", len(df))
print("Aligned first ROI:", df[roi_col].iloc[0])
print("Aligned last ROI:", df[roi_col].iloc[-1])
print(df.head())

if len(df) != expected_n_roi:
    raise ValueError(
        f"Expected {expected_n_roi} LH ROIs after merge, "
        f"got {len(df)}."
    )

df = df.dropna(
    subset=["disease_t", "ahba_srpk1"]
).copy()

print("ROI count after dropna:", len(df))

if len(df) != expected_n_roi:
    raise ValueError(
        f"ROI count after dropna is not {expected_n_roi}."
    )


# =========================
# Map 1 and Map 2 vectors
# =========================
x_raw = df["ahba_srpk1"].to_numpy(dtype=float)
y_raw  = df["disease_t"].to_numpy(dtype=float)

x = transform_map(x_raw)
y = transform_map(y_raw)

df["ahba_srpk1_used"] = x
df["disease_t_used"] = y


if np.std(x) == 0:
    raise ValueError(
        "Map 1 has zero variance; spatial correlation "
        "cannot be computed."
    )

if np.std(y) == 0:
    raise ValueError(
        "Map 2 has zero variance; spatial correlation "
        "cannot be computed."
    )

df.to_csv(out_merged, index=False)


# =========================
# generate/load Map 1-specific LH Burt nulls
# =========================
null_maps, null_signature = generate_or_load_burt_nulls(
    x,
    lh_roi_order,
)


# =========================
# spatial correlation
# =========================
r_obs, p_burt = stats.compare_images(
    x,
    y,
    nulls=null_maps,
)

print(f"Observed r = {r_obs:.6f}")
print(f"Burt-null p = {p_burt:.6f}")


# =========================
# naive Pearson and null summary
# =========================
r_naive = np.corrcoef(x, y)[0, 1]

null_corrs = np.array(
    [
        np.corrcoef(null_maps[:, i], y)[0, 1]
        for i in range(null_maps.shape[1])
    ],
    dtype=float,
)

if not np.isfinite(null_corrs).all():
    raise ValueError(
        "Null correlation distribution contains "
        "non-finite values."
    )

pd.DataFrame({
    "null_r": null_corrs
}).to_csv(
    out_null,
    index=False,
)


# =========================
# result summary
# =========================
result_df = pd.DataFrame([{
    "comparison": (
        "MDD_HC_MSN_tmap_LH_vs_"
        "AHBA_SRPK1_expression_LH"
    ),
    "map_1": "MDD_vs_HC_left_cortical_MSN_tmap",
    "map_2": "AHBA_derived_left_cortical_SRPK1_expression_map",
    "map_1_file": str(disease_map_file),
    "map_2_file": str(ahba_map_file),
    "dk308_info_file": str(dk308_info_file),
    "roi_order_file": str(roi_order_file),
    "dk308_lh_parcellation_file": str(
        DK308_LH_PARCELLATION
    ),
    "burt_null_file": str(burt_null_file),
    "n_roi": len(df),
    "n_lh": int(
        (df["hemisphere"] == "lh").sum()
    ),
    "n_rh": int(
        (df["hemisphere"] == "rh").sum()
    ),
    "aligned_first_roi": df[roi_col].iloc[0],
    "aligned_last_roi": df[roi_col].iloc[-1],
    "use_zscore": use_zscore,
    "r_obs": r_obs,
    "r_naive": r_naive,
    "p_burt": p_burt,
    "n_perm": null_maps.shape[1],
    "null_mean": float(np.mean(null_corrs)),
    "null_sd": float(np.std(null_corrs)),
    "null_min": float(np.min(null_corrs)),
    "null_max": float(np.max(null_corrs)),
    "merged_file": str(out_merged.name),
    "null_corr_file": str(out_null.name),
}])

result_df.to_csv(
    out_result,
    index=False,
)


# =========================
# metadata
# =========================
meta = {
    "analysis": (
        "cortical_A3_MDD_HC_MSN_tmap_LH_vs_"
        "AHBA_SRPK1_expression_LH"
    ),
    "analysis_version": "v4",
    "map_1": "MDD_vs_HC_left_cortical_MSN_tmap",
    "map_2": "AHBA_derived_left_cortical_SRPK1_expression_map",
    "disease_map_file": str(disease_map_file),
    "ahba_map_file": str(ahba_map_file),
    "dk308_info_file": str(dk308_info_file),
    "roi_order_file": str(roi_order_file),
    "roi_order_check_file": str(
        roi_order_check_file
    ),
    "info_order_check_file": str(
        info_order_check_file
    ),
    "dk308_lh_parcellation_file": str(
        DK308_LH_PARCELLATION
    ),
    "burt_null_file": str(burt_null_file),
    "burt_null_meta_file": str(
        burt_null_meta_file
    ),
    "burt_null_signature": null_signature,
    "burt_atlas": burt_atlas,
    "burt_density": burt_density,
    "n_perm_requested": n_perm,
    "n_perm_used": int(
        null_maps.shape[1]
    ),
    "seed": seed,
    "n_proc": n_proc,
    "expected_n_roi": expected_n_roi,
    "expected_n_lh": expected_n_lh,
    "expected_n_rh": 0,
    "use_zscore": use_zscore,
    "note": (
        "Map 1 is the MDD-HC left-cortical MSN t map. "
        "Burt2020 null maps are generated dynamically from Map 1 "
        "using the 152-label DK-308 left-hemisphere parcellation "
        "in MNI152 2 mm space. Map 2 is the AHBA-derived "
        "left-cortical SRPK1 expression map. DK-308 LH info IDs "
        "1-152 are explicitly verified against the LH ROI order "
        "used by dk308_roi_order.txt and labels 1-152 of the "
        "left-hemisphere parcellation."
    ),
}

with open(
    out_meta,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        meta,
        f,
        indent=2,
    )


print("\n==============================")
print("All done.")
print("Saved Map 1-specific LH Burt nulls to:", burt_null_file)
print("Saved null metadata to:", burt_null_meta_file)
print("Saved merged input to:", out_merged)
print("Saved result to:", out_result)
print("Saved null distribution to:", out_null)
print("Saved analysis metadata to:", out_meta)
print("Saved ROI order check to:", roi_order_check_file)
print("Saved info-order check to:", info_order_check_file)
print(result_df)
