# Subcortical spatial correspondence scripts

This folder contains GitHub-ready Python scripts for the subcortical map-to-map analyses in the miRNA-mRNA-MSN project. The scripts reproduce BrainSMASH-based spatial-null analyses for Tian S4 subcortical maps.

## What was changed for GitHub

- Removed personal absolute paths such as `/home/...`.
- Added portable path handling through environment variables.
- Added short English docstrings and clearer section comments.
- Preserved the original statistical workflow, including ROI alignment, Pearson/Spearman correlations, BrainSMASH surrogate maps, and two-sided empirical spatial P values.
- Added `requirements.txt` and `.gitignore`.

## Required input layout

Set `MSN_PROJECT_ROOT` to the project folder that contains the expected input files, for example:

```bash
export MSN_PROJECT_ROOT=/path/to/msn_2026
```

Expected files and folders include:

```text
$MSN_PROJECT_ROOT/
├── msn_results_subcort/
│   ├── tmap_mdd_vs_hc_covs_subcort.csv
│   ├── mir1395p_msn_subcort_assoc/
│   └── srpk1_msn_subcort_assoc_v2/
├── ahba_tian_subcortex_srpk1/
│   └── tian_subcortex_SRPK1_main_named.csv
└── tian_s4_distance_matrix.csv
```

If the Tian S4 distance matrix is stored elsewhere, set:

```bash
export TIAN_S4_DISTANCE_MATRIX=/path/to/tian_s4_distance_matrix.csv
```

To redirect outputs:

```bash
export MSN_OUTPUT_ROOT=/path/to/output_directory
```

If `MSN_OUTPUT_ROOT` is not set, outputs are written under `$MSN_PROJECT_ROOT/msn_results_subcort/`.

## Scripts

| Script | Analysis |
|---|---|
| `6_a1_mirna_map_disorder_map_v1.py` | Bilateral MDD-HC subcortical MSN map vs miR-139-5p-associated subcortical MSN maps |
| `6_a1_mirna_map_disorder_map_lh_v1.py` | Left-hemisphere MDD-HC subcortical MSN map vs miR-139-5p-associated subcortical MSN maps |
| `6_a2_srpk1_map_disorder_map_v2.py` | Bilateral MDD-HC subcortical MSN map vs peripheral SRPK1-associated subcortical MSN maps |
| `6_a2_srpk1_map_disorder_map_lh_v2.py` | Left-hemisphere MDD-HC subcortical MSN map vs peripheral SRPK1-associated subcortical MSN maps |
| `6_a3_ahba_map_disorder_map_v2.py` | Left-hemisphere AHBA-derived SRPK1 expression vs MDD-HC subcortical MSN map |
| `6_a4_ahba_srpk1_expr_peri_map_v5.py` | Left-hemisphere AHBA-derived SRPK1 expression vs peripheral SRPK1-associated subcortical MSN maps |
| `6_a5_srpk1_map_mirna_map_v4.py` | Bilateral peripheral SRPK1-associated vs miR-139-5p-associated subcortical MSN maps |
| `6_a5_srpk1_map_mirna_map_lh_v3.py` | Left-hemisphere peripheral SRPK1-associated vs miR-139-5p-associated subcortical MSN maps |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Example run

```bash
export MSN_PROJECT_ROOT=/path/to/msn_2026
export TIAN_S4_DISTANCE_MATRIX=/path/to/tian_s4_distance_matrix.csv
python scripts/6_a1_mirna_map_disorder_map_v1.py
```

## Notes

- The Tian S4 subcortical atlas is assumed to contain 54 ROIs. Scripts with `_lh_` use the last 27 ROIs of the Tian S4 distance matrix as the left hemisphere, matching the original analysis workflow.
- BrainSMASH null maps are cached as `.npy` files in the script-specific output directory to avoid regeneration.
- Add a `LICENSE` file before making the repository public if others should be allowed to reuse or redistribute the code.
