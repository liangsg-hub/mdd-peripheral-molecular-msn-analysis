# Spatial correspondence analysis scripts

This folder contains Python scripts used for cortical spatial correspondence analyses in the miRNA-mRNA/MSN project.
The scripts compare molecular-associated MSN maps, MDD-related MSN maps, and AHBA-derived SRPK1 expression maps using precomputed spatial null maps.

## Scripts

- `5_a1_mirna_map_disorder_map_v3.py`: bilateral miR-139-5p-associated MSN map versus MDD-related MSN map.
- `5_a1_mirna_map_disorder_map_lh_v1.py`: left-hemisphere miR-139-5p-associated MSN map versus MDD-related MSN map.
- `5_a2_srpk1_map_disorder_map_v5.py`: bilateral SRPK1-associated MSN map versus MDD-related MSN map.
- `5_a2_srpk1_map_disorder_map_lh_v1.py`: left-hemisphere SRPK1-associated MSN map versus MDD-related MSN map.
- `5_a3_ahba_map_disorder_map_v3.py`: AHBA-derived SRPK1 expression map versus MDD-related MSN map.
- `5_a4_ahba_map_periph_srpk1_msn_v5.py`: AHBA-derived SRPK1 expression map versus peripheral SRPK1-associated MSN maps.
- `5_a5_srpk1_map_mirna_map_v6.py`: bilateral SRPK1-associated MSN maps versus miR-139-5p-associated MSN maps.
- `5_a5_srpk1_map_mirna_map_lh_v2.py`: left-hemisphere SRPK1-associated MSN maps versus miR-139-5p-associated MSN maps.

## Configuration

The original personal absolute paths have been replaced with an environment-variable-based project root.
Before running a script, set `MSN_PROJECT_ROOT` to the directory that contains the required input files:

```bash
export MSN_PROJECT_ROOT=/path/to/msn_2026
python 5_a1_mirna_map_disorder_map_v3.py
```

On Windows PowerShell:

```powershell
$env:MSN_PROJECT_ROOT = "D:/path/to/msn_2026"
python 5_a1_mirna_map_disorder_map_v3.py
```

## Required Python packages

See `requirements.txt`. The scripts assume that precomputed DK-308 Burt null maps and ROI-order files are already available under `MSN_PROJECT_ROOT`.

## Notes

These scripts are analysis scripts rather than a standalone software package. They are intended to improve reproducibility by documenting the exact map alignment, spatial-null testing, and output-generation steps used in the manuscript.
