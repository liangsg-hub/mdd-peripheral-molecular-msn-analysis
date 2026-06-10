
# MDD peripheral molecular-MSN analysis

This repository contains analysis scripts accompanying the manuscript:

**Peripheral miRNA-mRNA dysregulation and morphometric similarity network organization in major depressive disorder**

The project integrates peripheral miRNA and mRNA expression profiles with cortical and subcortical morphometric similarity network (MSN) measures in major depressive disorder (MDD). The scripts support upstream molecular analyses and spatial correspondence analyses between peripheral molecular-associated MSN maps, MDD-related MSN maps, and AHBA-derived SRPK1 expression maps.

## Repository structure

```text
mdd-peripheral-molecular-msn/
├── README.md
├── requirements.txt
├── Cortical_spatial_scripts/
├── Subcortical_spatial_scripts/
├── miRNA_upstream_analysis_scripts/
└── mRNA_upstream_analysis_scripts/
```

## Folder description

### `Cortical_spatial_scripts/`

Python scripts for cortical map-level spatial correspondence analyses using DK-308 cortical parcels. These scripts test spatial associations among:

* MDD versus healthy control cortical MSN t maps
* miR-139-5p-associated cortical MSN maps
* peripheral SRPK1-associated cortical MSN maps
* AHBA-derived left-hemisphere SRPK1 expression maps

Spatial autocorrelation is accounted for using precomputed Burt null maps and `neuromaps`.

### `Subcortical_spatial_scripts/`

Python scripts for subcortical map-level spatial correspondence analyses using the Tian S4 subcortical atlas. These scripts test spatial associations among:

* MDD versus healthy control subcortical MSN t maps
* miR-139-5p-associated subcortical MSN maps
* peripheral SRPK1-associated subcortical MSN maps
* AHBA-derived SRPK1 expression maps

Spatial autocorrelation is accounted for using BrainSMASH surrogate maps based on the Tian S4 distance matrix.

### `miRNA_upstream_analysis_scripts/`

Scripts for upstream miRNA processing and statistical analyses, including quality control, expression filtering, normalization, differential-expression testing, drug-naive sensitivity analyses, and preparation of miRNA features for downstream integration.

### `mRNA_upstream_analysis_scripts/`

Scripts for upstream mRNA processing and statistical analyses, including quality control, expression filtering, normalization, differential-expression testing, drug-naive sensitivity analyses, neutrophil-adjusted sensitivity analyses, CIBERSORT-derived cell-composition analyses, and preparation of mRNA features for downstream integration.

## Requirements

Python dependencies for the spatial analyses are listed in `requirements.txt`.

Core Python packages include:

```text
numpy
pandas
scipy
neuromaps
brainsmash
```

Upstream molecular analyses were implemented in R. Exact R package versions and analysis settings are described in the manuscript and Supplementary Methods.

## Installation

Clone the repository:

```bash
git clone https://github.com/<username>/mdd-peripheral-molecular-msn.git
cd mdd-peripheral-molecular-msn
```

Create and activate a Python environment:

```bash
conda create -n molecular-msn python=3.10
conda activate molecular-msn
pip install -r requirements.txt
```

## Data organization

Raw sequencing and MRI data are not included in this repository because they contain participant-level information and are subject to ethical and institutional data-use restrictions.

The scripts assume access to processed, de-identified input files, including:

* miRNA expression matrices and differential-expression result files
* mRNA expression matrices and differential-expression result files
* CIBERSORT-derived cell-fraction estimates
* cortical and subcortical MSN regional maps
* DK-308 cortical ROI order files
* Tian S4 subcortical distance matrix
* AHBA-derived SRPK1 expression maps
* precomputed cortical Burt null maps or BrainSMASH distance matrices

For local use, set project-level paths using environment variables rather than hard-coded personal paths.

Example:

```bash
export MSN_PROJECT_ROOT=/path/to/msn_2026
export MSN_OUTPUT_ROOT=/path/to/output
export TIAN_S4_DISTANCE_MATRIX=/path/to/tian_s4_distance_matrix.csv
```

## Running cortical spatial analyses

Example:

```bash
python Cortical_spatial_scripts/5_a1_mirna_map_disorder_map_v3.py
python Cortical_spatial_scripts/5_a2_srpk1_map_disorder_map_v5.py
python Cortical_spatial_scripts/5_a3_ahba_map_disorder_map_v3.py
python Cortical_spatial_scripts/5_a4_ahba_map_periph_srpk1_msn_v5.py
python Cortical_spatial_scripts/5_a5_srpk1_map_mirna_map_v6.py
```

Left-hemisphere sensitivity analyses can be run using the corresponding `_lh_` scripts.

## Running subcortical spatial analyses

Example:

```bash
python Subcortical_spatial_scripts/6_a1_mirna_map_disorder_map_v1.py
python Subcortical_spatial_scripts/6_a2_srpk1_map_disorder_map_v2.py
python Subcortical_spatial_scripts/6_a3_ahba_map_disorder_map_v2.py
python Subcortical_spatial_scripts/6_a4_ahba_srpk1_expr_peri_map_v5.py
python Subcortical_spatial_scripts/6_a5_srpk1_map_mirna_map_v3.py
```

Left-hemisphere sensitivity analyses can be run using the corresponding `_lh_` scripts.

## Main outputs

Most scripts generate:

* aligned input map files
* ordered ROI vector files
* null distributions
* spatial correlation summary tables
* metadata JSON files documenting input files, model tags, ROI order, number of permutations, and random seeds

Outputs are written to the configured output directory and are not tracked by Git by default.

## Reproducibility notes

* Spatial-null analyses use fixed random seeds where applicable.
* Cortical analyses use precomputed Burt null maps aligned to the DK-308 ROI order.
* Subcortical analyses use BrainSMASH surrogate maps generated from the Tian S4 distance matrix.
* ROI alignment checks are saved to output files to document the exact regional order used in each analysis.
* The scripts are intended to document and reproduce the analyses reported in the manuscript, rather than to provide a general-purpose software package.

## Citation

If you use this code, please cite the associated manuscript:

```text
Liang S, et al. Peripheral miRNA-mRNA dysregulation and morphometric similarity network organization in major depressive disorder. 2026.
```

A formal citation will be added after publication.

## License

Please add an appropriate license before public release. If the repository is intended for open reuse, an MIT, BSD, Apache-2.0, or GPL-compatible license may be considered depending on institutional and journal requirements.

## Contact

For questions about the analysis scripts, please contact:

```text
liangsugai@zju.edu.cn
```
