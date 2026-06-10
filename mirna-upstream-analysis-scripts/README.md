# miRNA upstream processing scripts

This repository contains shell scripts for upstream processing of small RNA sequencing data used for mature miRNA quantification. The workflow uses Read 1 FASTQ files, performs read-level quality control, filters reads to the expected miRNA length range, collapses identical reads with miRDeep2 `mapper.pl`, and quantifies mature miRNAs with miRDeep2 `quantifier.pl`.

## Workflow

Step 1. Run FastQC and MultiQC on raw Read 1 FASTQ files.

Step 2. Filter Read 1 FASTQ files with fastp. Reads are retained when their length is between 18 and 30 nt. Adapter trimming is disabled because adapter processing was performed before this step in the original sequencing workflow.

Step 3. Collapse filtered reads using miRDeep2 `mapper.pl`.

Step 4. Quantify mature miRNAs using miRDeep2 `quantifier.pl` with mature and hairpin miRNA reference FASTA files.

Step 5. Copy per-sample `miRNAs_expressed_all_samples_<sample>.csv` output files into a lightweight results directory for downstream count matrix generation.

## Repository structure

```text
miRNA_upstream_processing/
  config.example.sh
  scripts/
    01_fastqc.sh
    02_fastp_filter.sh
    03_collapse_reads_mirdeep2.sh
    04_quantifier_mirdeep2.sh
    05_collect_quantifier_outputs.sh
    run_all.sh
```

## Required software

The following tools should be available in `PATH` before running the scripts.

```text
FastQC
MultiQC
fastp
miRDeep2, including mapper.pl and quantifier.pl
gzip or gunzip
```

The original analysis used miRBase mature and hairpin reference FASTA files for human miRNAs. The species parameter for miRDeep2 `quantifier.pl` is set to `hsa` by default.

## Usage

Copy the example configuration file and edit paths.

```bash
cp config.example.sh config.sh
nano config.sh
```

Run individual steps.

```bash
bash scripts/01_fastqc.sh config.sh
bash scripts/02_fastp_filter.sh config.sh
bash scripts/03_collapse_reads_mirdeep2.sh config.sh
bash scripts/04_quantifier_mirdeep2.sh config.sh
bash scripts/05_collect_quantifier_outputs.sh config.sh
```

Or run the complete workflow.

```bash
bash scripts/run_all.sh config.sh
```

## Expected input file names

Raw Read 1 FASTQ files should be named as follows.

```text
<sample>_R1.fastq.gz
```

The scripts generate cleaned FASTQ files, collapsed FASTA files, per-sample quantifier output folders, log files, and a manifest file recording the collapsed input and quantifier output directory for each sample.

## Notes for public release

All local usernames and absolute paths have been removed. Users should define project-specific paths in `config.sh`, which is intentionally not required for public release. The scripts are intended to support reproducibility of the upstream miRNA processing workflow rather than to store raw sequencing data or private sample metadata.
