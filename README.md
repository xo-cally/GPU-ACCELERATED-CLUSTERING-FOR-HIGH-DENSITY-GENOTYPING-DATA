# GPU-ACCELERATED-CLUSTERING-FOR-HIGH-DENSITY-GENOTYPING-DATA

This project implements a GPU-accelerated SNP-genotyping pipeline for high-density Illumina arrays. Using synthetic, anonymised IDATs from the Sydney Brenner Institute (mounted on the Wits Core Cluster), it:

- parses Red/Green IDAT files,
- transforms intensities to Theta–R space,
- clusters per-SNP genotype calls with a Gaussian Mixture Model (GMM),
- and benchmarks a CUDA path against a multithreaded CPU baseline.

The goal is to achieve faster population-scale genotyping while keeping the codebase open, inspectable, and reproducible.

---

## Overview

At a high level, the pipeline:

1. **Discovers IDAT pairs**  
   Walks a root directory, finds matching Red/Green IDATs for each sample, and builds a per-sample list of probe IDs and intensities.

2. **Intersects SNP IDs across the cohort**  
   Computes the intersection of Illumina IDs so that SNP index `j` refers to the same SNP for every sample (deterministic alignment).

3. **Transforms intensities to Theta–R space**  
   Converts raw fluorescence `(R, G)` to  
   - `θ = 2·arctan(G/R)/π` and  
   - `r = ln(R + G + 1)`  
   then applies configurable weights `(W_THETA, W_R)`.

4. **Fits per-SNP GMMs**  
   Uses a diagonal-covariance Gaussian Mixture Model with EM and BIC model selection (K ∈ {1, 2, 3}) to cluster genotypes (AA/AB/BB).

5. **Applies gates and feedback**  
   Uses an intensity floor `r_min` and posterior threshold τ (`p_min`) to drive NoCalls and feeds high-nocall SNPs into a persistent bad-SNP list.

6. **Runs on CPU or GPU**  
   - CPU: Rayon-based parallel EM/BIC over SNPs.  
   - GPU: Batched EM/BIC kernels via `cudarc`, with thresholds to decide when GPU is worthwhile.

7. **Writes outputs and timing summaries**  
   Outputs genotype calls, timing CSVs, plots for early SNPs, and feedback files to a timestamped results directory.

---

## Data & environment

- **IDATs are *not* in this repository.**  
  They live on the Wits Core Cluster (e.g. `/dataC/idat`) as synthetic, anonymised data from the Sydney Brenner Institute.

- **Typical environment**  
  - Wits Core Cluster node with:
    - 12–24 vCPUs (Intel Xeon)
    - NVIDIA L4 GPU (24 GB GDDR6)
  - CUDA toolkit (we use `/usr/local/cuda-13.0` on the cluster)
  - Rust toolchain (stable)

If you are running on a different cluster or workstation, ensure that CUDA and Rust are installed and adjust paths accordingly.

---

## Getting started (cluster)

We keep the repo checked out at `~/genotyping_pipeline` so SLURM scripts can rely on a fixed path.

```bash
# remove the folder if it's empty, or back it up if it exists
[ -d ~/genotyping_pipeline ] && rmdir ~/genotyping_pipeline 2>/dev/null || true

git clone git@github.com:xo-cally/GPU-ACCELERATED-CLUSTERING-FOR-HIGH-DENSITY-GENOTYPING-DATA.git \
  ~/genotyping_pipeline

cd ~/genotyping_pipeline
```

---

## Repository layout
```text
src/
  main.rs        - CLI entry point; parses flags and runs a cohort
  lib.rs         - Library root; wires together the internal modules

  idat.rs        - Low-level Illumina IDAT v3 reader (gz/plain) → IdatData
  pairs.rs       - Walks a root folder and finds matching Red/Green IDAT pairs
  model.rs       - Theta–R feature transform, 2D GMM, EM, BIC, prediction, r_min & τ helpers
  cohort.rs      - End-to-end cohort runner: loads samples, finds common SNPs, clusters, writes CSVs
  adaptive.rs    - Persistent “learning” state: feedback CSV + bad-SNP lists
  preflight.rs   - Pre-flight helper to estimate a sensible r_min from a subset of samples
  profiler.rs    - Lightweight timing profiler; writes timings.csv + timings_summary.csv

  gpu.rs         - (feature = "gpu") CUDA kernels + cudarc wrappers for batched EM and BIC

slurm/
  run_genotyping_fixed.sbatch  - SLURM wrapper used for CPU/GPU runs on Wits Core Cluster

logs/    - (created at runtime) SLURM stdout/stderr
results/ - (created at runtime) per-run outputs, timing summaries, plots, feedback
```

---

## Building
### CPU-only build
```bash
cd ~/genotyping_pipeline
cargo build --release
```

### GPU-enabled build
```bash
cd ~/genotyping_pipeline

# Make sure CUDA is on PATH/LD_LIBRARY_PATH (normally done by the SLURM script)
export CUDA_PATH=/usr/local/cuda-13.0
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}"

# Build with GPU feature
cargo build --release --features gpu
```

---

## Basic CLI usage
```bash
./target/release/genotyping_pipeline \
  --root /dataC/idat \
  --out-dir results/test_run \
  --feat thetar \
  --thetar 1.0 0.45 \
  --r-min 500 \
  --p-min 0.85 \
  --k-max 3 \
  --max-iters 200 \
  --tol 1e-6 \
  --seed 42 \
  --restarts 3 \
  --max-maha 3.0 \
  --plot-first-snps 50 \
  --first 100 \
  --max-snps 50000 \
  --feedback-in feedback.csv \
  --feedback-out feedback.csv \
  --bad-snps-in bad_snps.txt \
  --bad-snps-out bad_snps.txt \
  --nocall-bad-thresh 0.05 \
  --drop-bad-snps
```

---

## Running on SLURM (Wits Core Cluster)
For large cohorts, we submit jobs via SLURM.

### Example: GPU timing run (2290 samples):
```bash
sbatch \
  --export=ALL,USE_GPU=1,GPU_DEBUG=0,GPU_FORCE=1,GPU_MIN_N=0,FIRST_N=2290,MAX_SNPS=500000 \
  --job-name=gpu-2290 \
  -p batch \
  --cpus-per-task=24 \
  --mem=128G \
  --gres=gpu:1 \
  --time=72:00:00 \
  slurm/run_genotyping_fixed.sbatch
```

### Example: CPU baseline (no GPU):
```bash
sbatch \
  --export=ALL,USE_GPU=0,GPU_DEBUG=0,GPU_FORCE=0,GPU_MIN_N=0,FIRST_N=2000,MAX_SNPS=500000 \
  --job-name=cpu-2000 \
  -p batch \
  --cpus-per-task=24 \
  --mem=128G \
  --time=72:00:00 \
  slurm/run_genotyping_fixed.sbatch
```

---

## Logs and outputs
SLURM stdout/stderr go to:
```text 
logs/<job-name>_<jobid>.out
logs/<job-name>_<jobid>.err
```

The pipeline itself writes to a timestamped directory:
```text
results/YYYYMMDD_HHMM_fixedR<R_MIN>tau<PMIN><gpu|cpu>_noNorm/
```

This directory typically contains:
- Genotype call CSVs
- timings.csv and timings_summary.csv
- Plots for a subset of SNPs (first N or random N), as configured
- Additional diagnostic files depending on run options

---

## Reproducibility notes
- Fixed seeds & restarts: Runs are controlled via SEED and RESTARTS, which govern GMM initialisation and EM restarts. Keeping these fixed allows fair CPU vs GPU comparisons.
- Feedback loop is persistent: feedback.csv and bad_snps.txt are updated across runs. If you want a “fresh” experiment, either point FEEDBACK_PATH/BAD_SNPS_PATH to new files or delete the old ones.
- Here, `p_min` / `PMIN` (τ) is the minimum posterior probability required to make a genotype call.

---

## Acknowledgements
- Sydney Brenner Institute for synthetic, anonymised SNP array data.
- Wits Core Cluster and system administrators for GPU and compute resources.
- Supervisor (Prof Scott Hazelhurst).
