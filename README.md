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
   Computes the intersection of Illumina IDs so that SNP index `j` refers to the same probe for every sample (deterministic alignment).

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

## CLI options

Below is a summary of the main command-line flags. Many of the example values shown (e.g. `r_min = 500`, `p_min = 0.85`) are the ones we used in our experiments; adjust as needed for your dataset.

### Core I/O

| Flag              | Description                                                                                 |
|-------------------|---------------------------------------------------------------------------------------------|
| `--root PATH`     | **Required.** Root directory containing the Red/Green IDAT files (e.g. `/dataC/idat`).     |
| `--out-dir PATH`  | **Required.** Output directory for this run (CSV outputs, timings, plots, feedback, etc.). |

### Feature space (Theta–R)

| Flag                       | Description                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------|
| `--feat NAME`              | Feature space to use. Currently `thetar` is supported (Theta–R transform).                           |
| `--thetar W_THETA W_R`     | Weights applied to θ and r respectively (e.g. `1.0 0.45`) before clustering.                         |

### Gating and quality thresholds

| Flag                      | Description                                                                                                                        |
|---------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| `--r-min FLOAT`           | Intensity floor `r_min`. SNPs below this are treated as low-intensity and more likely to NoCall (e.g. `500`).                     |
| `--p-min FLOAT\|auto`     | Minimum posterior probability τ required to make a genotype call (e.g. `0.85`). Use `auto` to learn τ from FDR control.           |
| `--alpha FLOAT`           | FDR level for automatic `p_min` selection (only used when `--p-min auto`, e.g. `0.005`).                                          |
| `--max-maha FLOAT`        | Maximum allowed Mahalanobis distance from the assigned cluster centre before forcing a NoCall (e.g. `3.0`).                        |

### GMM / EM configuration

| Flag                 | Description                                                                                           |
|----------------------|-------------------------------------------------------------------------------------------------------|
| `--k-max INT`        | Maximum number of mixture components (clusters) per SNP. The model chooses `K ∈ {1, ..., k_max}`.     |
| `--max-iters INT`    | Maximum EM iterations per restart per SNP (e.g. `200`).                                              |
| `--tol FLOAT`        | Convergence tolerance on the EM log-likelihood (e.g. `1e-6`).                                        |
| `--seed INT`         | RNG seed used for initialisation and reproducibility (e.g. `42`).                                    |
| `--restarts INT`     | Number of EM restarts per SNP; best (highest likelihood) solution is kept (e.g. `3`).                |

### Cohort / SNP limiting

| Flag                    | Description                                                                                                           |
|-------------------------|-----------------------------------------------------------------------------------------------------------------------|
| `--first INT`           | Limit to the first `INT` samples discovered under `--root`. `0` means all samples.                                   |
| `--max-snps INT`        | If non-zero, cap the total number of SNPs processed (after intersection) to the first `INT` SNPs.                    |
| `--sample-snps INT`     | If non-zero, randomly sample `INT` SNPs from the cohort instead of taking the first `INT` SNPs (mutually exclusive with `--max-snps`). |

### Plotting

| Flag                      | Description                                                                                                  |
|---------------------------|--------------------------------------------------------------------------------------------------------------|
| `--plot-first-snps INT`   | Number of SNPs to plot for diagnostic scatter plots (e.g. first `N` SNPs or a random subset, depending on build). |

### Feedback / bad-SNP handling

| Flag                        | Description                                                                                                               |
|-----------------------------|---------------------------------------------------------------------------------------------------------------------------|
| `--feedback-in PATH`        | Path to an existing `feedback.csv` file to read per-SNP statistics from (if present).                                    |
| `--feedback-out PATH`       | Path to write updated `feedback.csv` after the run.                                                                      |
| `--bad-snps-in PATH`        | Path to an existing `bad_snps.txt` list to read SNPs to drop or downweight.                                              |
| `--bad-snps-out PATH`       | Path to write the updated `bad_snps.txt` list after the run.                                                             |
| `--nocall-bad-thresh FLOAT` | Threshold on the NoCall rate above which a SNP is treated as “bad” and added to the bad-SNP list (e.g. `0.05`).          |
| `--drop-bad-snps`           | If present, drops SNPs flagged as bad from the cohort instead of keeping them with low confidence.                       |

### GPU vs CPU selection

Whether the CPU or GPU path is used is controlled by:

- **Build-time:**  
  - CPU-only: `cargo build --release`  
  - GPU-enabled: `cargo build --release --features gpu`
- **Runtime (cluster):**  
  - Environment variable `USE_GPU=1` (GPU path on) or `USE_GPU=0` (CPU path only); see the SLURM examples above.

Additional GPU-related environment variables (set in the SLURM script) include:

- `GPU_DEBUG` – extra logging and internal checks when set to `1`.
- `GPU_MIN_N` – minimum per-batch size required before offloading to GPU; below this, the CPU path may be used instead.
- `GPU_FORCE` – when set to `1`, forces GPU use even for smaller workloads.
- `GPU_ARCH` – NVRTC JIT target (e.g. `compute_89` for NVIDIA L4).
- `CUDA_PATH` – CUDA toolkit location (e.g. `/usr/local/cuda-13.0`).

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
- Plots for the first N SNPs 
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
