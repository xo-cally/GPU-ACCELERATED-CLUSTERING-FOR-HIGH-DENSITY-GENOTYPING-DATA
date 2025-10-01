# GPU-ACCELERATED-CLUSTERING-FOR-HIGH-DENSITY-GENOTYPING-DATA
This project builds a GPU-accelerated SNP-genotyping pipeline using synthetic, anonymized IDATs from the Sydney Brenner Institute, accessed securely on the Wits Core Cluster. It extracts fluorescence intensities, applies GPU clustering, and benchmarks vs CPU to show faster, scalable, population-scale genotyping.

---

## Getting started (cluster)

We keep the repo checked out at `~/genotyping_pipeline` so SLURM scripts can use a fixed path.

**Clone into that exact folder (recommended):**
```bash
# remove the folder if it's empty, or back it up if it exists
[ -d ~/genotyping_pipeline ] && rmdir ~/genotyping_pipeline 2>/dev/null || true
git clone git@github.com:xo-cally/GPU-ACCELERATED-CLUSTERING-FOR-HIGH-DENSITY-GENOTYPING-DATA.git ~/genotyping_pipeline
cd ~/genotyping_pipeline
