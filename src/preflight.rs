// src/preflight.rs
use std::path::Path;
use crate::pairs::find_pairs_sorted;
use crate::idat::read_idat_any;
use crate::model::rmin_from_lnxy_sum;

pub fn compute_rmin_like_pipeline(root: &Path, first_pairs: Option<usize>) -> Result<u32, String> {
    let mut pairs = find_pairs_sorted(root);
    if pairs.is_empty() {
        return Err(format!("No IDAT pairs under {}", root.display()));
    }
    if let Some(n) = first_pairs { if pairs.len() > n { pairs.truncate(n); } }

    let mut per_sample = Vec::with_capacity(pairs.len());
    for (gpath, rpath) in pairs {
        let g = read_idat_any(&gpath).map_err(|e| format!("{}: {}", gpath.display(), e))?;
        let r = read_idat_any(&rpath).map_err(|e| format!("{}: {}", rpath.display(), e))?;
        let n = g.means.len().min(r.means.len());
        if n == 0 {
            return Err(format!("empty pair: {} / {}", gpath.display(), rpath.display()));
        }

        let ln_x: Vec<f64> = r.means[..n].iter().map(|&v| (v as f64 + 1.0).ln()).collect();
        let ln_y: Vec<f64> = g.means[..n].iter().map(|&v| (v as f64 + 1.0).ln()).collect();
        let z: Vec<f64> = ln_x.iter().zip(&ln_y).map(|(xr, yg)| xr + yg).collect();

        per_sample.push(rmin_from_lnxy_sum(&z));
    }

    per_sample.sort_unstable();
    Ok(per_sample[per_sample.len() / 2].clamp(300, 3000))
}

