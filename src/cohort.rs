// src/cohort.rs
use crate::adaptive;
use crate::idat::read_idat_any;
use crate::model::{
    auto_tau_from_posteriors, bic, fit_diag_restarts, maha2_diag, predict, rmin_from_lnxy_sum,
    to_features_from_logs, GmmDiag, ThetaR,
};
use csv::Writer;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};

// ================== Output toggles ==================
const WRITE_CALLS_CSV: bool = false; // <- turn off cohort_calls.csv

// =============== PMMode / Config ===============
pub enum PMMode {
    Fixed(f64),
    Auto { alpha: f64 },
}

pub struct CohortCfg {
    pub feature_space: ThetaR,
    pub r_min: u32, // 0 => auto median of per-sample 1D-GMM thresholds on ln(R+G+1)
    pub p_mode: PMMode,
    pub max_maha: f64,
    pub k_max: usize,
    pub max_iters: usize,
    pub tol: f64,
    pub seed: u64,
    pub restarts: usize,
    pub plot_first_snps: usize,

    // SNP limiting
    pub snp_first_n: Option<usize>,
    pub snp_sample_n: Option<usize>,

    // Learning / bad-SNP persistence
    pub feedback_in: Option<PathBuf>,
    pub feedback_out: Option<PathBuf>,
    pub bad_snps_in: Option<PathBuf>,
    pub bad_snps_out: Option<PathBuf>,
    pub drop_bad_snps: bool,
    pub nocall_bad_thresh: f64,

    // Debugging
    pub debug_snp: Option<u32>,
}

// =============== Sample cache ===============
#[derive(Clone)]
struct SampleData {
    name: String,
    ids: Vec<u32>,
    x_red: Vec<u16>,
    y_grn: Vec<u16>,
}

/// Parallel load of samples (no per-SNP position maps here).
fn load_samples(pairs: &[(PathBuf, PathBuf)]) -> Result<Vec<SampleData>, String> {
    // Preserve the original order of `pairs` by carrying the index through.
    let mut temp: Vec<(usize, SampleData)> = pairs
        .par_iter()
        .enumerate()
        .map(|(idx, (gpath, rpath))| -> Result<(usize, SampleData), String> {
            let g = read_idat_any(gpath).map_err(|e| format!("{}: {}", gpath.display(), e))?;
            let r = read_idat_any(rpath).map_err(|e| format!("{}: {}", rpath.display(), e))?;
            let n = g.means.len().min(r.means.len());
            if n == 0 {
                return Err("empty pair".into());
            }
            let stem = gpath.file_stem().and_then(|s| s.to_str()).unwrap_or("sample");
            let name = stem.strip_suffix("_Grn").unwrap_or(stem).to_string();
            Ok((
                idx,
                SampleData {
                    name,
                    ids: g.illumina_ids[..n].to_vec(),
                    x_red: r.means[..n].to_vec(),
                    y_grn: g.means[..n].to_vec(),
                },
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;

    temp.sort_by_key(|(i, _)| *i);
    Ok(temp.into_iter().map(|(_, s)| s).collect())
}

/// Intersection of sorted ID lists using k-way merge (two-pointer per step).
fn common_ids(samples: &[SampleData]) -> Vec<u32> {
    if samples.is_empty() {
        return vec![];
    }
    // Start with first sample's sorted ids
    let mut acc: Vec<u32> = samples[0].ids.clone();
    for s in &samples[1..] {
        let (a, b) = (&acc, &s.ids);
        let mut i = 0usize;
        let mut j = 0usize;
        let mut out = Vec::with_capacity(a.len().min(b.len()));
        while i < a.len() && j < b.len() {
            use std::cmp::Ordering::*;
            match a[i].cmp(&b[j]) {
                Less => i += 1,
                Greater => j += 1,
                Equal => {
                    out.push(a[i]);
                    i += 1;
                    j += 1;
                }
            }
        }
        acc = out;
        if acc.is_empty() {
            break;
        }
    }
    acc
}

/// Build, for each sample, a compact vector of positions aligned to `ids`.
/// Uses binary_search on each sample’s `ids` (Illumina IDs are typically sorted).
fn build_positions_for_ids(samples: &[SampleData], ids: &[u32]) -> Vec<Vec<usize>> {
    samples
        .par_iter()
        .map(|s| {
            let mut pos = Vec::with_capacity(ids.len());
            for &id in ids {
                let p = s
                    .ids
                    .binary_search(&id)
                    .expect("id from intersection not found in sample.ids");
                pos.push(p);
            }
            pos
        })
        .collect()
}

// =============== K=1 baseline & choose_k (CPU fallback) ===============
fn fit_k1(points: &[(f64, f64)]) -> GmmDiag {
    let n = points.len().max(1);
    let (mut mx, mut my) = (0.0, 0.0);
    for &(x, y) in points {
        mx += x;
        my += y;
    }
    mx /= n as f64;
    my /= n as f64;

    let (mut vx, mut vy) = (0.0, 0.0);
    for &(x, y) in points {
        vx += (x - mx) * (x - mx);
        vy += (y - my) * (y - my);
    }
    vx = (vx / n as f64).max(1e-2);
    vy = (vy / n as f64).max(1e-2);

    let mut ll = 0.0;
    for &(x, y) in points {
        let dx = x - mx;
        let dy = y - my;
        ll += -0.5 * (dx * dx / vx + dy * dy / vy)
            - 0.5 * ((vx * vy).ln() + 2.0 * std::f64::consts::PI.ln());
    }

    GmmDiag {
        k: 1,
        weights: vec![1.0],
        means: vec![(mx, my)],
        vars: vec![(vx, vy)],
        loglik: ll,
    }
}

fn choose_k(
    points: &[(f64, f64)],
    kmax: usize,
    max_it: usize,
    tol: f64,
    seed: u64,
    restarts: usize,
) -> GmmDiag {
    let g1 = fit_k1(points);
    if kmax <= 1 {
        return g1;
    }
    let g2 = fit_diag_restarts(points, 2, max_it, tol, seed, restarts);
    if kmax == 2 {
        let (b1, b2) = (bic(&g1, points.len()), bic(&g2, points.len()));
        return if b1 <= b2 { g1 } else { g2 };
    }
    let g3 = fit_diag_restarts(points, 3, max_it, tol, seed, restarts);
    let (b1, b2, b3) = (
        bic(&g1, points.len()),
        bic(&g2, points.len()),
        bic(&g3, points.len()),
    );
    if b1 <= b2 && b1 <= b3 {
        g1
    } else if b2 <= b3 {
        g2
    } else {
        g3
    }
}

/// Strict component -> genotype mapping
fn map_labels_for_snp_strict(means: &[(f64, f64)], w_theta: f64) -> Vec<&'static str> {
    let k = means.len();
    let mut names = vec![""; k];

    let wth = w_theta.max(1e-9);
    if k == 1 {
        let t = means[0].0 / wth;
        names[0] = if t <= 0.33 { "BB" } else if t >= 0.67 { "AA" } else { "AB" };
        return names;
    }

    let mut idx: Vec<(usize, f64)> = means
        .iter()
        .enumerate()
        .map(|(i, m)| (i, m.0 / wth))
        .collect();
    idx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    if k == 2 {
        names[idx[0].0] = "BB";
        names[idx[1].0] = "AA";
        return names;
    }

    names[idx[0].0] = "BB";
    names[idx[1].0] = "AB";
    names[idx[2].0] = "AA";
    names
}

#[inline]
fn safe_lab_index(snp_id: u32, jlab: usize, k: usize) -> usize {
    if jlab < k {
        jlab
    } else {
        eprintln!("FIXUP SNP {}: lab={} but k={} — forcing lab=0", snp_id, jlab, k);
        0
    }
}

#[inline]
fn theta_guard_ok(
    means: &[(f64, f64)],
    w_theta: f64,
    label: &str,
    theta: f64,
    band: f64,
) -> bool {
    let k = means.len();
    if k <= 1 {
        return true;
    }
    let wth = w_theta.max(1e-9);
    let mut t: Vec<f64> = means.iter().map(|m| m.0 / wth).collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());

    match k {
        2 => {
            let m = 0.5 * (t[0] + t[1]);
            match label {
                "BB" => theta <= m + band,
                "AA" => theta >= m - band,
                "AB" => (theta >= m - band) && (theta <= m + band),
                _ => true,
            }
        }
        _ => {
            let m01 = 0.5 * (t[0] + t[1]);
            let m12 = 0.5 * (t[1] + t[2]);
            match label {
                "BB" => theta <= m01 + band,
                "AB" => (theta >= m01 - band) && (theta <= m12 + band),
                "AA" => theta >= m12 - band,
                _ => true,
            }
        }
    }
}

// ---- small helpers for GPU path ----
#[inline]
fn use_gpu_runtime() -> bool {
    std::env::var("USE_GPU").as_deref() != Ok("0")
}

#[cfg(feature = "gpu")]
fn gpu_debug() -> bool {
    std::env::var("GPU_DEBUG").as_deref() == Ok("1")
}

#[cfg(feature = "gpu")]
fn gpu_min_n() -> usize {
    std::env::var("GPU_MIN_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096)
}

// ======== Main cohort runner (no normalization) ========
pub fn run_from_pairs(
    pairs: &[(PathBuf, PathBuf)],
    out_dir: &Path,
    cfg: CohortCfg,
) -> Result<(PathBuf, PathBuf), String> {
    use crate::profiler::Profiler;
    let mut prof = Profiler::new();

    std::fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    let plots_dir = out_dir.join("plots");
    if cfg.plot_first_snps > 0 {
        std::fs::create_dir_all(&plots_dir).map_err(|e| e.to_string())?;
    }

    // Load samples (parallel)
    let samples = {
        let _s = prof.scope("load_samples");
        load_samples(pairs)?
    };
    let s = samples.len();
    if s < 2 {
        return Err("need >=2 samples".into());
    }

    let mut ids = {
        let _s = prof.scope("common_ids");
        common_ids(&samples)
    };
    if ids.is_empty() {
        return Err("no common SNPs".into());
    }

    // Load learning state
    let mut fb = {
        let _s = prof.scope("load_feedback");
        if let Some(ref p) = cfg.feedback_in {
            adaptive::load_feedback(p)
        } else {
            HashMap::new()
        }
    };
    let mut bad = {
        let _s = prof.scope("load_bad_snps");
        if let Some(ref p) = cfg.bad_snps_in {
            adaptive::load_bad_snps(p)
        } else {
            HashSet::new()
        }
    };

    // Optionally drop known bad SNPs now
    {
        let _s = prof.scope("drop_bad_snps");
        if cfg.drop_bad_snps && !bad.is_empty() {
            ids.retain(|id| !bad.contains(id));
        }
    }

    // Apply SNP limiting (sampling takes precedence)
    {
        let _s = prof.scope("limit_snps");
        if let Some(n) = cfg.snp_sample_n {
            if ids.len() > n {
                let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);
                ids.shuffle(&mut rng);
                ids.truncate(n);
                ids.sort_unstable();
            }
        } else if let Some(n) = cfg.snp_first_n {
            if ids.len() > n {
                ids.truncate(n);
            }
        }
    }

    // Build compact per-sample positions aligned with *final* ids
    let pos_index: Vec<Vec<usize>> = {
        let _s = prof.scope("build_positions_after_limiting");
        build_positions_for_ids(&samples, &ids)
    };

    // r_min (0 => auto via per-sample ln(R+G+1) GMM and cohort median)
    let r_min = {
        let _s = prof.scope("compute_r_min");
        if cfg.r_min == 0 {
            let mut per_sample = Vec::with_capacity(s);
            for si in 0..s {
                let n = samples[si].x_red.len().min(samples[si].y_grn.len());
                let mut z: Vec<f64> = Vec::with_capacity(n);
                for k in 0..n {
                    let rr = samples[si].x_red[k] as u32;
                    let gg = samples[si].y_grn[k] as u32;
                    z.push(((rr + gg + 1) as f64).ln());
                }
                per_sample.push(rmin_from_lnxy_sum(&z));
            }
            per_sample.sort_unstable();
            per_sample[per_sample.len() / 2]
        } else {
            cfg.r_min
        }
    };

    // Prepare writers (calls & QC)
    let calls_path = out_dir.join("cohort_calls.csv");
    let qc_path = out_dir.join("cohort_qc_summary.csv");

    // Header for calls CSV (only if enabled)
    if WRITE_CALLS_CSV {
        let _s = prof.scope("write_calls_header");
        let mut w = Writer::from_path(&calls_path).map_err(|e| e.to_string())?;
        let mut header = vec!["SNP_ID".to_string()];
        header.extend(samples.iter().map(|s| s.name.clone()));
        w.write_record(&header).map_err(|e| e.to_string())?;
        w.flush().map_err(|e| e.to_string())?;
    }

    // stats per sample
    #[derive(Default)]
    struct PS {
        total: u64,
        pass_raw: u64,
        nocall_raw: u64,
        nocall_p: u64,
        nocall_maha: u64,
        calls_tau: u64,
        calls_p090: u64,
    }
    let mut stats: Vec<PS> = (0..s).map(|_| PS::default()).collect();

    // Decide tau (p_min)
    let mut tau_opt: Option<f64> = match cfg.p_mode {
        PMMode::Fixed(v) => Some(v),
        PMMode::Auto { .. } => None,
    };
    let alpha_default = match cfg.p_mode {
        PMMode::Auto { alpha } => alpha,
        _ => 0.005,
    };

    // Label for plots (ThetaR only)
    let feature_label =
        format!("θ·wθ={:.2} vs r·w_r={:.2}", cfg.feature_space.w_theta, cfg.feature_space.w_r);

    let m = ids.len();
    let block = 10_000.min(m).max(1);
    let mut plots_emitted: usize = 0;

    // Reopen calls writer in APPEND mode (or disable entirely)
    let mut w_opt: Option<csv::Writer<Box<dyn std::io::Write>>> = if WRITE_CALLS_CSV {
        let file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(&calls_path)
            .map_err(|e| e.to_string())?;
        Some(Writer::from_writer(Box::new(file)))
    } else {
        None
    };

    for chunk_start in (0..m).step_by(block) {
        let end = (chunk_start + block).min(m);
        let b = end - chunk_start; // SNPs in this chunk
        let s_samples = s;

        #[derive(Clone)]
        struct Row {
            idx: usize, // index into ids (so we can index pos_index later)
            id: u32,
            raw: Vec<u32>,
            labs: Vec<usize>,
            post: Vec<f64>,
            means: Vec<(f64, f64)>,
            vars: Vec<(f64, f64)>,
        }

        // ==================== GPU fast path (batched) ====================
        let try_gpu = {
            #[cfg(feature = "gpu")]
            {
                use_gpu_runtime() && cfg.k_max <= 3 && s_samples >= gpu_min_n()
            }
            #[cfg(not(feature = "gpu"))]
            {
                false
            }
        };

        let rows: Vec<Row> = if try_gpu {
            // --------------- Build SoA batch (xs,ys), raw sums, cache pts ---------------
            let _s = prof.scope("chunk_cluster_em_bic");
            let mut xs: Vec<f32> = Vec::with_capacity(b * s_samples);
            let mut ys: Vec<f32> = Vec::with_capacity(b * s_samples);
            let mut raw_sums: Vec<Vec<u32>> = vec![vec![0u32; s_samples]; b];
            let mut pts_batch: Vec<Vec<(f64, f64)>> = vec![Vec::with_capacity(s_samples); b];

            for (bj, j) in (chunk_start..end).enumerate() {
                for si in 0..s_samples {
                    let p = pos_index[si][j];
                    let red = samples[si].x_red[p] as f64;
                    let grn = samples[si].y_grn[p] as f64;
                    let lnr = (red + 1.0).ln();
                    let lng = (grn + 1.0).ln();
                    let (th, rr) = to_features_from_logs(&cfg.feature_space, lnr, lng);
                    xs.push(th as f32);
                    ys.push(rr as f32);
                    raw_sums[bj][si] =
                        (samples[si].x_red[p] as u32) + (samples[si].y_grn[p] as u32);
                    pts_batch[bj].push((th, rr));
                }
            }

            // --------------- Initial params per K (simple quantile seeding) ---------------
            #[inline]
            fn init_for_k(
                pts: &[(f64, f64)],
                k: usize,
            ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
                let n = pts.len().max(1);
                let mut mx = 0.0;
                let mut my = 0.0;
                for &(x, y) in pts {
                    mx += x;
                    my += y;
                }
                mx /= n as f64;
                my /= n as f64;
                let mut vx = 0.0;
                let mut vy = 0.0;
                for &(x, y) in pts {
                    vx += (x - mx) * (x - mx);
                    vy += (y - my) * (y - my);
                }
                vx = (vx / n as f64).max(1e-2);
                vy = (vy / n as f64).max(1e-2);

                let mut idx: Vec<usize> = (0..n).collect();
                idx.sort_by(|&a, &b| pts[a].0.partial_cmp(&pts[b].0).unwrap()); // sort by theta

                let mut pick = Vec::with_capacity(k);
                for t in 1..=k {
                    let pos =
                        ((t as f64) / ((k + 1) as f64) * (n as f64)).floor() as usize;
                    pick.push(idx[pos.min(n - 1)]);
                }

                let w = vec![1.0f32 / k as f32; k];
                let mut mxs = vec![0.0f32; k];
                let mut mys = vec![0.0f32; k];
                let vxs = vec![vx as f32; k];
                let vys = vec![vy as f32; k];
                for (j, &pi) in pick.iter().enumerate() {
                    mxs[j] = pts[pi].0 as f32;
                    mys[j] = pts[pi].1 as f32;
                }
                (w, mxs, mys, vxs, vys)
            }

            #[cfg(feature = "gpu")]
            use crate::model::{gpu_bic_batched, gpu_em_one_iter_batched_k, BatchEmOut};

            let kmax = cfg.k_max.min(3);
            struct KF {
                w: Vec<f32>,
                mx: Vec<f32>,
                my: Vec<f32>,
                vx: Vec<f32>,
                vy: Vec<f32>,
                ll: Vec<f32>,
                bic: Vec<f32>,
                k: usize,
            }
            let mut fits: Vec<KF> = Vec::new();

            for k in 1..=kmax {
                // concat initial params for the whole batch
                let mut w0: Vec<f32> = Vec::with_capacity(b * k);
                let mut mx0: Vec<f32> = Vec::with_capacity(b * k);
                let mut my0: Vec<f32> = Vec::with_capacity(b * k);
                let mut vx0: Vec<f32> = Vec::with_capacity(b * k);
                let mut vy0: Vec<f32> = Vec::with_capacity(b * k);

                for bj in 0..b {
                    let (w, mx, my, vx, vy) = init_for_k(&pts_batch[bj], k);
                    w0.extend_from_slice(&w);
                    mx0.extend_from_slice(&mx);
                    my0.extend_from_slice(&my);
                    vx0.extend_from_slice(&vx);
                    vy0.extend_from_slice(&vy);
                }

                // EM loop on GPU
                let mut cur_w = w0;
                let mut cur_mx = mx0;
                let mut cur_my = my0;
                let mut cur_vx = vx0;
                let mut cur_vy = vy0;
                let mut last_ll: Option<Vec<f32>> = None;

                for _it in 0..cfg.max_iters {
                    if let Some(BatchEmOut {
                        w,
                        mx,
                        my,
                        vx,
                        vy,
                        ll,
                    }) = gpu_em_one_iter_batched_k(
                        &xs,
                        &ys,
                        s_samples,
                        b,
                        k,
                        &cur_w,
                        &cur_mx,
                        &cur_my,
                        &cur_vx,
                        &cur_vy,
                    ) {
                        // check mean improvement over batch
                        if let Some(prev) = last_ll.as_ref() {
                            let mut delta = 0.0f64;
                            for i in 0..b {
                                delta += (ll[i] as f64 - prev[i] as f64).abs();
                            }
                            delta /= b as f64;
                            if delta <= cfg.tol {
                                cur_w = w;
                                cur_mx = mx;
                                cur_my = my;
                                cur_vx = vx;
                                cur_vy = vy;
                                last_ll = Some(ll);
                                break;
                            }
                        }
                        cur_w = w;
                        cur_mx = mx;
                        cur_my = my;
                        cur_vx = vx;
                        cur_vy = vy;
                        last_ll = Some(ll);
                    } else {
                        // GPU failed -> abort GPU path
                        last_ll = None;
                        break;
                    }
                }

                let Some(ll_vec) = last_ll else {
                    // abort GPU branch entirely -> fall back to CPU below
                    fits.clear();
                    break;
                };

                // BIC on GPU
                let bic_vec = gpu_bic_batched(s_samples, b, k, &ll_vec).unwrap_or_else(|| {
                    // tiny CPU fallback just in case
                    let p = ((k - 1) + 4 * k) as f32;
                    ll_vec
                        .iter()
                        .map(|&ll| -2.0 * ll + p * (s_samples as f32).ln())
                        .collect()
                });

                fits.push(KF {
                    w: cur_w,
                    mx: cur_mx,
                    my: cur_my,
                    vx: cur_vx,
                    vy: cur_vy,
                    ll: ll_vec,
                    bic: bic_vec,
                    k,
                });
            }

            // If GPU path failed mid-way, drop to CPU path
            let rows_gpu_or_cpu: Vec<Row> = if fits.is_empty() {
                if cfg!(feature = "gpu") {
                    #[cfg(feature = "gpu")]
                    if gpu_debug() {
                        eprintln!(
                            "[GPU] batched path unavailable -> CPU fallback"
                        );
                    }
                }
                // ==== CPU fallback, identical to the non-GPU branch ====
                (chunk_start..end)
                    .into_par_iter()
                    .map(|j| {
                        let id = ids[j];
                        let mut pts = Vec::with_capacity(s_samples);
                        let mut raw = Vec::with_capacity(s_samples);

                        for si in 0..s_samples {
                            let p = pos_index[si][j];
                            let red = samples[si].x_red[p] as f64;
                            let grn = samples[si].y_grn[p] as f64;
                            let lnr = (red + 1.0).ln();
                            let lng = (grn + 1.0).ln();
                            pts.push(to_features_from_logs(
                                &cfg.feature_space, lnr, lng,
                            ));
                            let rsum = (samples[si].x_red[p] as u32)
                                + (samples[si].y_grn[p] as u32);
                            raw.push(rsum);
                        }

                        let mut model = choose_k(
                            &pts,
                            cfg.k_max,
                            cfg.max_iters,
                            cfg.tol,
                            cfg.seed,
                            cfg.restarts,
                        );

                        // lightweight heuristics (unchanged)
                        const GAP3_TO_2: f64 = 0.035;
                        const GAP2_TO_1: f64 = 0.020;
                        const MIN_W: f64 = 0.03;
                        let sorted_thetas =
                            |means: &[(f64, f64)], w_theta: f64| -> Vec<f64> {
                                let wth = w_theta.max(1e-9);
                                let mut t: Vec<f64> =
                                    means.iter().map(|m| m.0 / wth).collect();
                                t.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                t
                            };
                        {
                            let wth = cfg.feature_space.w_theta;
                            if model.k > 1
                                && model.weights.iter().any(|&w| w < MIN_W)
                            {
                                model = if model.k == 3 {
                                    fit_diag_restarts(
                                        &pts,
                                        2,
                                        cfg.max_iters,
                                        cfg.tol,
                                        cfg.seed,
                                        cfg.restarts,
                                    )
                                } else {
                                    fit_k1(&pts)
                                };
                            }
                            let t = sorted_thetas(&model.means, wth);
                            if model.k == 3 {
                                let min_gap = (t[1] - t[0])
                                    .abs()
                                    .min((t[2] - t[1]).abs());
                                if min_gap < GAP3_TO_2 {
                                    let m2 = fit_diag_restarts(
                                        &pts,
                                        2,
                                        cfg.max_iters,
                                        cfg.tol,
                                        cfg.seed,
                                        cfg.restarts,
                                    );
                                    let t2 = sorted_thetas(&m2.means, wth);
                                    let gap = (t2[1] - t2[0]).abs();
                                    model =
                                        if gap < GAP2_TO_1 { fit_k1(&pts) } else { m2 };
                                }
                            } else if model.k == 2 {
                                let gap = (t[1] - t[0]).abs();
                                if gap < GAP2_TO_1 {
                                    model = fit_k1(&pts);
                                }
                            }
                        }

                        let (mut labs, mut post) = (Vec::new(), Vec::new());
                        predict(&model, &pts, &mut labs, &mut post);

                        Row {
                            idx: j,
                            id,
                            raw,
                            labs,
                            post,
                            means: model.means.clone(),
                            vars: model.vars.clone(),
                        }
                    })
                    .collect()
            } else {
                // ------- choose best K per SNP; compute posteriors/labels on CPU; build rows -------
                let mut rows_gpu: Vec<Row> = Vec::with_capacity(b);

                for bj in 0..b {
                    let id = ids[chunk_start + bj];

                    // pick K with min BIC
                    let mut best_k_idx = 0usize;
                    let mut best_bic = f32::INFINITY;
                    for (ki, f) in fits.iter().enumerate() {
                        let bval = f.bic[bj];
                        if bval < best_bic {
                            best_bic = bval;
                            best_k_idx = ki;
                        }
                    }
                    let fbest = &fits[best_k_idx];
                    let k = fbest.k;

                    // extract params for SNP bj
                    let off = bj * k;
                    let mut weights: Vec<f64> = Vec::with_capacity(k);
                    let mut means: Vec<(f64, f64)> = Vec::with_capacity(k);
                    let mut vars: Vec<(f64, f64)> = Vec::with_capacity(k);
                    for j in 0..k {
                        weights.push(fbest.w[off + j] as f64);
                        means.push((
                            fbest.mx[off + j] as f64,
                            fbest.my[off + j] as f64,
                        ));
                        vars.push((
                            fbest.vx[off + j] as f64,
                            fbest.vy[off + j] as f64,
                        ));
                    }

                    // compute posteriors/labels per-sample on CPU
                    let pts = &pts_batch[bj];
                    let mut labs: Vec<usize> = Vec::new();
                    let mut post: Vec<f64> = Vec::new();
                    let model = GmmDiag {
                        k,
                        weights: weights.clone(),
                        means: means.clone(),
                        vars: vars.clone(),
                        loglik: 0.0,
                    };
                    predict(&model, pts, &mut labs, &mut post);

                    rows_gpu.push(Row {
                        idx: chunk_start + bj,
                        id,
                        raw: raw_sums[bj].clone(),
                        labs,
                        post,
                        means,
                        vars,
                    });
                }

                rows_gpu
            };

            rows_gpu_or_cpu
        } else {
            // ==================== CPU fallback (original) ====================
            let _s = prof.scope("chunk_cluster_em_bic");
            (chunk_start..end)
                .into_par_iter()
                .map(|j| {
                    let id = ids[j];
                    let mut pts = Vec::with_capacity(s_samples);
                    let mut raw = Vec::with_capacity(s_samples);

                    for si in 0..s_samples {
                        let p = pos_index[si][j];
                        let red = samples[si].x_red[p] as f64;
                        let grn = samples[si].y_grn[p] as f64;
                        let lnr = (red + 1.0).ln();
                        let lng = (grn + 1.0).ln();
                        pts.push(to_features_from_logs(
                            &cfg.feature_space, lnr, lng,
                        ));
                        let rsum = (samples[si].x_red[p] as u32)
                            + (samples[si].y_grn[p] as u32);
                        raw.push(rsum);
                    }

                    let mut model = choose_k(
                        &pts,
                        cfg.k_max,
                        cfg.max_iters,
                        cfg.tol,
                        cfg.seed,
                        cfg.restarts,
                    );

                    // lightweight heuristics (unchanged)
                    const GAP3_TO_2: f64 = 0.035;
                    const GAP2_TO_1: f64 = 0.020;
                    const MIN_W: f64 = 0.03;
                    let sorted_thetas =
                        |means: &[(f64, f64)], w_theta: f64| -> Vec<f64> {
                            let wth = w_theta.max(1e-9);
                            let mut t: Vec<f64> =
                                means.iter().map(|m| m.0 / wth).collect();
                            t.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            t
                        };
                    {
                        let wth = cfg.feature_space.w_theta;
                        if model.k > 1 && model.weights.iter().any(|&w| w < MIN_W) {
                            model = if model.k == 3 {
                                fit_diag_restarts(
                                    &pts,
                                    2,
                                    cfg.max_iters,
                                    cfg.tol,
                                    cfg.seed,
                                    cfg.restarts,
                                )
                            } else {
                                fit_k1(&pts)
                            };
                        }
                        let t = sorted_thetas(&model.means, wth);
                        if model.k == 3 {
                            let min_gap = (t[1] - t[0])
                                .abs()
                                .min((t[2] - t[1]).abs());
                            if min_gap < GAP3_TO_2 {
                                let m2 = fit_diag_restarts(
                                    &pts,
                                    2,
                                    cfg.max_iters,
                                    cfg.tol,
                                    cfg.seed,
                                    cfg.restarts,
                                );
                                let t2 = sorted_thetas(&m2.means, wth);
                                let gap = (t2[1] - t2[0]).abs();
                                model = if gap < GAP2_TO_1 {
                                    fit_k1(&pts)
                                } else {
                                    m2
                                };
                            }
                        } else if model.k == 2 {
                            let gap = (t[1] - t[0]).abs();
                            if gap < GAP2_TO_1 {
                                model = fit_k1(&pts);
                            }
                        }
                    }

                    let (mut labs, mut post) = (Vec::new(), Vec::new());
                    predict(&model, &pts, &mut labs, &mut post);

                    Row {
                        idx: j,
                        id,
                        raw,
                        labs,
                        post,
                        means: model.means.clone(),
                        vars: model.vars.clone(),
                    }
                })
                .collect()
        };

        // ================= τ (auto once) =================
        if tau_opt.is_none() {
            let _s = prof.scope("auto_tau");
            let mut ps = Vec::new();
            for r in &rows {
                for (si, &p) in r.post.iter().enumerate() {
                    if r.raw[si] >= r_min {
                        ps.push(p);
                    }
                }
            }
            let tau = auto_tau_from_posteriors(ps, alpha_default);
            tau_opt = Some(tau);
            eprintln!(
                "AUTO τ via FDR α={:.4} -> τ={:.4}",
                alpha_default, tau
            );
        }
        let tau = tau_opt.unwrap();

       // per-SNP nocall counter
        let mut nocall_per_snp: HashMap<u32, u64> = HashMap::new();

        // ---- Finalize calls + write rows ----
        {
            let idx_finalize = prof.start("chunk_finalize_and_write");
            for r in rows {
                let mut out = Vec::with_capacity(1 + s_samples);
                out.push(r.id.to_string());

                let names: Vec<&'static str> =
                    map_labels_for_snp_strict(&r.means, cfg.feature_space.w_theta);

                let mut this_snp_nocall_all: u64 = 0;

                // Precompute features for geometry/maha
                let mut pts_cache: Vec<(f64, f64)> = Vec::with_capacity(s_samples);
                for si in 0..s_samples {
                    let p = pos_index[si][r.idx];
                    let red = samples[si].x_red[p] as f64;
                    let grn = samples[si].y_grn[p] as f64;
                    let lnr = (red + 1.0).ln();
                    let lng = (grn + 1.0).ln();
                    pts_cache.push(to_features_from_logs(
                        &cfg.feature_space, lnr, lng,
                    ));
                }

                // Refined centres (same logic as before)
                let mut refined = r.means.clone();
                if !r.means.is_empty() {
                    let k = r.means.len();
                    let mut acc = vec![(0.0f64, 0.0f64, 0.0f64); k]; // sumx,sumy,sumw
                    for si in 0..s_samples {
                        if r.post[si] >= tau {
                            let j = safe_lab_index(r.id, r.labs[si], k);
                            let (x, y) = pts_cache[si];
                            let w = r.post[si];
                            acc[j].0 += x * w;
                            acc[j].1 += y * w;
                            acc[j].2 += w;
                        }
                    }
                    for j in 0..k {
                        if acc[j].2 > 1e-6 {
                            refined[j].0 = acc[j].0 / acc[j].2;
                            refined[j].1 = acc[j].1 / acc[j].2;
                        }
                    }
                }

                // Mahalanobis gate vs refined centres
                let mut m_gate_refined = vec![false; s_samples];
                for si in 0..s_samples {
                    if r.post[si] >= tau {
                        let j = safe_lab_index(r.id, r.labs[si], refined.len());
                        let m2r = maha2_diag(pts_cache[si], refined[j], r.vars[j]);
                        if m2r.sqrt() > cfg.max_maha {
                            m_gate_refined[si] = true;
                        }
                    }
                }

                // Write calls & update stats/feedback
                for si in 0..s_samples {
                    let st = &mut stats[si];
                    st.total += 1;

                    if r.raw[si] < r_min {
                        st.nocall_raw += 1;
                        out.push("NoCall".into());
                        this_snp_nocall_all += 1;
                        let e = fb.entry(r.id).or_default();
                        e.seen += 1;
                        e.nocall_raw += 1;
                        continue;
                    }

                    st.pass_raw += 1;
                    if r.post[si] >= 0.90 {
                        st.calls_p090 += 1;
                    }

                    if m_gate_refined[si] {
                        st.nocall_p += 1;
                        st.nocall_maha += 1;
                        out.push("NoCall".into());
                        this_snp_nocall_all += 1;
                        let e = fb.entry(r.id).or_default();
                        e.seen += 1;
                        e.nocall_p += 1;
                        continue;
                    }

                    if r.post[si] < tau {
                        st.nocall_p += 1;
                        out.push("NoCall".into());
                        this_snp_nocall_all += 1;
                        let e = fb.entry(r.id).or_default();
                        e.seen += 1;
                        e.nocall_p += 1;
                    } else {
                        let jlab0 =
                            safe_lab_index(r.id, r.labs[si], r.means.len());
                        let call = names[jlab0];
                        let strong = 0.97;
                        let band = 0.04;

                        let theta_sample = pts_cache[si].0
                            / cfg.feature_space.w_theta.max(1e-9);
                        let ok = r.post[si] >= strong
                            || theta_guard_ok(
                                &refined,
                                cfg.feature_space.w_theta,
                                call,
                                theta_sample,
                                band,
                            );

                        if ok {
                            st.calls_tau += 1;
                            out.push(call.into());
                            let e = fb.entry(r.id).or_default();
                            e.seen += 1;
                        } else {
                            st.nocall_p += 1;
                            out.push("NoCall".into());
                            this_snp_nocall_all += 1;
                            let e = fb.entry(r.id).or_default();
                            e.seen += 1;
                            e.nocall_p += 1;
                        }
                    }
                }

                // learn bad SNPs
                nocall_per_snp.insert(r.id, this_snp_nocall_all);

                // append row (only if enabled)
                if let Some(w) = w_opt.as_mut() {
                    w.write_record(&out).map_err(|e| e.to_string())?;
                }

                // ----- Plots + Debug -----
                let is_debug_this =
                    cfg.debug_snp.map(|ds| ds == r.id).unwrap_or(false);
                let do_plot = (cfg.plot_first_snps > 0
                    && plots_emitted < cfg.plot_first_snps)
                    || is_debug_this;

                // recompute feature points for this SNP (same as clustering pass)
                let mut pts = Vec::with_capacity(s_samples);
                for si in 0..s_samples {
                    let p = pos_index[si][r.idx];
                    let red = samples[si].x_red[p] as f64;
                    let grn = samples[si].y_grn[p] as f64;
                    let lnr = (red + 1.0).ln();
                    let lng = (grn + 1.0).ln();
                    pts.push(to_features_from_logs(
                        &cfg.feature_space, lnr, lng,
                    ));
                }

                // Build final per-sample calls exactly like CSV logic (incl refined maha gate)
                let mut final_calls: Vec<String> = Vec::with_capacity(s_samples);
                let mut maha_flags: Vec<bool> = Vec::with_capacity(s_samples);
                for si in 0..s_samples {
                    let c = if r.raw[si] < r_min {
                        "NoCall".to_string()
                    } else if r.post[si] < tau {
                        "NoCall".to_string()
                    } else if m_gate_refined[si] {
                        "NoCall".to_string()
                    } else {
                        let jlab0 =
                            safe_lab_index(r.id, r.labs[si], r.means.len());
                        let call = names[jlab0];
                        let strong = 0.97;
                        let band = 0.04;
                        let theta_sample = pts[si].0
                            / cfg.feature_space.w_theta.max(1e-9);
                        if r.post[si] >= strong
                            || theta_guard_ok(
                                &refined,
                                cfg.feature_space.w_theta,
                                call,
                                theta_sample,
                                band,
                            ) {
                            call.to_string()
                        } else {
                            "NoCall".to_string()
                        }
                    };
                    final_calls.push(c);
                    maha_flags.push(m_gate_refined[si]); // use refined gate flags in plot
                }

                // 1) Plot
                if do_plot {
                    {
                        let _s_plot = prof.scope("plot_one_snp");
                        let plot_path_base =
                            plots_dir.join(format!("snp_{:08}", r.id));
                        let _ = write_cluster_scatter_png_calls(
                            &plot_path_base,
                            r.id,
                            &feature_label,
                            &pts,
                            &final_calls,
                            &maha_flags,
                            &r.means,
                            Some(&refined),
                        );
                    }
                    if cfg.plot_first_snps > 0 && plots_emitted < cfg.plot_first_snps
                    {
                        plots_emitted += 1;
                    }
                }

                // 2) Debug CSV
                if is_debug_this {
                    {
                        let _s_dbg = prof.scope("debug_csv_one_snp");
                        let _ = write_debug_csv_for_snp(
                            out_dir,
                            r.id,
                            &pts,
                            &r.labs,
                            &r.post,
                            &m_gate_refined, // refined gate flags
                            &r.raw,
                            &final_calls,
                            &r.means,
                            Some(&refined),
                            tau,
                            r_min,
                        );
                    }
                }
            }
            if let Some(w) = w_opt.as_mut() {
                w.flush().map_err(|e| e.to_string())?;
            }
            prof.end(idx_finalize);
        }

        // After chunk: update bad SNPs set
        if cfg.nocall_bad_thresh > 0.0 {
            let _s = prof.scope("update_bad_snps_after_chunk");
            for (id, n_no) in nocall_per_snp {
                let rate = (n_no as f64) / (s_samples as f64);
                if rate >= cfg.nocall_bad_thresh {
                    bad.insert(id);
                    let e = fb.entry(id).or_default();
                    e.bad = true;
                }
            }
        }
    }

    // QC summary
    {
        let _s = prof.scope("write_qc_summary");
        let mut qcw = Writer::from_path(&qc_path).map_err(|e| e.to_string())?;
        qcw.write_record(&[
            "Sample",
            "total_snps",
            "pass_raw",
            "nocall_raw",
            "nocall_p_at_tau",
            "nocall_maha_at_tau",
            "calls_at_tau",
            "calls_p090",
            "call_rate_tau",
            "call_rate_p090",
            "r_min_used",
            "tau_used",
        ])
        .map_err(|e| e.to_string())?;
        let tau_report = tau_opt.unwrap_or(0.90);
        for (i, st) in stats.iter().enumerate() {
            let name = &samples[i].name;
            let rate_tau = (st.calls_tau as f64) / (st.total as f64);
            let rate_p = (st.calls_p090 as f64) / (st.total as f64);
            qcw.write_record(&[
                name.to_string(),
                st.total.to_string(),
                st.pass_raw.to_string(),
                st.nocall_raw.to_string(),
                st.nocall_p.to_string(),
                st.nocall_maha.to_string(),
                st.calls_tau.to_string(),
                st.calls_p090.to_string(),
                format!("{:.6}", rate_tau),
                format!("{:.6}", rate_p),
                r_min.to_string(),
                format!("{:.4}", tau_report),
            ])
            .map_err(|e| e.to_string())?;
        }
        qcw.flush().map_err(|e| e.to_string())?;
    }

    // Save learning state
    {
        let _s = prof.scope("save_learning_state");
        if let Some(ref p) = cfg.feedback_out {
            let _ = adaptive::save_feedback(p, &fb);
        }
        if let Some(ref p) = cfg.bad_snps_out {
            let _ = adaptive::save_bad_snps(p, &bad);
        }
    }

    // Persist timings
    {
        let idx = prof.start("write_timings_csv");
        let _ = prof.write_csv(out_dir);
        prof.end(idx);
    }

    Ok((calls_path, qc_path))
}

// ---------- Plotting by FINAL CALLS (AA/AB/BB; NoCall hidden) ----------
fn write_cluster_scatter_png_calls(
    out_path_base: &Path,
    snp_id: u32,
    feature_label: &str,
    pts: &[(f64, f64)],
    final_calls: &[String], // "AA"/"AB"/"BB"/"NoCall"
    maha_flags: &[bool],    // refined gate flags
    means: &[(f64, f64)],   // EM-fitted centres (fallback)
    refined_means: Option<&[(f64, f64)]>, // preferred centres
) -> Result<PathBuf, String> {
    use plotters::prelude::*;

    assert_eq!(pts.len(), final_calls.len());
    assert_eq!(pts.len(), maha_flags.len());

    let file = out_path_base.with_extension("png");
    let file_to_return = file.clone();
    let outfile_str: String = file.to_string_lossy().into_owned();

    // Count only called points (ignore NoCall entirely)
    let mut n_aa = 0usize;
    let mut n_ab = 0usize;
    let mut n_bb = 0usize;

    let kept: Vec<usize> = final_calls
        .iter()
        .enumerate()
        .filter(|(_, c)| matches!(c.as_str(), "AA" | "AB" | "BB"))
        .map(|(i, _)| i)
        .collect();

    for &i in &kept {
        match final_calls[i].as_str() {
            "AA" => n_aa += 1,
            "AB" => n_ab += 1,
            "BB" => n_bb += 1,
            _ => {}
        }
    }

    // Prefer refined centres if provided; otherwise use EM means
    let centres: &[(f64, f64)] = if let Some(rm) = refined_means { rm } else { means };

    // Bounds: include centres + kept points (fallback to all pts if none kept)
    let pts_for_bounds: Vec<(f64, f64)> = if kept.is_empty() {
        pts.to_vec()
    } else {
        kept.iter().map(|&i| pts[i]).collect()
    };

    let mut xmin = f64::INFINITY;
    let mut xmax = -f64::INFINITY;
    let mut ymin = f64::INFINITY;
    let mut ymax = -f64::INFINITY;
    for &(x, y) in pts_for_bounds.iter().chain(centres.iter()) {
        xmin = xmin.min(x);
        xmax = xmax.max(x);
        ymin = ymin.min(y);
        ymax = ymax.max(y);
    }

    let padx = (xmax - xmin).abs().max(1.0) * 0.06;
    let pady = (ymax - ymin).abs().max(1.0) * 0.06;

    let mut parts = feature_label.split(" vs ");
    let xlab = parts.next().unwrap_or("x");
    let ylab = parts.next().unwrap_or("y");

    // Larger canvas for higher-resolution text
    let root = BitMapBackend::new(&outfile_str, (1400, 1000)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| e.to_string())?;

    // Bigger margins and label areas to fit larger fonts
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("SNP {} — {}", snp_id, feature_label),
            ("sans-serif", 32).into_font(), 
        )
        .margin(25)
        .x_label_area_size(70)           
        .y_label_area_size(80)           
        .build_cartesian_2d((xmin - padx)..(xmax + padx), (ymin - pady)..(ymax + pady))
        .map_err(|e| e.to_string())?;
    chart
        .configure_mesh()
        .x_desc(xlab)
        .y_desc(ylab)
        .axis_desc_style(("sans-serif", 35)) // larger axis titles
        .label_style(("sans-serif", 30))     // larger tick labels
        .draw()
        .map_err(|e| e.to_string())?;

    // Group indices by final call (NoCall excluded from drawing)
    let mut idx_aa = Vec::new();
    let mut idx_ab = Vec::new();
    let mut idx_bb = Vec::new();
    for &i in &kept {
        match final_calls[i].as_str() {
            "AA" => idx_aa.push(i),
            "AB" => idx_ab.push(i),
            "BB" => idx_bb.push(i),
            _ => {}
        }
    }

    // Legend labels with counts
    let lbl_aa = format!("AA (n={})", n_aa);
    let lbl_ab = format!("AB (n={})", n_ab);
    let lbl_bb = format!("BB (n={})", n_bb);

    // Draw helper: solid = pass; hollow = Mahalanobis-gated but still passed τ
    let mut draw_group =
        |label: &str, idxs: &[usize], color: RGBColor| -> Result<(), String> {
            chart
                .draw_series(
                    idxs.iter()
                        .filter(|&&i| !maha_flags[i])
                        .map(|&i| {
                            let (x, y) = pts[i];
                            Circle::new((x, y), 3, color.filled())
                        }),
                )
                .map_err(|e| e.to_string())?
                .label(label.to_string())
                .legend(move |(x, y)| Circle::new((x, y), 5, color.filled()));

            chart
                .draw_series(
                    idxs.iter()
                        .filter(|&&i| maha_flags[i])
                        .map(|&i| {
                            let (x, y) = pts[i];
                            Circle::new((x, y), 4, color.stroke_width(1))
                        }),
                )
                .map_err(|e| e.to_string())?;

            Ok(())
        };

    if !idx_aa.is_empty() {
        draw_group(&lbl_aa, &idx_aa, RED)?;
    }
    if !idx_ab.is_empty() {
        draw_group(&lbl_ab, &idx_ab, BLUE)?;
    }
    if !idx_bb.is_empty() {
        draw_group(&lbl_bb, &idx_bb, GREEN)?;
    }

    // Draw chosen centres as black crosses
    chart
        .draw_series(centres.iter().map(|&(x, y)| Cross::new((x, y), 7, &BLACK)))
        .map_err(|e| e.to_string())?
        .label("Centre")
        .legend(|(x, y)| Cross::new((x, y), 7, &BLACK));

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .border_style(&BLACK)
        .background_style(WHITE.mix(0.85))
        .label_font(("sans-serif", 30))
        .draw()
        .map_err(|e| e.to_string())?;
    
    root.present().map_err(|e| e.to_string())?;
    Ok(file_to_return)
}

// =============== Debug CSV writer ===============
fn write_debug_csv_for_snp(
    out_dir: &Path,
    snp_id: u32,
    pts: &[(f64, f64)],
    labs: &[usize],
    post: &[f64],
    maha_flags: &[bool], // refined gate flags
    raw_sums: &[u32],
    final_calls: &[String],
    means: &[(f64, f64)],
    refined_means: Option<&[(f64, f64)]>,
    tau: f64,
    r_min: u32,
) -> Result<(), String> {
    let path = out_dir.join(format!("debug_snp_{:08}.csv", snp_id));
    let mut w = csv::Writer::from_path(&path).map_err(|e| e.to_string())?;
    w.write_record(&[
        "snp_id",
        "sample_idx",
        "theta",
        "r",
        "lab",
        "posterior",
        "maha_gated",
        "raw_sum",
        "final_call",
        "tau",
        "r_min",
    ])
    .map_err(|e| e.to_string())?;
    for i in 0..pts.len() {
        w.write_record(&[
            snp_id.to_string(),
            i.to_string(),
            format!("{:.6}", pts[i].0),
            format!("{:.6}", pts[i].1),
            labs[i].to_string(),
            format!("{:.6}", post[i]),
            maha_flags[i].to_string(),
            raw_sums[i].to_string(),
            final_calls[i].clone(),
            format!("{:.4}", tau),
            r_min.to_string(),
        ])
        .map_err(|e| e.to_string())?;
    }
    // EM means
    w.write_record(&["MEAN_IDX", "theta", "r"])
        .map_err(|e| e.to_string())?;
    for (j, m) in means.iter().enumerate() {
        w.write_record(&[j.to_string(), format!("{:.6}", m.0), format!("{:.6}", m.1)])
            .map_err(|e| e.to_string())?;
    }
    // refined means
    if let Some(rm) = refined_means {
        w.write_record(&["REFINED_MEAN_IDX", "theta", "r"])
            .map_err(|e| e.to_string())?;
        for (j, m) in rm.iter().enumerate() {
            w.write_record(&[j.to_string(), format!("{:.6}", m.0), format!("{:.6}", m.1)])
                .map_err(|e| e.to_string())?;
        }
    }
    w.flush().map_err(|e| e.to_string())?;
    Ok(())
}
