use crate::idat::read_idat_any;
use crate::model::{
    auto_tau_from_posteriors, bic, fit_diag_restarts, maha2_diag, predict, rmin_from_lnxy_sum,
    to_features_from_logs, FeatureSpace, GmmDiag,
};
use csv::Writer;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use plotters::prelude::*;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use crate::model::ln1p_u16;


// =============== PMMode / Config ===============
pub enum PMMode {
    Fixed(f64),
    Auto { alpha: f64 },
}

pub struct CohortCfg {
    pub feature_space: FeatureSpace,
    pub r_min: u32, // 0 => auto median of per-sample 1D-GMM thresholds on ln(R+G+1)
    pub p_mode: PMMode,
    pub max_maha: f64,
    pub k_max: usize,
    pub max_iters: usize,
    pub tol: f64,
    pub seed: u64,
    pub restarts: usize,
    pub plot_first_snps: usize,

    // Normalization toggles + bead pools
    pub norm_affine: bool,              // whitening path
    pub norm_illumina: bool,            // Illumina-style per-pool transform
    pub bead_pools_csv: Option<PathBuf>, // CSV: IlluminaID,pool_id (u32,u16)

    // SNP limiting
    pub snp_first_n: Option<usize>,   // take first N SNPs after intersect+sort
    pub snp_sample_n: Option<usize>,  // OR: take a random sample of N SNPs (seeded)

}

// =============== Sample cache ===============
#[derive(Clone)]
struct SampleData {
    name: String,
    ids: Vec<u32>,
    x_red: Vec<u16>,
    y_grn: Vec<u16>,
    pos: HashMap<u32, usize>,
}

fn load_samples(pairs: &[(PathBuf, PathBuf)]) -> Result<Vec<SampleData>, String> {
    let mut out = Vec::with_capacity(pairs.len());
    for (gpath, rpath) in pairs {
        let g = read_idat_any(gpath).map_err(|e| format!("{}: {}", gpath.display(), e))?;
        let r = read_idat_any(rpath).map_err(|e| format!("{}: {}", rpath.display(), e))?;
        let n = g.means.len().min(r.means.len());
        if n == 0 {
            return Err("empty pair".into());
        }
        let mut pos = HashMap::with_capacity(n);
        for (i, &id) in g.illumina_ids.iter().take(n).enumerate() {
            pos.insert(id, i);
        }
        let stem = gpath.file_stem().and_then(|s| s.to_str()).unwrap_or("sample");
        let name = stem.strip_suffix("_Grn").unwrap_or(stem).to_string();
        out.push(SampleData {
            name,
            ids: g.illumina_ids[..n].to_vec(),
            x_red: r.means[..n].to_vec(),
            y_grn: g.means[..n].to_vec(),
            pos,
        });
    }
    Ok(out)
}

fn common_ids(samples: &[SampleData]) -> Vec<u32> {
    if samples.is_empty() { return vec![]; }
    let mut set: HashSet<u32> = samples[0].ids.iter().copied().collect();
    for s in &samples[1..] {
        let t: HashSet<u32> = s.ids.iter().copied().collect();
        set = set.intersection(&t).copied().collect();
    }
    let mut v: Vec<u32> = set.into_iter().collect();
    v.sort_unstable();
    v
}

// =============== Normalization support ===============

// ---- bead pools CSV ----
fn load_bead_pools_csv(path: &Path) -> Result<HashMap<u32, u16>, String> {
    let mut out = HashMap::new();
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(path)
        .map_err(|e| e.to_string())?;
    for rec in rdr.records().flatten() {
        let id: u32 = rec.get(0).unwrap_or("").trim().parse().map_err(|_| "bad illumina_id")?;
        let pool: u16 = rec.get(1).unwrap_or("0").trim().parse().unwrap_or(0);
        out.insert(id, pool);
    }
    Ok(out)
}

// ---- existing whitening (kept) ----
#[derive(Clone, Copy, Debug)]
struct PoolAffine { mx:f64, my:f64, w11:f64, w12:f64, w21:f64, w22:f64 }

fn apply_affine(pa: Option<&PoolAffine>, x: f64, y: f64) -> (f64, f64) {
    if let Some(p) = pa {
        let cx = x - p.mx;
        let cy = y - p.my;
        (p.w11 * cx + p.w12 * cy, p.w21 * cx + p.w22 * cy)
    } else { (x, y) }
}

// helper for 2x2 symmetric eigendecomp
fn eig2_sym(a: f64, b: f64, c: f64) -> ((f64,f64,f64),(f64,f64,f64)) {
    let tr = a + c;
    let det = a*c - b*b;
    let disc = (tr*tr - 4.0*det).max(0.0).sqrt();
    let l1 = 0.5*(tr + disc);
    let l2 = 0.5*(tr - disc);
    let (v1x, v1y) = if b.abs() > 1e-12 { (l1 - c, b) } else { (1.0, 0.0) };
    let n1 = (v1x*v1x + v1y*v1y).sqrt().max(1e-18);
    let v1x = v1x / n1; let v1y = v1y / n1;
    let v2x = -v1y; let v2y = v1x;
    ((l1, v1x, v1y), (l2, v2x, v2y))
}

fn compute_pool_affines(
    samples: &[SampleData],
    ids: &[u32],
    bead_pools: &HashMap<u32, u16>,
) -> HashMap<u16, PoolAffine> {
    #[derive(Default)]
    struct Acc { n:f64, sx:f64, sy:f64, sxx:f64, syy:f64, sxy:f64 }
    let mut acc: HashMap<u16, Acc> = HashMap::new();

    for &id in ids {
        let pool = bead_pools.get(&id).copied().unwrap_or(0u16);
        let a = acc.entry(pool).or_default();
        for sd in samples {
            if let Some(&p) = sd.pos.get(&id) {
                let x = sd.x_red[p] as f64;
                let y = sd.y_grn[p] as f64;
                a.n += 1.0;
                a.sx += x; a.sy += y;
                a.sxx += x*x; a.syy += y*y; a.sxy += x*y;
            }
        }
    }

    let mut out = HashMap::new();
    for (pool, a) in acc {
        if a.n < 10.0 {
            out.insert(pool, PoolAffine { mx:0.0, my:0.0, w11:1.0, w12:0.0, w21:0.0, w22:1.0 });
            continue;
        }
        let mx = a.sx / a.n;
        let my = a.sy / a.n;
        let cxx = (a.sxx/a.n) - mx*mx;
        let cyy = (a.syy/a.n) - my*my;
        let cxy = (a.sxy/a.n) - mx*my;

        let ((l1, v1x, v1y), (l2, v2x, v2y)) = eig2_sym(cxx, cxy, cyy);
        let eps = 1e-9;
        let s1 = 1.0 / (l1.max(eps)).sqrt();
        let s2 = 1.0 / (l2.max(eps)).sqrt();

        let w11 = s1*v1x*v1x + s2*v2x*v2x;
        let w12 = s1*v1x*v1y + s2*v2x*v2y;
        let w21 = w12;
        let w22 = s1*v1y*v1y + s2*v2y*v2y;

        out.insert(pool, PoolAffine { mx, my, w11, w12, w21, w22 });
    }
    out
}

// ---- Illumina-style translate→rotate→shear→scale ----
#[derive(Clone, Copy, Debug)]
struct IlluminaAffine {
    ox: f64, oy: f64,
    cos_t: f64, sin_t: f64,
    shx: f64,
    sx: f64, sy: f64,
}

#[inline]
fn apply_illumina_affine(a: Option<&IlluminaAffine>, x: f64, y: f64) -> (f64, f64) {
    if let Some(t) = a {
        // translate
        let cx = x - t.ox;
        let cy = y - t.oy;
        // rotate
        let rx =  t.cos_t * cx + t.sin_t * cy;
        let ry = -t.sin_t * cx + t.cos_t * cy;
        // shear X|Y
        let sx = rx - t.shx * ry;
        let sy = ry;
        // scale
        (sx * t.sx, sy * t.sy)
    } else { (x, y) }
}

fn percentile(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() { return 0.0; }
    let p = p.clamp(0.0, 1.0);
    let k = ((v.len() as f64 - 1.0) * p).round() as usize;
    v.sort_by(|a,b| a.partial_cmp(b).unwrap());
    v[k]
}

fn trim_bounds(xs: &[f64], lo: f64, hi: f64) -> (f64, f64) {
    let mut v = xs.to_vec();
    let a = percentile(&mut v, lo);
    let b = percentile(&mut v, hi);
    (a, b)
}

fn robust_center_cov(xs: &[f64], ys: &[f64]) -> (f64,f64, f64,f64,f64) {
    assert_eq!(xs.len(), ys.len());
    let n = xs.len();
    if n == 0 { return (0.0,0.0, 1.0,1.0, 0.0); }

    // 2%–98% trimming on each axis
    let (xl, xh) = trim_bounds(xs, 0.02, 0.98);
    let (yl, yh) = trim_bounds(ys, 0.02, 0.98);

    let mut cx = 0.0; let mut cy = 0.0; let mut cnt = 0.0;
    for i in 0..n {
        let (x, y) = (xs[i], ys[i]);
        if x < xl || x > xh || y < yl || y > yh { continue; }
        cx += x; cy += y; cnt += 1.0;
    }
    if cnt < 5.0 { return (0.0,0.0, 1.0, 1.0, 0.0); }
    cx /= cnt; cy /= cnt;

    let mut sxx=0.0; let mut syy=0.0; let mut sxy=0.0; let mut m=0.0;
    for i in 0..n {
        let (x, y) = (xs[i], ys[i]);
        if x < xl || x > xh || y < yl || y > yh { continue; }
        let dx = x - cx; let dy = y - cy;
        sxx += dx*dx; syy += dy*dy; sxy += dx*dy; m += 1.0;
    }
    if m < 5.0 { return (cx, cy, 1.0, 1.0, 0.0); }
    (cx, cy, sxx.max(1e-9)/m, syy.max(1e-9)/m, sxy/m)
}

fn compute_pool_affines_illumina(
    samples: &[SampleData],
    ids: &[u32],
    bead_pools: &HashMap<u32, u16>,
) -> HashMap<u16, IlluminaAffine> {
    let mut xs_by_pool: HashMap<u16, Vec<f64>> = HashMap::new();
    let mut ys_by_pool: HashMap<u16, Vec<f64>> = HashMap::new();

    for &id in ids {
        let pool = bead_pools.get(&id).copied().unwrap_or(0u16);
        let xs = xs_by_pool.entry(pool).or_default();
        let ys = ys_by_pool.entry(pool).or_default();
        for s in samples {
            if let Some(&p) = s.pos.get(&id) {
                xs.push(s.x_red[p] as f64);
                ys.push(s.y_grn[p] as f64);
            }
        }
    }

    let mut out: HashMap<u16, IlluminaAffine> = HashMap::new();

    for (pool, xs) in xs_by_pool.into_iter() {
        let ys = ys_by_pool.get(&pool).cloned().unwrap_or_default();
        if xs.len() < 50 || ys.len() < 50 {
            out.insert(pool, IlluminaAffine { ox:0.0, oy:0.0, cos_t:1.0, sin_t:0.0, shx:0.0, sx:1.0, sy:1.0 });
            continue;
        }

        let (cx, cy, vxx, vyy, vxy) = robust_center_cov(&xs, &ys);
        // principal axis rotation
        let theta = 0.5 * (2.0 * vxy).atan2(vxx - vyy);
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // rotate points around robust center to estimate shear/scale
        let mut rx: Vec<f64> = Vec::with_capacity(xs.len());
        let mut ry: Vec<f64> = Vec::with_capacity(xs.len());
        for i in 0..xs.len() {
            let dx = xs[i] - cx; let dy = ys[i] - cy;
            rx.push( cos_t*dx + sin_t*dy );
            ry.push(-sin_t*dx + cos_t*dy );
        }

        // shear to remove residual covariance: shx ≈ cov(rx,ry)/var(ry)
        let (_mx, my, _vx2, vy2, vxy2) = robust_center_cov(&rx, &ry);
        let shx = if vy2 > 1e-12 { vxy2 / vy2 } else { 0.0 };

        // estimate scales on sheared coords
        let mut sxv=0.0; let mut syv=0.0; let mut cnt=0.0;
        for i in 0..rx.len() {
            let sx_ = rx[i] - shx * ry[i];
            let sy_ = ry[i] - my; // zero-mean on Y after rotation
            sxv += sx_*sx_; syv += sy_*sy_; cnt += 1.0;
        }
        let (vx_, vy_) = if cnt > 1.0 { (sxv/cnt, syv/cnt) } else { (1.0, 1.0) };
        let sx = 1.0 / vx_.max(1e-9).sqrt();
        let sy = 1.0 / vy_.max(1e-9).sqrt();

        out.insert(pool, IlluminaAffine { ox:cx, oy:cy, cos_t, sin_t, shx, sx, sy });
    }

    out
}

#[derive(Clone, Copy)]
enum NormMode {
    None,
    AffinePool,
    IlluminaPool,
}

// =============== K=1 baseline & choose_k ===============
fn fit_k1(points: &[(f64, f64)]) -> GmmDiag {
    let n = points.len().max(1);
    let (mut mx, mut my) = (0.0, 0.0);
    for &(x, y) in points {
        mx += x; my += y;
    }
    mx /= n as f64; my /= n as f64;

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

    GmmDiag { k: 1, weights: vec![1.0], means: vec![(mx, my)], vars: vec![(vx, vy)], loglik: ll }
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
    if kmax <= 1 { return g1; }
    let g2 = fit_diag_restarts(points, 2, max_it, tol, seed, restarts);
    if kmax == 2 {
        let (b1, b2) = (bic(&g1, points.len()), bic(&g2, points.len()));
        return if b1 <= b2 { g1 } else { g2 };
    }
    let g3 = fit_diag_restarts(points, 3, max_it, tol, seed, restarts);
    let (b1, b2, b3) = (bic(&g1, points.len()), bic(&g2, points.len()), bic(&g3, points.len()));
    if b1 <= b2 && b1 <= b3 { g1 } else if b2 <= b3 { g2 } else { g3 }
}

// =============== Single-cloud call helper ===============
fn single_cloud_call(feature_space: &FeatureSpace, mu: (f64, f64)) -> &'static str {
    let theta = match *feature_space {
        FeatureSpace::ThetaR { w_theta, .. } => {
            let w = w_theta.max(1e-6);
            (mu.0 / w).clamp(0.0, 1.0)
        }
        FeatureSpace::XYLog1p => {
            let r = mu.0.exp() - 1.0;
            let g = mu.1.exp() - 1.0;
            let t = g.atan2(r) / std::f64::consts::PI;
            if t.is_finite() { t.clamp(0.0, 1.0) } else { 0.5 }
        }
    };
    if theta <= 0.33 { "BB" } else if theta >= 0.67 { "AA" } else { "AB" }
}

// =============== Main cohort runner (with normalization) ===============
pub fn run_from_pairs(
    pairs: &[(PathBuf, PathBuf)],
    out_dir: &Path,
    cfg: CohortCfg,
) -> Result<(PathBuf, PathBuf), String> {
    std::fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    let plots_dir = out_dir.join("plots");
    if cfg.plot_first_snps > 0 {
        std::fs::create_dir_all(&plots_dir).map_err(|e| e.to_string())?;
    }

    let samples = load_samples(pairs)?; let s = samples.len();
    if s < 2 { return Err("need >=2 samples".into()); }
    let mut ids = common_ids(&samples);
    if ids.is_empty() { return Err("no common SNPs".into()); }

    // Apply SNP limiting (sampling takes precedence if both are set)
    if let Some(n) = cfg.snp_sample_n {
        if ids.len() > n {
            let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);
            ids.shuffle(&mut rng);
            ids.truncate(n);
            ids.sort_unstable(); // keep plots/files in increasing ID order
        }
    } else if let Some(n) = cfg.snp_first_n {
        if ids.len() > n {
            ids.truncate(n);
        }
    }


    // Precompute ln1p(Red/Green) for features/r_min
    let mut ln_x: Vec<Vec<f64>> = Vec::with_capacity(s);
    let mut ln_y: Vec<Vec<f64>> = Vec::with_capacity(s);
    for sd in &samples {
        ln_x.push(sd.x_red.iter().map(|&v| ln1p_u16(v)).collect());
        ln_y.push(sd.y_grn.iter().map(|&v| ln1p_u16(v)).collect());
    }


    // --- Normalization setup (bead pools + mode + transforms) ---
    let bead_pools: HashMap<u32, u16> = if let Some(ref p) = cfg.bead_pools_csv {
        match load_bead_pools_csv(p) {
            Ok(m) => { eprintln!("NORMALIZATION: bead pools entries = {}", m.len()); m }
            Err(e) => { eprintln!("WARN: bead pools CSV unreadable: {} (using pool 0)", e); HashMap::new() }
        }
    } else { HashMap::new() };

    let norm_mode = if cfg.norm_illumina {
        if cfg.norm_affine {
            eprintln!("WARN: norm_illumina + norm_affine both set; using Illumina transform");
        }
        NormMode::IlluminaPool
    } else if cfg.norm_affine {
        NormMode::AffinePool
    } else {
        NormMode::None
    };

    let pool_affines = match norm_mode {
        NormMode::AffinePool => Some(compute_pool_affines(&samples, &ids, &bead_pools)),
        _ => None
    };
    let pool_illumina = match norm_mode {
        NormMode::IlluminaPool => Some(compute_pool_affines_illumina(&samples, &ids, &bead_pools)),
        _ => None
    };

    // r_min (0 => auto)
    let r_min = if cfg.r_min == 0 {
        let mut per_sample = Vec::with_capacity(s);
        for si in 0..s {
            let z: Vec<f64> = ln_x[si].iter().zip(&ln_y[si]).map(|(xr, yg)| xr + yg).collect();
            per_sample.push(rmin_from_lnxy_sum(&z));
        }
        let mut v = per_sample;
        v.sort_unstable();
        v[v.len() / 2] // median
    } else {
        cfg.r_min
    };

    // Prepare writers (calls & QC)
    let calls_path = out_dir.join("cohort_calls.csv");
    let qc_path = out_dir.join("cohort_qc_summary.csv");
    let mut w = Writer::from_path(&calls_path).map_err(|e| e.to_string())?;
    let mut header = vec!["SNP_ID".to_string()];
    header.extend(samples.iter().map(|s| s.name.clone()));
    w.write_record(&header).map_err(|e| e.to_string())?;

    // stats per sample
    #[derive(Default)]
    struct PS {
        total: u64,
        pass_raw: u64,
        nocall_raw: u64,
        nocall_p: u64,
        calls_tau: u64,
        calls_p090: u64,
    }
    let mut stats: Vec<PS> = (0..s).map(|_| PS::default()).collect();

    // Decide tau
    let mut tau_opt: Option<f64> = match cfg.p_mode {
        PMMode::Fixed(v) => Some(v),
        PMMode::Auto { .. } => None,
    };
    let alpha_default = match cfg.p_mode {
        PMMode::Auto { alpha } => alpha,
        _ => 0.005,
    };

    // Nice label for plots
    let feature_label = match cfg.feature_space {
        FeatureSpace::XYLog1p => "ln(R) vs ln(G)",
        FeatureSpace::ThetaR { w_theta, w_r } => {
            Box::leak(format!("θ·wθ={:.2} vs r·w_r={:.2}", w_theta, w_r).into_boxed_str())
        }
    };

    let m = ids.len();
    let block = 10_000.min(m).max(1);
    let mut plots_emitted: usize = 0;

    for chunk_start in (0..m).step_by(block) {
        let end = (chunk_start + block).min(m);

        #[derive(Clone)]
        struct Row {
            id: u32,
            raw: Vec<u32>,
            labs: Vec<usize>,
            post: Vec<f64>,
            means: Vec<(f64, f64)>,
        }

        let rows: Vec<Row> = (chunk_start..end)
            .into_par_iter()
            .map(|j| {
                let id = ids[j];
                let mut pts = Vec::with_capacity(s);
                let mut raw = Vec::with_capacity(s);
                for si in 0..s {
                    let p = *samples[si].pos.get(&id).unwrap();
                    // normalization on raw ints
                    let red = samples[si].x_red[p] as f64;
                    let grn = samples[si].y_grn[p] as f64;
                    let pool_id = bead_pools.get(&id).copied().unwrap_or(0u16);
                    let (nx, ny) = match norm_mode {
                        NormMode::None => (red, grn),
                        NormMode::AffinePool => {
                            let a = pool_affines.as_ref().unwrap().get(&pool_id);
                            apply_affine(a, red, grn)
                        }
                        NormMode::IlluminaPool => {
                            let a = pool_illumina.as_ref().unwrap().get(&pool_id);
                            apply_illumina_affine(a, red, grn)
                        }
                    };

                    // features from normalized ints via logs
                    let lnr = (nx + 1.0).ln();
                    let lng = (ny + 1.0).ln();
                    pts.push(to_features_from_logs(&cfg.feature_space, lnr, lng));

                    // Raw intensity sum still from originals (for r_min gate)
                    let rsum = (samples[si].x_red[p] as u32) + (samples[si].y_grn[p] as u32);
                    raw.push(rsum);
                }

                let model = choose_k(&pts, cfg.k_max, cfg.max_iters, cfg.tol, cfg.seed, cfg.restarts);

                let (mut labs, mut post) = (Vec::new(), Vec::new());
                predict(&model, &pts, &mut labs, &mut post);

                // Mahalanobis gate
                for si in 0..s {
                    let jlab = labs[si];
                    let m2 = maha2_diag(pts[si], model.means[jlab], model.vars[jlab]);
                    if m2.sqrt() > cfg.max_maha {
                        post[si] = 0.0; // force NoCall downstream
                    }
                }

                Row { id, raw, labs, post, means: model.means.clone() }
            })
            .collect();

        if tau_opt.is_none() {
            let mut ps = Vec::new();
            for r in &rows {
                for (si, &p) in r.post.iter().enumerate() {
                    if r.raw[si] >= r_min { ps.push(p); }
                }
            }
            let tau = auto_tau_from_posteriors(ps, alpha_default);
            tau_opt = Some(tau);
            eprintln!("AUTO τ via FDR α={:.4} -> τ={:.4}", alpha_default, tau);
        }
        let tau = tau_opt.unwrap();

        for r in rows {
            let mut out = Vec::with_capacity(1 + s);
            out.push(r.id.to_string());

            // Map cluster index -> genotype name (order by x-mean)
            let names: Vec<&'static str> = match r.means.len() {
                1 => {
                    let call = single_cloud_call(&cfg.feature_space, r.means[0]);
                    vec![call]
                }
                2 => {
                    let mut idx: Vec<(usize, f64)> =
                        r.means.iter().enumerate().map(|(i, m)| (i, m.0)).collect();
                    idx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    let mut n = vec![""; 2];
                    n[idx[0].0] = "BB"; n[idx[1].0] = "AA";
                    n
                }
                _ => {
                    let mut idx: Vec<(usize, f64)> =
                        r.means.iter().enumerate().map(|(i, m)| (i, m.0)).collect();
                    idx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    let mut n = vec![""; r.means.len()];
                    n[idx[0].0] = "BB"; n[idx[1].0] = "AB"; n[idx[2].0] = "AA";
                    n
                }
            };

            // Write calls & update stats
            for si in 0..s {
                let st = &mut stats[si];
                st.total += 1;
                if r.raw[si] < r_min {
                    st.nocall_raw += 1;
                    out.push("NoCall".into());
                    continue;
                }
                st.pass_raw += 1;

                if r.post[si] >= 0.90 { st.calls_p090 += 1; }

                if r.post[si] < tau {
                    st.nocall_p += 1;
                    out.push("NoCall".into());
                } else {
                    st.calls_tau += 1;
                    let jlab = r.labs[si];
                    out.push(names[jlab].into());
                }
            }
            w.write_record(out).map_err(|e| e.to_string())?;

            // Plots: first N SNPs only
            if cfg.plot_first_snps > 0 && plots_emitted < cfg.plot_first_snps {
                let mut pts = Vec::with_capacity(s);
                for si in 0..s {
                    let p = *samples[si].pos.get(&r.id).unwrap();
                    let red = samples[si].x_red[p] as f64;
                    let grn = samples[si].y_grn[p] as f64;
                    let pool_id = bead_pools.get(&r.id).copied().unwrap_or(0u16);
                    let (nx, ny) = match norm_mode {
                        NormMode::None => (red, grn),
                        NormMode::AffinePool => {
                            let a = pool_affines.as_ref().unwrap().get(&pool_id);
                            apply_affine(a, red, grn)
                        }
                        NormMode::IlluminaPool => {
                            let a = pool_illumina.as_ref().unwrap().get(&pool_id);
                            apply_illumina_affine(a, red, grn)
                        }
                    };
                    let lnr = (nx + 1.0).ln();
                    let lng = (ny + 1.0).ln();
                    pts.push(to_features_from_logs(&cfg.feature_space, lnr, lng));
                }

                // Build per-cluster genotype names for legend coloring
                let names_plot: Vec<&'static str> = names.clone();

                let plot_path_base = plots_dir.join(format!("snp_{:08}", r.id));
                let _ = write_cluster_scatter_png(
                    &plot_path_base,
                    r.id,
                    feature_label,
                    &pts,
                    &r.labs,
                    &r.means,
                    &names_plot,
                );
                plots_emitted += 1;
            }
        }
        w.flush().map_err(|e| e.to_string())?;
    }

    // QC summary
    let mut qcw = Writer::from_path(&qc_path).map_err(|e| e.to_string())?;
    qcw.write_record(&[
        "Sample","total_snps","pass_raw","nocall_raw","nocall_p_at_tau","calls_at_tau",
        "calls_at_p090","call_rate_tau","call_rate_p090","r_min_used","tau_used",
    ]).map_err(|e| e.to_string())?;
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
            st.calls_tau.to_string(),
            st.calls_p090.to_string(),
            format!("{:.6}", rate_tau),
            format!("{:.6}", rate_p),
            r_min.to_string(),
            format!("{:.4}", tau_report),
        ]).map_err(|e| e.to_string())?;
    }
    qcw.flush().map_err(|e| e.to_string())?;

    Ok((calls_path, qc_path))
}

// ---------- Plotting (AA=red, AB=blue, BB=green, legend top-right) ----------
fn write_cluster_scatter_png(
    out_path_base: &Path,
    snp_id: u32,
    feature_label: &str, // e.g., "ln(R) vs ln(G)" or "θ·wθ=1.00 vs r·w_r=0.45"
    pts: &[(f64, f64)],
    labels: &[usize],
    means: &[(f64, f64)],
    names: &[&'static str],       // cluster index -> "AA"/"AB"/"BB"
) -> Result<PathBuf, String> {
    let file = out_path_base.with_extension("png");
    let file_to_return = file.clone();
    let outfile_str: String = file.to_string_lossy().into_owned();

    // Axis bounds (include centers)
    let mut xmin = f64::INFINITY;
    let mut xmax = -f64::INFINITY;
    let mut ymin = f64::INFINITY;
    let mut ymax = -f64::INFINITY;
    for &(x, y) in pts.iter().chain(means.iter()) {
        xmin = xmin.min(x);
        xmax = xmax.max(x);
        ymin = ymin.min(y);
        ymax = ymax.max(y);
    }
    let padx = (xmax - xmin).abs().max(1.0) * 0.06;
    let pady = (ymax - ymin).abs().max(1.0) * 0.06;

    // Split the axis labels from the feature label
    let mut parts = feature_label.split(" vs ");
    let xlab = parts.next().unwrap_or("x");
    let ylab = parts.next().unwrap_or("y");

    let root = BitMapBackend::new(&outfile_str, (1000, 740)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| e.to_string())?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("SNP {} — {}", snp_id, feature_label),
            ("sans-serif", 26).into_font(),
        )
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(55)
        .build_cartesian_2d((xmin - padx)..(xmax + padx), (ymin - pady)..(ymax + pady))
        .map_err(|e| e.to_string())?;

    chart
        .configure_mesh()
        .x_desc(xlab)
        .y_desc(ylab)
        .label_style(("sans-serif", 14))
        .draw()
        .map_err(|e| e.to_string())?;

    // Helper: color by genotype name
    let color_for = |nm: &str| -> RGBColor {
        match nm {
            "AA" => RED,          // homozygous red
            "AB" => BLUE,         // heterozygous
            "BB" => GREEN,        // homozygous green
            _    => BLACK,
        }
    };

    // Draw one series per cluster so we can attach legend entries
    for (j, _mu) in means.iter().enumerate() {
        let nm = if j < names.len() { names[j] } else { "" };
        let col = color_for(nm);
        chart
            .draw_series(
                pts.iter()
                    .zip(labels.iter())
                    .filter_map(|(&(x, y), &lab)| if lab == j { Some((x, y)) } else { None })
                    .map(|(x, y)| Circle::new((x, y), 3, col.filled())),
            )
            .map_err(|e| e.to_string())?
            .label(if nm.is_empty() { format!("C{}", j) } else { nm.to_string() })
            .legend(move |(x, y)| Circle::new((x, y), 5, col.filled()));
    }

    // Mark cluster means
    chart
        .draw_series(means.iter().map(|&(x, y)| Cross::new((x, y), 7, &BLACK)))
        .map_err(|e| e.to_string())?;

    // Legend in top-right
    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .border_style(&BLACK)
        .background_style(WHITE.mix(0.85))
        .label_font(("sans-serif", 14))
        .draw()
        .map_err(|e| e.to_string())?;

    root.present().map_err(|e| e.to_string())?;
    Ok(file_to_return)
}
