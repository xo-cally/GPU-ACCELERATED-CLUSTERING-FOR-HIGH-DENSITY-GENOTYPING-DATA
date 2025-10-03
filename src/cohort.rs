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
use crate::adaptive::{Feedback,FbEntry,load_feedback,save_feedback,load_bad_snps,save_bad_snps};

/// Posterior mode for NoCall threshold
pub enum PMMode {
    Fixed(f64),
    Auto { alpha: f64 },
}

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
        if n == 0 { return Err("empty pair".into()); }
        let mut pos = HashMap::with_capacity(n);
        for (i, &id) in g.illumina_ids.iter().take(n).enumerate() { pos.insert(id, i); }
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

// ---- K=1 candidate (single-cloud SNPs) ----
fn fit_k1(points: &[(f64, f64)]) -> GmmDiag {
    let n = points.len().max(1);
    let (mut mx, mut my) = (0.0, 0.0);
    for &(x, y) in points { mx += x; my += y; }
    mx /= n as f64; my /= n as f64;

    let (mut vx, mut vy) = (0.0, 0.0);
    for &(x, y) in points {
        vx += (x - mx) * (x - mx);
        vy += (y - my) * (y - my);
    }
    vx = (vx / n as f64).max(1e-2);
    vy = (vy / n as f64).max(1e-2);

    // log-likelihood under single diagonal Gaussian
    let mut ll = 0.0;
    for &(x, y) in points {
        let dx = x - mx; let dy = y - my;
        ll += -0.5 * (dx * dx / vx + dy * dy / vy)
            - 0.5 * ((vx * vy).ln() + 2.0 * std::f64::consts::PI.ln());
    }

    GmmDiag { k:1, weights: vec![1.0], means: vec![(mx,my)], vars: vec![(vx,vy)], loglik: ll }
}

/// Compare K = 1, 2, 3 via BIC (capped by kmax)
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

/// When K=1, call genotype from BOTH axes via a θ-like projection.
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

pub struct CohortCfg {
    pub feature_space: FeatureSpace,
    pub r_min: u32, // 0 => auto (per-sample 1D-GMM median)
    pub p_mode: PMMode,
    pub max_maha: f64,
    pub k_max: usize,
    pub max_iters: usize,
    pub tol: f64,
    pub seed: u64,
    pub restarts: usize,
    // how many SNPs to plot from the beginning (0 = none)
    pub plot_first_snps: usize,

    // NEW: feedback/bad-SNP handling
    pub feedback_in: Option<PathBuf>,
    pub feedback_out: Option<PathBuf>,
    pub bad_snps_in: Option<PathBuf>,
    pub bad_snps_out: Option<PathBuf>,
    pub nocall_bad_thresh: f64, // e.g., 0.35
    pub drop_bad_snps: bool,    // true => skip “bad” rows in outputs/stats
}

pub fn run_from_pairs(
    pairs: &[(PathBuf, PathBuf)],
    out_dir: &Path,
    mut cfg: CohortCfg,
) -> Result<(PathBuf, PathBuf), String> {
    std::fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    let plots_dir = out_dir.join("plots");
    if cfg.plot_first_snps > 0 {
        std::fs::create_dir_all(&plots_dir).map_err(|e| e.to_string())?;
    }

    let samples = load_samples(pairs)?; let s = samples.len();
    if s < 2 {
        return Err("need >=2 samples".into());
    }
    let ids = common_ids(&samples);
    if ids.is_empty() {
        return Err("no common SNPs".into());
    }

    // Precompute ln1p(Red/Green) for r_min auto (and reuse for features)
    let mut ln_x: Vec<Vec<f64>> = Vec::with_capacity(s);
    let mut ln_y: Vec<Vec<f64>> = Vec::with_capacity(s);
    for sd in &samples {
        ln_x.push(sd.x_red.iter().map(|&v| (v as f64 + 1.0).ln()).collect());
        ln_y.push(sd.y_grn.iter().map(|&v| (v as f64 + 1.0).ln()).collect());
    }

    // r_min (auto per-sample via 1D GMM, then median)
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

    // Prepare writers (ONLY calls & QC csvs)
    let calls_path = out_dir.join("cohort_calls.csv");
    let qc_path = out_dir.join("cohort_qc_summary.csv");
    let mut w = Writer::from_path(&calls_path).map_err(|e| e.to_string())?;
    let mut header = vec!["SNP_ID".to_string()];
    header.extend(samples.iter().map(|s| s.name.clone()));
    w.write_record(&header).map_err(|e| e.to_string())?;

    // stats per sample (no Clone bound)
    #[derive(Default)]
    struct PS {
        total: u64,
        pass_raw: u64,
        nocall_raw: u64,
        nocall_p: u64,
        calls_tau: u64,
        calls_p090: u64,
        calls_aa_tau: u64,
        calls_ab_tau: u64,
        calls_bb_tau: u64,
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

    // NEW: feedback & bad SNP lists
    let mut feedback: Feedback = match &cfg.feedback_in {
        Some(p) => { eprintln!("FEEDBACK: loading {}", p.display()); load_feedback(p) }
        None => HashMap::new(),
    };
    let mut bad_snps_set: HashSet<u32> = match &cfg.bad_snps_in {
        Some(p) => { let s = load_bad_snps(p); eprintln!("ADAPTIVE: loaded {} bad SNPs", s.len()); s }
        None => HashSet::new(),
    };
    for (snp, fbe) in &feedback { if fbe.bad { bad_snps_set.insert(*snp); } }

    // streaming blocks
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
            names: Vec<&'static str>,
        }

        let rows: Vec<Row> = (chunk_start..end)
            .into_par_iter()
            .map(|j| {
                let id = ids[j];
                let mut pts = Vec::with_capacity(s);
                let mut raw = Vec::with_capacity(s);
                for si in 0..s {
                    let p = *samples[si].pos.get(&id).unwrap();
                    let lnr = ln_x[si][p];
                    let lng = ln_y[si][p];
                    pts.push(to_features_from_logs(&cfg.feature_space, lnr, lng));

                    let red = samples[si].x_red[p] as u32;
                    let grn = samples[si].y_grn[p] as u32;
                    raw.push(red + grn);
                }

                let model = choose_k(&pts, cfg.k_max, cfg.max_iters, cfg.tol, cfg.seed, cfg.restarts);

                let (mut labs, mut post) = (Vec::new(), Vec::new());
                predict(&model, &pts, &mut labs, &mut post);

                // Mahalanobis gate
                for si in 0..s {
                    let jlab = labs[si];
                    let m2 = maha2_diag(pts[si], model.means[jlab], model.vars[jlab]);
                    if m2.sqrt() > cfg.max_maha {
                        post[si] = 0.0;
                    }
                }

                // Map cluster index -> genotype name
                let names: Vec<&'static str> = match model.means.len() {
                    1 => {
                        let call = single_cloud_call(&cfg.feature_space, model.means[0]);
                        vec![call]
                    }
                    2 => {
                        let mut idx: Vec<(usize, f64)> =
                            model.means.iter().enumerate().map(|(i, m)| (i, m.0)).collect();
                        idx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                        let mut n = vec![""; 2];
                        n[idx[0].0] = "BB";
                        n[idx[1].0] = "AA";
                        n
                    }
                    _ => {
                        let mut idx: Vec<(usize, f64)> =
                            model.means.iter().enumerate().map(|(i, m)| (i, m.0)).collect();
                        idx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                        let mut n = vec![""; model.means.len()];
                        n[idx[0].0] = "BB";
                        n[idx[1].0] = "AB";
                        n[idx[2].0] = "AA";
                        n
                    }
                };

                Row { id, raw, labs, post, means: model.means.clone(), names }
            })
            .collect();

        // Learn τ on first chunk if not fixed; exclude “bad” SNPs from τ pool
        if tau_opt.is_none() {
            let mut ps = Vec::new();
            for r in &rows {
                if bad_snps_set.contains(&r.id) || feedback.get(&r.id).map(|f| f.bad).unwrap_or(false) {
                    continue;
                }
                for (si, &p) in r.post.iter().enumerate() {
                    if r.raw[si] >= r_min {
                        ps.push(p);
                    }
                }
            }
            let tau = auto_tau_from_posteriors(ps, alpha_default);
            tau_opt = Some(tau);
            eprintln!("AUTO τ via FDR α={:.4} -> τ={:.4}", alpha_default, tau);
        }
        let tau = tau_opt.unwrap();

        for r in rows {
            // final bad decision (feedback/list based)
            let bad_now = bad_snps_set.contains(&r.id) || feedback.get(&r.id).map(|f| f.bad).unwrap_or(false);

            // Update feedback “seen” for this SNP
            let fbe = feedback.entry(r.id).or_insert(FbEntry::default());
            fbe.seen += s as u64;

            if cfg.drop_bad_snps && bad_now {
                // still accumulate feedback on NoCalls (as “fully NoCall”)
                fbe.nocall_raw += r.raw.iter().filter(|&&rv| rv < r_min).count() as u64;
                // consider all as NoCall at τ (conservative learning)
                fbe.nocall_p += s as u64;
                // hard-mark bad to persist
                fbe.bad = true;
                bad_snps_set.insert(r.id);
                continue;
            }

            let mut out = Vec::with_capacity(1 + s);
            out.push(r.id.to_string());

            let mut nocall_p_cnt_for_this_row = 0u64;

            for si in 0..s {
                let st = &mut stats[si];
                st.total += 1;
                if r.raw[si] < r_min {
                    st.nocall_raw += 1;
                    st.nocall_p += 1;
                    out.push("NoCall".into());
                    continue;
                }
                st.pass_raw += 1;

                if r.post[si] >= 0.90 {
                    st.calls_p090 += 1;
                }

                if r.post[si] < tau {
                    st.nocall_p += 1;
                    nocall_p_cnt_for_this_row += 1;
                    out.push("NoCall".into());
                } else {
                    st.calls_tau += 1;
                    let jlab = r.labs[si];
                    let call = r.names[jlab];
                    match call {
                        "AA" => st.calls_aa_tau += 1,
                        "AB" => st.calls_ab_tau += 1,
                        "BB" => st.calls_bb_tau += 1,
                        _ => {}
                    }
                    out.push(call.into());
                }
            }

            // update feedback counts from this row
            fbe.nocall_raw += r.raw.iter().filter(|&&rv| rv < r_min).count() as u64;
            fbe.nocall_p += nocall_p_cnt_for_this_row;

            // mark bad if sustained high NoCall rate
            let rate_p = if fbe.seen > 0 { (fbe.nocall_p as f64) / (fbe.seen as f64) } else { 0.0 };
            if rate_p >= cfg.nocall_bad_thresh {
                fbe.bad = true;
                bad_snps_set.insert(r.id);
            }

            // Plots: first N kept SNPs only
            if cfg.plot_first_snps > 0 && plots_emitted < cfg.plot_first_snps {
                let mut pts = Vec::with_capacity(s);
                for si in 0..s {
                    let p = *samples[si].pos.get(&r.id).unwrap();
                    let lnr = ln_x[si][p];
                    let lng = ln_y[si][p];
                    pts.push(to_features_from_logs(&cfg.feature_space, lnr, lng));
                }
                let plot_path_base = plots_dir.join(format!("snp_{:08}", r.id));
                let _ = write_cluster_scatter_png(
                    &plot_path_base,
                    r.id,
                    feature_label,
                    &pts,
                    &r.labs,
                    &r.means,
                    &r.names,
                );
                plots_emitted += 1;
            }

            w.write_record(out).map_err(|e| e.to_string())?;
        }
        w.flush().map_err(|e| e.to_string())?;
    }

    // QC summary
    let mut qcw = Writer::from_path(&qc_path).map_err(|e| e.to_string())?;
    qcw.write_record(&[
        "Sample",
        "total_snps",
        "pass_raw",
        "nocall_raw",
        "nocall_p_at_tau",
        "calls_at_tau",
        "calls_at_p090",
        "call_rate_tau",
        "call_rate_p090",
        "calls_AA_tau",
        "calls_AB_tau",
        "calls_BB_tau",
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
            st.calls_tau.to_string(),
            st.calls_p090.to_string(),
            format!("{:.6}", rate_tau),
            format!("{:.6}", rate_p),
            st.calls_aa_tau.to_string(),
            st.calls_ab_tau.to_string(),
            st.calls_bb_tau.to_string(),
            r_min.to_string(),
            format!("{:.4}", tau_report),
        ])
        .map_err(|e| e.to_string())?;
    }
    qcw.flush().map_err(|e| e.to_string())?;

    // Persist adaptive state
    if let Some(p) = &cfg.bad_snps_out {
        if let Err(e) = save_bad_snps(p, &bad_snps_set) {
            eprintln!("WARN: saving bad SNPs failed: {}", e);
        } else {
            eprintln!("ADAPTIVE: bad SNP list saved -> {}", p.display());
        }
    }
    if let Some(p) = &cfg.feedback_out {
        if let Err(e) = save_feedback(p, &feedback) {
            eprintln!("WARN: saving feedback failed: {}", e);
        } else {
            eprintln!("FEEDBACK_OK -> {}", p.display());
        }
    }

    Ok((calls_path, qc_path))
}


// ---------- Plotting ----------
fn write_cluster_scatter_png(
    out_path_base: &Path,
    snp_id: u32,
    feature_label: &str, // "ln(R) vs ln(G)" or "θ·wθ=.. vs r·w_r=.."
    pts: &[(f64, f64)],
    labels: &[usize],
    means: &[(f64, f64)],
    names: &[&'static str],
) -> Result<PathBuf, String> {
    let file = out_path_base.with_extension("png");
    let file_to_return = file.clone();
    let outfile_str: String = file.to_string_lossy().into_owned();

    // Axis bounds (include centers)
    let mut xmin = f64::INFINITY; let mut xmax = f64::NEG_INFINITY;
    let mut ymin = f64::INFINITY; let mut ymax = f64::NEG_INFINITY;
    for &(x, y) in pts.iter().chain(means.iter()) {
        xmin = xmin.min(x); xmax = xmax.max(x);
        ymin = ymin.min(y); ymax = ymax.max(y);
    }
    if !xmin.is_finite() || !xmax.is_finite() || !ymin.is_finite() || !ymax.is_finite() {
        xmin = -1.0; xmax = 1.0; ymin = -1.0; ymax = 1.0;
    }
    if (xmax - xmin).abs() < 1e-9 { xmax = xmin + 1.0; }
    if (ymax - ymin).abs() < 1e-9 { ymax = ymin + 1.0; }

    let padx = (xmax - xmin) * 0.06;
    let pady = (ymax - ymin) * 0.06;

    // Split the axis labels from the feature label
    let mut parts = feature_label.split(" vs ");
    let xlab = parts.next().unwrap_or("x");
    let ylab = parts.next().unwrap_or("y");

    // Group points for color/legend by genotype name
    let mut aa = Vec::new(); let mut ab = Vec::new(); let mut bb = Vec::new();
    for (i, &(x, y)) in pts.iter().enumerate() {
        match names[labels[i]] {
            "AA" => aa.push((x, y)),
            "AB" => ab.push((x, y)),
            "BB" => bb.push((x, y)),
            _ => {}
        }
    }

    let root = BitMapBackend::new(&outfile_str, (1000, 740)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| e.to_string())?;

    let mut chart = ChartBuilder::on(&root)
        .caption(format!("SNP {} — {}", snp_id, feature_label), ("sans-serif", 26).into_font())
        .margin(20).x_label_area_size(45).y_label_area_size(55)
        .build_cartesian_2d((xmin - padx)..(xmax + padx), (ymin - pady)..(ymax + pady))
        .map_err(|e| e.to_string())?;

    chart.configure_mesh().x_desc(xlab).y_desc(ylab).label_style(("sans-serif", 14)).draw()
        .map_err(|e| e.to_string())?;

    // Color map: BB = red, AB = blue, AA = green
    let color_for = |name: &str| match name { "BB" => RED, "AB" => BLUE, "AA" => GREEN, _ => BLACK };

    let mut draw_group = |group_pts: &[(f64, f64)], gname: &str| -> Result<(), String> {
        if group_pts.is_empty() { return Ok(()); }
        let c = color_for(gname);
        chart.draw_series(group_pts.iter().map(|&(x, y)| Circle::new((x, y), 3, c.filled())))
            .map_err(|e| e.to_string())?
            .label(match gname {
                "AA" => "AA (homozygous green)",
                "AB" => "AB (heterozygous)",
                "BB" => "BB (homozygous red)",
                _ => gname,
            })
            .legend(move |(x, y)| Circle::new((x, y), 6, c.filled()));
        Ok(())
    };

    draw_group(&aa, "AA")?; draw_group(&ab, "AB")?; draw_group(&bb, "BB")?;

    chart.draw_series(means.iter().map(|&(x, y)| Cross::new((x, y), 7, &BLACK)))
        .map_err(|e| e.to_string())?;

    chart.configure_series_labels().border_style(&BLACK).background_style(WHITE.mix(0.85))
        .position(SeriesLabelPosition::UpperRight).label_font(("sans-serif", 16)).draw()
        .map_err(|e| e.to_string())?;

    root.present().map_err(|e| e.to_string())?;
    Ok(file_to_return)
}
