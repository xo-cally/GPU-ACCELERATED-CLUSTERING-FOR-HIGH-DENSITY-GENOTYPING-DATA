use rand::{rngs::StdRng, Rng, SeedableRng};
use std::cmp::Ordering;

// ---- Feature spaces ----
#[derive(Debug, Clone, Copy)]
pub enum FeatureSpace {
    /// (ln(R), ln(G))
    XYLog1p,
    /// (theta*w_theta, ln(R+G+1)*w_r) where theta = atan2(G,R)/π in [0,1]
    ThetaR { w_theta: f64, w_r: f64 },
}

#[inline]
fn ln1p_u16(v: u16) -> f64 {
    (v as f64 + 1.0).ln()
}

#[inline]
pub fn to_features_from_logs(fs: &FeatureSpace, lnr: f64, lng: f64) -> (f64, f64) {
    match *fs {
        FeatureSpace::XYLog1p => (lnr, lng),
        FeatureSpace::ThetaR { w_theta, w_r } => {
            let r = lnr.exp() - 1.0;
            let g = lng.exp() - 1.0;
            let theta = g.atan2(r) / std::f64::consts::PI; // [0,1]
            let rr = (r + g + 1.0).ln();
            (theta * w_theta, rr * w_r)
        }
    }
}

// ---- 2D diagonal-covariance GMM ----
#[inline]
fn lse(xs: &[f64]) -> f64 {
    let m = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() { return m; }
    m + xs.iter().map(|&v| (v - m).exp()).sum::<f64>().ln()
}

#[inline]
fn nlogpdf_diag(x: (f64, f64), mu: (f64, f64), var: (f64, f64)) -> f64 {
    let (vx, vy) = (var.0.max(1e-8), var.1.max(1e-8));
    let dx = x.0 - mu.0;
    let dy = x.1 - mu.1;
    -0.5 * (dx * dx / vx + dy * dy / vy)
        - 0.5 * ((vx * vy).ln() + 2.0 * std::f64::consts::PI.ln())
}

#[derive(Clone, Debug)]
pub struct GmmDiag {
    pub k: usize,
    pub weights: Vec<f64>,
    pub means: Vec<(f64, f64)>,
    pub vars: Vec<(f64, f64)>,
    pub loglik: f64,
}

fn kpp(points: &[(f64, f64)], k: usize, seed: u64) -> Vec<(f64, f64)> {
    let n = points.len().max(1);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut centers = Vec::with_capacity(k);
    centers.push(points[rng.gen_range(0..n)]);
    let mut d2 = vec![0.0; n];
    while centers.len() < k {
        for (i, &p) in points.iter().enumerate() {
            let mut best = f64::INFINITY;
            for &q in &centers {
                let dx = p.0 - q.0; let dy = p.1 - q.1;
                let dd = dx * dx + dy * dy;
                if dd < best { best = dd; }
            }
            d2[i] = best;
        }
        let sum: f64 = d2.iter().sum();
        if sum <= 0.0 { while centers.len() < k { centers.push(centers[0]); } break; }
        let mut r = rng.gen_range(0.0..sum);
        let mut idx = 0usize;
        for (i, &w) in d2.iter().enumerate() {
            r -= w; if r <= 0.0 { idx = i; break; }
        }
        centers.push(points[idx]);
    }
    centers
}

pub fn fit_diag(points: &[(f64, f64)], k: usize, max_iter: usize, tol: f64, seed: u64) -> GmmDiag {
    let n = points.len().max(1);

    let mut weights = vec![1.0 / k as f64; k];
    let mut means = kpp(points, k, seed);

    // global variance init
    let (mut mx, mut my) = (0.0, 0.0);
    for &(x, y) in points { mx += x; my += y; }
    mx /= n as f64; my /= n as f64;

    let (mut vx, mut vy) = (0.0, 0.0);
    for &(x, y) in points { vx += (x - mx) * (x - mx); vy += (y - my) * (y - my); }
    vx = (vx / n as f64).max(1e-2); vy = (vy / n as f64).max(1e-2);

    let mut vars = vec![(vx, vy); k];
    let mut loglik = f64::NEG_INFINITY;
    let mut logs = vec![0.0; k];

    for _ in 0..max_iter {
        // E-step accumulators
        let (mut nk, mut sx, mut sy, mut sxx, mut syy) =
            (vec![0.0; k], vec![0.0; k], vec![0.0; k], vec![0.0; k], vec![0.0; k]);
        let mut ll = 0.0;

        for &(x, y) in points {
            for j in 0..k { logs[j] = weights[j].max(1e-12).ln() + nlogpdf_diag((x, y), means[j], vars[j]); }
            let l = lse(&logs); ll += l;
            for j in 0..k {
                let r = (logs[j] - l).exp();
                nk[j] += r; sx[j] += r * x; sy[j] += r * y;
                sxx[j] += r * x * x; syy[j] += r * y * y;
            }
        }

        let mut moved = 0.0;
        for j in 0..k {
            let nkj = nk[j].max(1e-8);
            let mxn = sx[j] / nkj; let myn = sy[j] / nkj;
            moved += ((means[j].0 - mxn).powi(2) + (means[j].1 - myn).powi(2)).sqrt();
            means[j] = (mxn, myn);

            let vx = (sxx[j] / nkj - mxn * mxn).max(5e-6);
            let vy = (syy[j] / nkj - myn * myn).max(5e-6);
            vars[j] = (vx, vy);

            weights[j] = (nk[j] / n as f64).clamp(1e-12, 1.0);
        }
        let wsum: f64 = weights.iter().sum();
        for w in &mut weights { *w /= wsum; }

        if (ll - loglik).abs() <= tol { loglik = ll; break; }
        loglik = ll;
        if moved / (k as f64) < tol { break; }
    }

    GmmDiag { k, weights, means, vars, loglik }
}

pub fn fit_diag_restarts(points: &[(f64, f64)], k: usize, max_iter: usize, tol: f64, seed: u64, restarts: usize) -> GmmDiag {
    let mut best: Option<GmmDiag> = None;
    for r in 0..restarts {
        let s = seed.wrapping_add((r as u64) * 1469598103934665603);
        let m = fit_diag(points, k, max_iter, tol, s);
        best = Some(match best { None => m, Some(b) => if m.loglik > b.loglik { m } else { b } });
    }
    best.unwrap()
}

pub fn bic(m: &GmmDiag, n: usize) -> f64 {
    let p = (m.k as f64 - 1.0) + 4.0 * m.k as f64; // weights + means/vars
    -2.0 * m.loglik + p * (n as f64).ln()
}

pub fn predict(model: &GmmDiag, pts: &[(f64, f64)], out_lab: &mut Vec<usize>, out_p: &mut Vec<f64>) {
    out_lab.clear(); out_p.clear();
    let k = model.k; let mut logs = vec![0.0; k];
    for &(x, y) in pts {
        for j in 0..k { logs[j] = model.weights[j].max(1e-12).ln() + nlogpdf_diag((x, y), model.means[j], model.vars[j]); }
        let l = lse(&logs);
        let mut bj = 0usize; let mut bp = f64::NEG_INFINITY;
        for j in 0..k {
            let lp = logs[j] - l;
            if lp > bp { bp = lp; bj = j; }
        }
        out_lab.push(bj); out_p.push(bp.exp());
    }
}

#[inline]
pub fn maha2_diag(p: (f64, f64), mu: (f64, f64), var: (f64, f64)) -> f64 {
    let (vx, vy) = (var.0.max(1e-8), var.1.max(1e-8));
    let dx = p.0 - mu.0; let dy = p.1 - mu.1;
    dx * dx / vx + dy * dy / vy
}

// ---- r_min auto & FDR-style tau ----
pub fn rmin_from_lnxy_sum(z: &[f64]) -> u32 {
    // 2-comp 1D GMM on z = ln((R+1)+(G+1))
    #[derive(Clone, Copy, Debug)]
    struct G1 { w0:f64, mu0:f64, s0:f64, w1:f64, mu1:f64, s1:f64 }

    fn em(z: &[f64]) -> G1 {
        use std::cmp::Ordering;
        let n = z.len().max(1);
        let mut v = z.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let q = |p: f64| v[((n as f64 * p).floor() as usize).min(n - 1)];
        let (mut w0, mut w1) = (0.5, 0.5);
        let (mut mu0, mut mu1) = (q(0.25), q(0.75));
        let mean: f64 = z.iter().sum::<f64>() / (n as f64);
        let var: f64 = z.iter().map(|&t| (t - mean) * (t - mean)).sum::<f64>() / (n as f64);
        let (mut s0, mut s1) = (var.sqrt().max(1e-3), var.sqrt().max(1e-3));
        let mut prev = f64::NEG_INFINITY;
        for _ in 0..50 {
            let (mut r0s, mut r1s, mut r0t, mut r1t, mut r0tt, mut r1tt) = (0.0,0.0,0.0,0.0,0.0,0.0);
            let mut ll = 0.0;
            for &t in z {
                let n0 = ((-(t - mu0).powi(2) / (2.0 * s0 * s0)) - s0.ln()).exp() * w0;
                let n1 = ((-(t - mu1).powi(2) / (2.0 * s1 * s1)) - s1.ln()).exp() * w1;
                let den = (n0 + n1).max(1e-300);
                let r0 = n0 / den; let r1 = 1.0 - r0;
                ll += den.ln();
                r0s += r0; r1s += r1; r0t += r0 * t; r1t += r1 * t; r0tt += r0 * t * t; r1tt += r1 * t * t;
            }
            w0 = (r0s / (n as f64)).clamp(1e-6, 1.0 - 1e-6); w1 = 1.0 - w0;
            mu0 = r0t / (r0s.max(1e-8)); mu1 = r1t / (r1s.max(1e-8));
            let v0 = (r0tt / (r0s.max(1e-8)) - mu0 * mu0).max(1e-6);
            let v1 = (r1tt / (r1s.max(1e-8)) - mu1 * mu1).max(1e-6);
            s0 = v0.sqrt(); s1 = v1.sqrt();
            if (ll - prev).abs() < 1e-6 { break; }
            prev = ll;
        }
        if mu0 <= mu1 { G1 { w0, mu0, s0, w1, mu1, s1 } }
        else { G1 { w0:w1, mu0:mu1, s0:s1, w1:w0, mu1:mu0, s1:s0 } }
    }

    fn quad(a:f64,b:f64,c:f64)->Option<(f64,f64)>{
        if a.abs()<1e-12{ if b.abs()<1e-12{return None;} let t=-c/b; return Some((t,t)); }
        let d=b*b-4.0*a*c; if d<0.0 { return None; }
        let s=d.sqrt(); Some(((-b - s)/(2.0*a), (-b + s)/(2.0*a)))
    }

    let g = em(z);
    let (w0, mu0, s0, w1, mu1, s1) = (g.w0, g.mu0, g.s0, g.w1, g.mu1, g.s1);

    let a = 0.5 * (1.0 / (s1 * s1) - 1.0 / (s0 * s0));
    let b = (-mu1 / (s1 * s1) + mu0 / (s0 * s0)) * 2.0;
    let c = 0.5 * (mu1 * mu1 / (s1 * s1) - mu0 * mu0 / (s0 * s0)) + (w1 * s0 / (w0 * s1)).ln();

    let mut t = (mu0 + mu1) * 0.5;
    if let Some((r1, r2)) = quad(a, b, c) {
        if (r1 >= mu0 && r1 <= mu1) || (r1 >= mu1 && r1 <= mu0) { t = r1 } else { t = r2 }
    }

    let raw = (t.exp() - 1.0).max(0.0);
    raw.round().clamp(300.0, 3000.0) as u32
}

pub fn auto_tau_from_posteriors(ps: Vec<f64>, alpha: f64) -> f64 {
    if ps.is_empty() { return 0.99; }
    let mut es: Vec<f64> = ps.into_iter().map(|p| (1.0 - p).clamp(0.0, 1.0)).collect();
    es.sort_by(|a, b| a.partial_cmp(&b).unwrap_or(Ordering::Equal));
    let (mut sum, mut k) = (0.0, 0usize);
    for (i, &e) in es.iter().enumerate() {
        sum += e;
        let mean = sum / ((i + 1) as f64);
        if mean <= alpha { k = i + 1; } else { break; }
    }
    if k == 0 { return 0.99; }
    let tau = 1.0 - es[k - 1];
    tau.clamp(0.85, 0.9999)
}
