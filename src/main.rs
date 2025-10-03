use chrono::Local;
use std::path::PathBuf;

use genotyping_pipeline::pairs::find_pairs_sorted;
use genotyping_pipeline::model::FeatureSpace;
use genotyping_pipeline::cohort::{run_from_pairs, CohortCfg, PMMode};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Defaults (you can wire your richer CLI back in if you want the whole optimizer/feedback stack on top)
    let mut root = PathBuf::from("/dataC/idat");
    let mut out_dir = PathBuf::from(format!("results/{}", Local::now().format("%Y%m%d_%H%M")));

    // feature + gates defaults
    let mut feat = FeatureSpace::XYLog1p;
    let mut r_min: u32 = 0;                 // 0 => auto (per-sample 1D-GMM median)
    let mut p_mode: PMMode = PMMode::Fixed(0.90);
    let mut alpha = 0.005f64;               // when p_mode=Auto
    let mut max_maha = 6.0f64;              // radius gate (sqrt of chi^2_2)

    let (mut k_max, mut max_iters, mut tol, mut seed, mut restarts) =
        (3usize, 200usize, 1e-6f64, 42u64, 3usize);

    let mut plot_first_snps: usize = 0;  // default: no plots
    let mut first_pairs: Option<usize> = None; // cap number of sample pairs processed

    // --- CLI (light) ---
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--root"      => root    = PathBuf::from(args.next().unwrap()),
            "--out-dir"   => out_dir = PathBuf::from(args.next().unwrap()),
            "--feat" => {
                let v = args.next().unwrap();
                if v == "xy" {
                    feat = FeatureSpace::XYLog1p;
                } else if v == "thetar" {
                    let wt: f64 = args.next().unwrap().parse().unwrap();
                    let wr: f64 = args.next().unwrap().parse().unwrap();
                    feat = FeatureSpace::ThetaR { w_theta: wt, w_r: wr };
                }
            }
            "--r-min"     => r_min     = args.next().unwrap().parse().unwrap(),
            "--p-min" => {
                let v = args.next().unwrap();
                if v.eq_ignore_ascii_case("auto") {
                    p_mode = PMMode::Auto { alpha };
                } else {
                    p_mode = PMMode::Fixed(v.parse().unwrap());
                }
            }
            "--alpha"     => { 
                alpha = args.next().unwrap().parse().unwrap();
                if let PMMode::Auto { .. } = p_mode {
                    p_mode = PMMode::Auto { alpha };
                }
            }
            "--max-maha"  => max_maha  = args.next().unwrap().parse().unwrap(),
            "--k-max"     => k_max     = args.next().unwrap().parse().unwrap(),
            "--max-iters" => max_iters = args.next().unwrap().parse().unwrap(),
            "--tol"       => tol       = args.next().unwrap().parse().unwrap(),
            "--seed"      => seed      = args.next().unwrap().parse().unwrap(),
            "--restarts"  => restarts  = args.next().unwrap().parse().unwrap(),
            "--first"     => first_pairs = args.next().and_then(|v| v.parse().ok()),
            "--plot-first-snps" => plot_first_snps = args.next().unwrap().parse().unwrap(),
            _ => {}
        }
    }

    std::fs::create_dir_all(&out_dir)?;
    let mut pairs = find_pairs_sorted(&root);
    if pairs.is_empty() {
        println!("No IDAT pairs found under {}", root.display());
        return Ok(());
    }
    if let Some(n) = first_pairs { if pairs.len() > n { pairs.truncate(n); } }
    println!("Found {} pairs under {}", pairs.len(), root.display());

    let cfg = CohortCfg {
        feature_space: feat,
        r_min,
        p_mode,
        max_maha,
        k_max,
        max_iters,
        tol,
        seed,
        restarts,
        plot_first_snps,   
    };

    let (calls, qc) = run_from_pairs(&pairs, &out_dir, cfg)
        .map_err(|e| format!("cohort error: {e}"))?;
    println!("Calls: {}\nQC:    {}", calls.display(), qc.display());
    Ok(())
}
