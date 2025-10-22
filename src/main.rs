// src/main.rs
use chrono::Local;
use std::path::PathBuf;

use genotyping_pipeline::pairs::find_pairs_sorted;
use genotyping_pipeline::model::ThetaR;
use genotyping_pipeline::cohort::{run_from_pairs, CohortCfg, PMMode};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ===== Defaults =====
    let mut root = PathBuf::from("/dataC/idat");
    let mut out_dir = PathBuf::from(format!("results/{}", Local::now().format("%Y%m%d_%H%M")));

    // ThetaR only
    let mut feat = ThetaR { w_theta: 1.0, w_r: 0.45 };

    // ===== Fixed gates (pinned) =====
    let mut r_min: u32 = 500;                         // fixed r_min
    let mut p_mode: PMMode = PMMode::Fixed(0.85);     // fixed p_min (τ)

    // Other gates/knobs
    let mut max_maha = 6.0f64;

    let (mut k_max, mut max_iters, mut tol, mut seed, mut restarts) =
        (3usize, 200usize, 1e-6f64, 42u64, 3usize);

    let mut plot_first_snps: usize = 20;
    let mut first_pairs: Option<usize> = None;

    // SNP limiting
    let mut snp_first_n: Option<usize> = None;
    let mut snp_sample_n: Option<usize> = None;

    // ===== Learning / persistence (KEEP ON) =====
    let mut feedback_in: Option<PathBuf> = None;
    let mut feedback_out: Option<PathBuf> = None;
    let mut bad_snps_in: Option<PathBuf> = None;
    let mut bad_snps_out: Option<PathBuf> = None;
    let mut drop_bad_snps = true;           // keep excluding known-bad SNPs
    let mut nocall_bad_thresh: f64 = 0.35;  // mark SNP as bad if NoCall@τ rate ≥ 35%

    // --- CLI ---
    let mut alpha = 0.005f64; // only used if --p-min auto
    let mut args = std::env::args().skip(1);
    let mut debug_snp: Option<u32> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--root"      => root    = PathBuf::from(args.next().unwrap()),
            "--out-dir"   => out_dir = PathBuf::from(args.next().unwrap()),

            // ThetaR only (compat flag)
            "--feat" => {
                let v = args.next().unwrap();
                if v != "thetar" {
                    eprintln!("WARN: only 'thetar' is supported; using ThetaR.");
                }
            }
            "--thetar" => {
                let wt: f64 = args.next().unwrap().parse().unwrap();
                let wr: f64 = args.next().unwrap().parse().unwrap();
                feat = ThetaR { w_theta: wt, w_r: wr };
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
            "--max-snps"  => snp_first_n = args.next().and_then(|v| v.parse().ok()),
            "--sample-snps" => snp_sample_n = args.next().and_then(|v| v.parse().ok()),

            // Learning flags
            "--feedback-in"   => { feedback_in   = Some(PathBuf::from(args.next().unwrap())); }
            "--feedback-out"  => { feedback_out  = Some(PathBuf::from(args.next().unwrap())); }
            "--bad-snps-in"   => { bad_snps_in   = Some(PathBuf::from(args.next().unwrap())); }
            "--bad-snps-out"  => { bad_snps_out  = Some(PathBuf::from(args.next().unwrap())); }
            "--drop-bad-snps" => { drop_bad_snps = true; }
            "--nocall-bad-thresh" => { nocall_bad_thresh = args.next().unwrap().parse().unwrap(); }

            "--debug-snp" => { debug_snp = args.next().and_then(|v| v.parse().ok()); }

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
    println!("Plots: first {} SNPs will be rendered into {}/plots/", plot_first_snps, out_dir.display());

    match (snp_sample_n, snp_first_n) {
        (Some(n), _) => println!("SNPs: sampling {} (seed={}) from common set", n, seed),
        (None, Some(n)) => println!("SNPs: taking first {} (lexicographic Illumina IDs)", n),
        _ => println!("SNPs: using all common SNPs"),
    }

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

        // SNP limiting
        snp_first_n,
        snp_sample_n,

        // Learning / persistence
        feedback_in,
        feedback_out,
        bad_snps_in,
        bad_snps_out,
        drop_bad_snps,
        nocall_bad_thresh,

        debug_snp,
    };

    let (calls, qc) = run_from_pairs(&pairs, &out_dir, cfg)
        .map_err(|e| format!("cohort error: {e}"))?;
    println!("Calls: {}\nQC:    {}", calls.display(), qc.display());
    Ok(())
}
