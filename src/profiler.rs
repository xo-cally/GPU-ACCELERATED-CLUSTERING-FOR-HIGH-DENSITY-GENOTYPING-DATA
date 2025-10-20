// src/profiler.rs
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};

/// A single timing record (flat; we don't need a tree)
#[derive(Clone, Debug)]
pub struct Rec {
    pub label: &'static str,
    pub start: Instant,
    pub dur: Duration,
}

#[derive(Default)]
pub struct Profiler {
    recs: Vec<Rec>,
    // Active starts: parallel-safe if you don't share the Profiler across threads.
    starts: Vec<(&'static str, Instant)>,
}

impl Profiler {
    pub fn new() -> Self { Self { recs: Vec::new(), starts: Vec::new() } }

    /// Start a span and return its index (use this if you prefer manual start/end).
    pub fn start(&mut self, label: &'static str) -> usize {
        let idx = self.starts.len();
        self.starts.push((label, Instant::now()));
        idx
    }
    /// End a span opened with `start`.
    pub fn end(&mut self, idx: usize) {
        let (label, t0) = self.starts[idx];
        let dur = t0.elapsed();
        self.recs.push(Rec { label, start: t0, dur });
    }

    /// RAII scope helper: times from creation to drop.
    pub fn scope<'a>(&'a mut self, label: &'static str) -> Scope<'a> {
        Scope { prof: self, label, t0: Instant::now(), done: false }
    }

    /// Write raw records and an aggregated summary CSV.
    /// Output: timings.csv (raw) and timings_summary.csv (per-label aggregates)
    pub fn write_csv(&self, dir: &Path) -> Result<(), String> {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        let raw = dir.join("timings.csv");
        let sum = dir.join("timings_summary.csv");

        // Raw
        let mut f = File::create(&raw).map_err(|e| e.to_string())?;
        writeln!(f, "label,ms").map_err(|e| e.to_string())?;
        for r in &self.recs {
            let ms = r.dur.as_secs_f64() * 1000.0;
            writeln!(f, "{},{}", r.label, format!("{:.3}", ms)).map_err(|e| e.to_string())?;
        }

        // Aggregate without HashMap (keep it simple; stable labels in code)
        #[derive(Clone)]
        struct Agg { label: &'static str, total_ms: f64, count: u64 }
        let mut aggs: Vec<Agg> = Vec::new();
        'outer: for r in &self.recs {
            for a in &mut aggs {
                if a.label == r.label {
                    a.total_ms += r.dur.as_secs_f64() * 1000.0;
                    a.count += 1;
                    continue 'outer;
                }
            }
            aggs.push(Agg { label: r.label, total_ms: r.dur.as_secs_f64() * 1000.0, count: 1 });
        }
        // Sort by total time desc
        aggs.sort_by(|a, b| b.total_ms.partial_cmp(&a.total_ms).unwrap());

        let mut g = File::create(&sum).map_err(|e| e.to_string())?;
        writeln!(g, "label,total_ms,count,avg_ms").map_err(|e| e.to_string())?;
        for a in &aggs {
            let avg = a.total_ms / (a.count as f64);
            writeln!(g, "{},{:.3},{},{}",
                a.label, a.total_ms, a.count, format!("{:.3}", avg)
            ).map_err(|e| e.to_string())?;
        }

        // Also dump a human-friendly summary to stderr
        eprintln!("\n=== TIMING SUMMARY (ms) ===");
        for a in &aggs {
            let avg = a.total_ms / (a.count as f64);
            eprintln!(
                "{:28} total {:10.1}  count {:6}  avg {:8.2}",
                a.label, a.total_ms, a.count, avg
            );
        }
        eprintln!("===========================\n");
        Ok(())
    }
}

pub struct Scope<'a> {
    prof: &'a mut Profiler,
    label: &'static str,
    t0: Instant,
    done: bool,
}

impl<'a> Drop for Scope<'a> {
    fn drop(&mut self) {
        if self.done { return; }
        let dur = self.t0.elapsed();
        self.prof.recs.push(Rec { label: self.label, start: self.t0, dur });
        self.done = true;
    }
}