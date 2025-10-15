// src/adaptive.rs
use std::collections::{HashMap, HashSet};
use std::path::Path;

#[derive(Clone, Debug, Default)]
pub struct FbEntry {
    pub seen: u64,
    pub nocall_raw: u64,
    pub nocall_p: u64,
    pub bad: bool,
}
pub type Feedback = HashMap<u32, FbEntry>;

pub fn load_feedback(path: &Path) -> Feedback {
    let mut map: Feedback = HashMap::new();
    if !path.exists() { return map; }
    if let Ok(mut rdr) = csv::ReaderBuilder::new().has_headers(true).from_path(path) {
        for rec in rdr.records().flatten() {
            let snp_id: u32 = rec.get(0).unwrap_or("0").parse().unwrap_or(0);
            let seen: u64 = rec.get(1).unwrap_or("0").parse().unwrap_or(0);
            let nocall_raw: u64 = rec.get(2).unwrap_or("0").parse().unwrap_or(0);
            let nocall_p: u64 = rec.get(3).unwrap_or("0").parse().unwrap_or(0);
            let bad: bool = rec.get(4).unwrap_or("false").parse().unwrap_or(false);
            if snp_id != 0 {
                map.insert(snp_id, FbEntry { seen, nocall_raw, nocall_p, bad });
            }
        }
    }
    map
}

pub fn save_feedback(path: &Path, fb: &Feedback) -> Result<(), String> {
    let mut w = csv::WriterBuilder::new().has_headers(true)
        .from_path(path).map_err(|e| e.to_string())?;
    w.write_record(&["snp_id","seen","nocall_raw","nocall_p","bad"])
        .map_err(|e| e.to_string())?;
    let mut keys: Vec<u32> = fb.keys().cloned().collect();
    keys.sort_unstable();
    for k in keys {
        let v = &fb[&k];
        w.write_record(&[
            k.to_string(),
            v.seen.to_string(),
            v.nocall_raw.to_string(),
            v.nocall_p.to_string(),
            v.bad.to_string(),
        ]).map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())
}

pub fn load_bad_snps(path: &Path) -> HashSet<u32> {
    let mut set = HashSet::new();
    if !path.exists() { return set; }
    if let Ok(s) = std::fs::read_to_string(path) {
        for t in s.lines().map(|l| l.trim()) {
            if t.is_empty() || t.starts_with('#') { continue; }
            if let Ok(id) = t.parse::<u32>() { set.insert(id); }
        }
    }
    set
}

pub fn save_bad_snps(path: &Path, ids: &std::collections::HashSet<u32>) -> Result<(), String> {
    let mut lines: Vec<String> = ids.iter().cloned().map(|v| v.to_string()).collect();
    lines.sort();
    std::fs::write(path, lines.join("\n")).map_err(|e| e.to_string())
}