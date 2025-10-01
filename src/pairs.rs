use std::collections::HashMap;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

fn stem_base(name: &str) -> Option<(&str, bool)> {
    let n = name.strip_suffix(".gz").unwrap_or(name);
    if let Some(b) = n.strip_suffix("_Grn.idat").or_else(|| n.strip_suffix("_Green.idat")) {
        return Some((b, true));
    }
    if let Some(b) = n.strip_suffix("_Red.idat") { return Some((b, false)); }
    None
}

pub fn find_pairs_sorted(root: &Path) -> Vec<(PathBuf, PathBuf)> {
    let mut groups: HashMap<String, (Option<PathBuf>, Option<PathBuf>)> = HashMap::new();
    for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
        let p = entry.path(); if !p.is_file() { continue; }
        let name = match p.file_name().and_then(|s| s.to_str()) { Some(s) => s, None => continue };
        if !(name.ends_with(".idat") || name.ends_with(".idat.gz")) { continue; }
        if let Some((base, is_green)) = stem_base(name) {
            let ent = groups.entry(base.to_string()).or_insert((None, None));
            if is_green { ent.0 = Some(p.to_path_buf()); } else { ent.1 = Some(p.to_path_buf()); }
        }
    }
    let mut out: Vec<(PathBuf, PathBuf)> = groups.into_iter()
        .filter_map(|(_b, (g, r))| match (g, r) { (Some(gg), Some(rr)) => Some((gg, rr)), _ => None })
        .collect();
    out.sort_by(|a, b| a.0.as_os_str().cmp(&b.0.as_os_str()));
    out
}
