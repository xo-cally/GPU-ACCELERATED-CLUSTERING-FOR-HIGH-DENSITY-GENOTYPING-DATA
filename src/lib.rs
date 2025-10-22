// src/lib.rs
pub mod idat;
pub mod pairs;
pub mod model;
pub mod cohort;
pub mod preflight;
pub mod adaptive;
pub mod profiler;

#[cfg(feature = "gpu")]
pub mod gpu; 
