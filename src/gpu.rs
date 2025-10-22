// src/gpu.rs
//! Batched EM (E+M) for 2D diagonal-cov GMM (K<=3).
//! E-step: one block per SNP, threads loop over samples with shared-memory reduction.
//! M-step: per-SNP, per-K update, then weight renorm.
//! cudarc = "0.11.9" verified.

#![allow(clippy::too_many_arguments)]

use std::os::raw::c_void;
use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, DevicePtr, DeviceSlice, LaunchAsync, LaunchConfig,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

// ====================== env helpers ======================
#[inline] fn gpu_debug() -> bool { std::env::var("GPU_DEBUG").as_deref() == Ok("1") }
#[inline] fn gpu_force() -> bool { std::env::var("GPU_FORCE").as_deref() == Ok("1") }
#[inline]
fn gpu_min_n() -> usize {
    std::env::var("GPU_MIN_N").ok().and_then(|s| s.parse().ok()).unwrap_or(4096)
}

// E-step threads per block: must be <= 256 to match kernel shared-memory arrays
const TPB_ESTEP: u32 = 256;

// ====================== device handle ======================
static DEV: OnceLock<Option<Arc<CudaDevice>>> = OnceLock::new();

fn device() -> Option<&'static Arc<CudaDevice>> {
    let _ = DEV.get_or_init(|| {
        if gpu_debug() { eprintln!("[GPU] init CUDA device 0"); }
        match CudaDevice::new(0) {
            Ok(d) => Some(d),
            Err(e) => { eprintln!("[GPU] CudaDevice::new(0) failed: {e}"); None }
        }
    });
    DEV.get().and_then(|o| o.as_ref())
}

// ====================== kernels ======================
pub const KERNEL_SRC: &str = r#"
extern "C" __global__
void estep_reduce_k3_linear_2d(
    const float* __restrict__ xs,   // [B*S] (chunk, SNP-major)
    const float* __restrict__ ys,   // [B*S]
    int S, int B, int K,
    int N_XY,                       // == B*S
    const float* __restrict__ w,    // [B*K]
    const float* __restrict__ mx,   // [B*K]
    const float* __restrict__ my,   // [B*K]
    const float* __restrict__ vx,   // [B*K]
    const float* __restrict__ vy,   // [B*K]
    int N_BK,                       // == B*K
    float* __restrict__ nk,         // [B*3]
    float* __restrict__ sx,         // [B*3]
    float* __restrict__ sy,         // [B*3]
    float* __restrict__ sxx,        // [B*3]
    float* __restrict__ syy,        // [B*3]
    int N_B3,                       // == B*3
    float* __restrict__ ll_out      // [B]
){
    const unsigned int snp = blockIdx.x;
    const unsigned int t   = threadIdx.x;
    if (blockDim.x > 256) return;
    if (S <= 0 || B <= 0 || K <= 0 || K > 3) return;
    if (N_XY != B * S) return;
    if (N_BK != B * K) return;
    if (N_B3 != B * 3) return;
    if (snp >= (unsigned)B) return;

    const int pk = (int)(snp * (unsigned)K);
    if (pk < 0 || pk + (K - 1) >= N_BK) return;

    float wj[3]  = {0.f,0.f,0.f};
    float mxj[3] = {0.f,0.f,0.f};
    float myj[3] = {0.f,0.f,0.f};
    float vxj[3] = {0.f,0.f,0.f};
    float vyj[3] = {0.f,0.f,0.f};
    #pragma unroll
    for (int j = 0; j < K; ++j) {
        const int idx = pk + j;
        wj[j]  = fmaxf(w [idx],  1e-12f);
        mxj[j] = mx[idx];
        myj[j] = my[idx];
        vxj[j] = fmaxf(vx[idx], 1e-8f);
        vyj[j] = fmaxf(vy[idx], 1e-8f);
    }

    float l_nk[3]  = {0.f,0.f,0.f};
    float l_sx[3]  = {0.f,0.f,0.f};
    float l_sy[3]  = {0.f,0.f,0.f};
    float l_sxx[3] = {0.f,0.f,0.f};
    float l_syy[3] = {0.f,0.f,0.f};
    float l_ll     = 0.f;

    const float LOG_2PI = 1.8378770664093453f;

    for (unsigned int i = t; i < (unsigned)S; i += blockDim.x) {
        const int idx_xy = (int)(snp * (unsigned)S + i);
        if (idx_xy < 0 || idx_xy >= N_XY) break;
        const float x = xs[idx_xy];
        const float y = ys[idx_xy];

        float logs[3] = { -1e30f, -1e30f, -1e30f };
        #pragma unroll
        for (int j = 0; j < K; ++j) {
            const float dx = x - mxj[j];
            const float dy = y - myj[j];
            const float quad = 0.5f * (dx*dx / vxj[j] + dy*dy / vyj[j]);
            const float norm = LOG_2PI + 0.5f * (logf(vxj[j]) + logf(vyj[j]));
            logs[j] = logf(wj[j]) - quad - norm;
        }
        float max_log = logs[0];
        #pragma unroll
        for (int j = 1; j < K; ++j) if (logs[j] > max_log) max_log = logs[j];

        float sum = 0.f;
        #pragma unroll
        for (int j = 0; j < K; ++j) sum += __expf(logs[j] - max_log);
        const float log_sum = max_log + logf(sum);

        #pragma unroll
        for (int j = 0; j < K; ++j) {
            const float r = __expf(logs[j] - log_sum);
            l_nk [j] += r;
            l_sx [j] += r * x;
            l_sy [j] += r * y;
            l_sxx[j] += r * x * x;
            l_syy[j] += r * y * y;
        }
        l_ll += log_sum;
    }

    __shared__ float sh_nk [3][256];
    __shared__ float sh_sx [3][256];
    __shared__ float sh_sy [3][256];
    __shared__ float sh_sxx[3][256];
    __shared__ float sh_syy[3][256];
    __shared__ float sh_ll[256];

    #pragma unroll
    for (int j = 0; j < K; ++j) {
        sh_nk [j][t] = l_nk [j];
        sh_sx [j][t] = l_sx [j];
        sh_sy [j][t] = l_sy [j];
        sh_sxx[j][t] = l_sxx[j];
        sh_syy[j][t] = l_syy[j];
    }
    sh_ll[t] = l_ll;
    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (t < stride) {
            #pragma unroll
            for (int j = 0; j < K; ++j) {
                sh_nk [j][t] += sh_nk [j][t + stride];
                sh_sx [j][t] += sh_sx [j][t + stride];
                sh_sy [j][t] += sh_sy [j][t + stride];
                sh_sxx[j][t] += sh_sxx[j][t + stride];
                sh_syy[j][t] += sh_syy[j][t + stride];
            }
            sh_ll[t] += sh_ll[t + stride];
        }
        __syncthreads();
    }

    if (t == 0) {
        const int base3 = (int)(snp * 3u);
        if (base3 >= 0 && base3 + (K - 1) < N_B3) {
            #pragma unroll
            for (int j = 0; j < K; ++j) {
                nk [base3 + j] = sh_nk [j][0];
                sx [base3 + j] = sh_sx [j][0];
                sy [base3 + j] = sh_sy [j][0];
                sxx[base3 + j] = sh_sxx[j][0];
                syy[base3 + j] = sh_syy[j][0];
            }
        }
        if ((int)snp >= 0 && (int)snp < B) {
            ll_out[snp] = sh_ll[0];
        }
    }
}

extern "C" __global__
void mstep_k3_batched(
    int S, int B, int K,
    const float* __restrict__ nk,   // [B*3]
    const float* __restrict__ sx,   // [B*3]
    const float* __restrict__ sy,   // [B*3]
    const float* __restrict__ sxx,  // [B*3]
    const float* __restrict__ syy,  // [B*3]
    int N_B3,                       // == B*3
    float* __restrict__ w_out,      // [B*K]
    float* __restrict__ mx_out,     // [B*K]
    float* __restrict__ my_out,     // [B*K]
    float* __restrict__ vx_out,     // [B*K]
    float* __restrict__ vy_out,     // [B*K]
    int N_BK                        // == B*K
){
    const unsigned int snp = blockIdx.x;
    if (snp >= (unsigned)B || S <= 0 || B <= 0 || K <= 0 || K > 3) return;

    const int pk3 = (int)(snp * 3u);
    const int pk  = (int)(snp * (unsigned)K);

    const int j = threadIdx.x;
    if (j < K) {
        const int stat_idx = pk3 + j;
        if (stat_idx >= 0 && stat_idx < N_B3 && pk + j < N_BK) {
            const float nkj = fmaxf(nk[stat_idx], 1e-8f);
            const float mx  = sx[stat_idx] / nkj;
            const float my  = sy[stat_idx] / nkj;
            const float vx  = fmaxf(sxx[stat_idx] / nkj - mx * mx, 1e-6f);
            const float vy  = fmaxf(syy[stat_idx] / nkj - my * my, 1e-6f);
            const float wj  = nkj / (float)S;

            mx_out[pk + j] = mx;
            my_out[pk + j] = my;
            vx_out[pk + j] = vx;
            vy_out[pk + j] = vy;
            w_out[pk + j]  = wj;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && pk + (K - 1) < N_BK) {
        float sum_w = 0.0f;
        for (int j2 = 0; j2 < K; j2++) sum_w += w_out[pk + j2];
        if (sum_w > 0.0f) {
            for (int j2 = 0; j2 < K; j2++) w_out[pk + j2] /= sum_w;
        } else {
            for (int j2 = 0; j2 < K; j2++) w_out[pk + j2] = 1.0f / (float)K;
        }
    }
}

extern "C" __global__
void bic_k_batched(
    const float* __restrict__ ll, // [B]
    int S, int B, int K,
    float* __restrict__ out       // [B]
){
    const unsigned int snp = blockIdx.x * blockDim.x + threadIdx.x;
    if ((int)snp >= B || S <= 0 || B <= 0 || K <= 0) return;
    const float p = (float)((K - 1) + 4 * K);
    out[snp] = -2.0f * ll[snp] + p * logf((float)S);
}
"#;

// ====================== module loader ======================
fn load_funcs(dev: &Arc<CudaDevice>) -> Option<(CudaFunction, CudaFunction, CudaFunction)> {
    const MODULE: &str = "gmm_k3_batched_mod";
    const F_E: &str = "estep_reduce_k3_linear_2d";
    const F_M: &str = "mstep_k3_batched";
    const F_B: &str = "bic_k_batched";

    if let (Some(e), Some(m), Some(b)) =
        (dev.get_func(MODULE, F_E), dev.get_func(MODULE, F_M), dev.get_func(MODULE, F_B))
    {
        if gpu_debug() { eprintln!("[GPU] kernels already loaded"); }
        return Some((e, m, b));
    }

    if gpu_debug() { eprintln!("[GPU] NVRTC compile kernels…"); }

    let mut opts = CompileOptions::default();
    opts.use_fast_math = Some(true);
    let arch_env = std::env::var("GPU_ARCH").ok();
    let arch = arch_env.as_deref().unwrap_or("compute_89");
    let leaked: &'static str = Box::leak(arch.to_string().into_boxed_str());
    opts.arch = Some(leaked);

    let ptx = match compile_ptx_with_opts(KERNEL_SRC, opts) {
        Ok(ptx) => ptx,
        Err(e) => { eprintln!("[GPU] NVRTC compile failed for arch={arch}: {e}"); return None; }
    };

    if let Err(e) = dev.load_ptx(ptx, MODULE, &[F_E, F_M, F_B]) {
        eprintln!("[GPU] Failed to load PTX: {e}");
        return None;
    }

    let e = dev.get_func(MODULE, F_E)?;
    let m = dev.get_func(MODULE, F_M)?;
    let b = dev.get_func(MODULE, F_B)?;
    if gpu_debug() { eprintln!("[GPU] kernels loaded successfully"); }
    Some((e, m, b))
}

// ====================== utils ======================
#[inline]
fn lengths_ok(
    xs: &[f32], ys: &[f32], s: usize, b: usize, k: usize,
    w: &[f32], mx: &[f32], my: &[f32], vx: &[f32], vy: &[f32],
) -> bool {
    let n_sb = match s.checked_mul(b) { Some(v) => v, None => return false };
    let n_bk = match b.checked_mul(k) { Some(v) => v, None => return false };
    if xs.len() != n_sb || ys.len() != n_sb {
        if gpu_debug() { eprintln!("[GPU] data length mismatch: xs={}, ys={}, expected={}", xs.len(), ys.len(), n_sb); }
        return false;
    }
    if w.len() != n_bk || mx.len() != n_bk || my.len() != n_bk || vx.len() != n_bk || vy.len() != n_bk {
        if gpu_debug() {
            eprintln!(
                "[GPU] param length mismatch: w={}, mx={}, my={}, vx={}, vy={}, expected={}",
                w.len(), mx.len(), my.len(), vx.len(), vy.len(), n_bk
            );
        }
        return false;
    }
    true
}

#[inline]
fn launch_and_sync(func: &CudaFunction, cfg: LaunchConfig, params: &mut [*mut c_void], dev: &Arc<CudaDevice>) -> Option<()> {
    if gpu_debug() {
        eprintln!("[GPU] Launch kernel: grid=({},{},{}), block=({},{},{})",
            cfg.grid_dim.0, cfg.grid_dim.1, cfg.grid_dim.2,
            cfg.block_dim.0, cfg.block_dim.1, cfg.block_dim.2);
    }
    if let Err(e) = unsafe { func.clone().launch(cfg, params) } {
        eprintln!("[GPU] kernel launch failed: {e}");
        return None;
    }
    if let Err(e) = dev.synchronize() {
        eprintln!("[GPU] device synchronize failed: {e}");
        return None;
    }
    Some(())
}

// ====================== Public API ======================

/// One EM iteration for a *batch* of SNPs (same S), fixed K<=3.
pub fn em_one_iter_k_batched(
    xs: &[f32], ys: &[f32],
    s: usize, b: usize, k: usize,
    w: &[f32], mx: &[f32], my: &[f32], vx: &[f32], vy: &[f32],
) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    if !(1..=3).contains(&k) || s == 0 || b == 0 {
        if gpu_debug() { eprintln!("[GPU] invalid parameters: k={k}, s={s}, b={b}"); }
        return None;
    }
    let total_points = s.checked_mul(b)?;
    if total_points < gpu_min_n() && !gpu_force() {
        if gpu_debug() { eprintln!("[GPU] skipping GPU: {total_points} < {}", gpu_min_n()); }
        return None;
    }
    if !lengths_ok(xs, ys, s, b, k, w, mx, my, vx, vy) { return None; }

    let dev = device()?;
    let (f_e, f_m, _f_b) = load_funcs(dev)?;

    let mut w2  = vec![0.0f32; b * k];
    let mut mx2 = vec![0.0f32; b * k];
    let mut my2 = vec![0.0f32; b * k];
    let mut vx2 = vec![0.0f32; b * k];
    let mut vy2 = vec![0.0f32; b * k];
    let mut ll  = vec![0.0f32; b];

    // safe casts for kernel ints
    let s_i: i32 = s.try_into().ok()?;
    let k_i: i32 = k.try_into().ok()?;

    // Chunk to cap device memory
    let max_chunk_size = 100usize;
    let mut snp0 = 0usize;
    while snp0 < b {
        let bc = std::cmp::min(max_chunk_size, b - snp0);
        if bc == 0 { break; }
        if gpu_debug() { eprintln!("[GPU] chunk SNPs [{}, {}) of {}", snp0, snp0 + bc, b); }

        let xs_off = snp0 * s;
        let pk_off = snp0 * k;

        // Debug sanity checks for chunk sizes
        debug_assert_eq!(bc * s, xs[xs_off..xs_off + bc * s].len(), "xs chunk size mismatch");
        debug_assert_eq!(bc * s, ys[xs_off..xs_off + bc * s].len(), "ys chunk size mismatch");
        debug_assert_eq!(bc * k, w [pk_off..pk_off + bc * k].len(), "w  chunk size mismatch");
        debug_assert_eq!(bc * k, mx[pk_off..pk_off + bc * k].len(), "mx chunk size mismatch");
        debug_assert_eq!(bc * k, my[pk_off..pk_off + bc * k].len(), "my chunk size mismatch");
        debug_assert_eq!(bc * k, vx[pk_off..pk_off + bc * k].len(), "vx chunk size mismatch");
        debug_assert_eq!(bc * k, vy[pk_off..pk_off + bc * k].len(), "vy chunk size mismatch");

        // Upload chunk inputs
        let d_xs: CudaSlice<f32> = dev.htod_copy(xs[xs_off..xs_off + bc * s].to_vec()).ok()?;
        let d_ys: CudaSlice<f32> = dev.htod_copy(ys[xs_off..xs_off + bc * s].to_vec()).ok()?;
        let d_w : CudaSlice<f32> = dev.htod_copy(w [pk_off..pk_off + bc * k].to_vec()).ok()?;
        let d_mx: CudaSlice<f32> = dev.htod_copy(mx[pk_off..pk_off + bc * k].to_vec()).ok()?;
        let d_my: CudaSlice<f32> = dev.htod_copy(my[pk_off..pk_off + bc * k].to_vec()).ok()?;
        let d_vx: CudaSlice<f32> = dev.htod_copy(vx[pk_off..pk_off + bc * k].to_vec()).ok()?;
        let d_vy: CudaSlice<f32> = dev.htod_copy(vy[pk_off..pk_off + bc * k].to_vec()).ok()?;

        // Reduction arrays
        let d_nk : CudaSlice<f32> = dev.alloc_zeros(bc * 3).ok()?;
        let d_sx : CudaSlice<f32> = dev.alloc_zeros(bc * 3).ok()?;
        let d_sy : CudaSlice<f32> = dev.alloc_zeros(bc * 3).ok()?;
        let d_sxx: CudaSlice<f32> = dev.alloc_zeros(bc * 3).ok()?;
        let d_syy: CudaSlice<f32> = dev.alloc_zeros(bc * 3).ok()?;
        let d_ll : CudaSlice<f32> = dev.alloc_zeros(bc).ok()?;

        // Outputs
        let d_w2 : CudaSlice<f32> = dev.alloc_zeros(bc * k).ok()?;
        let d_mx2: CudaSlice<f32> = dev.alloc_zeros(bc * k).ok()?;
        let d_my2: CudaSlice<f32> = dev.alloc_zeros(bc * k).ok()?;
        let d_vx2: CudaSlice<f32> = dev.alloc_zeros(bc * k).ok()?;
        let d_vy2: CudaSlice<f32> = dev.alloc_zeros(bc * k).ok()?;

        // --- E-step launch: one block per SNP, threads reduce samples (pointer-array) ---
        let tpb = TPB_ESTEP;
        debug_assert!(tpb <= 256, "TPB_ESTEP must be <= 256 to match kernel shared arrays");
        let lcfg_e = LaunchConfig {
            block_dim: (tpb, 1, 1),
            grid_dim: (bc as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let b_i: i32 = (bc as i32).try_into().ok()?;
        let n_xy: i32 = (bc * s).try_into().ok()?;
        let n_bk: i32 = (bc * k).try_into().ok()?;
        let n_b3: i32 = (bc * 3).try_into().ok()?;

        if gpu_debug() {
            eprintln!(
                "[GPU] E-step params: S={} B={} K={} N_XY={} N_BK={} N_B3={}, xs_off={}, pk_off={}",
                s_i, b_i, k_i, n_xy, n_bk, n_b3, xs_off, pk_off
            );
            eprintln!(
                "[GPU] Device d_* lens: xs={} ys={} w={} mx={} my={} vx={} vy={} nk={} sx={} sy={} sxx={} syy={} ll={}",
                d_xs.len(), d_ys.len(), d_w.len(), d_mx.len(), d_my.len(), d_vx.len(), d_vy.len(),
                d_nk.len(), d_sx.len(), d_sy.len(), d_sxx.len(), d_syy.len(), d_ll.len()
            );
        }

        // --- build raw pointers for kernel ABI (CUdeviceptr/u64 values) ---
        let mut xs_raw  = *d_xs.device_ptr();
        let mut ys_raw  = *d_ys.device_ptr();
        let mut w_raw   = *d_w.device_ptr();
        let mut mx_raw  = *d_mx.device_ptr();
        let mut my_raw  = *d_my.device_ptr();
        let mut vx_raw  = *d_vx.device_ptr();
        let mut vy_raw  = *d_vy.device_ptr();
        let mut nk_raw  = *d_nk.device_ptr();
        let mut sx_raw  = *d_sx.device_ptr();
        let mut sy_raw  = *d_sy.device_ptr();
        let mut sxx_raw = *d_sxx.device_ptr();
        let mut syy_raw = *d_syy.device_ptr();
        let mut ll_raw  = *d_ll.device_ptr();

        let mut s_i_mut  = s_i;
        let mut b_i_mut  = b_i;
        let mut k_i_mut  = k_i;
        let mut n_xy_mut = n_xy;
        let mut n_bk_mut = n_bk;
        let mut n_b3_mut = n_b3;

        let mut params_e: [*mut c_void; 19] = [
            &mut xs_raw  as *mut _ as *mut c_void,
            &mut ys_raw  as *mut _ as *mut c_void,
            &mut s_i_mut as *mut _ as *mut c_void,
            &mut b_i_mut as *mut _ as *mut c_void,
            &mut k_i_mut as *mut _ as *mut c_void,
            &mut n_xy_mut as *mut _ as *mut c_void,
            &mut w_raw   as *mut _ as *mut c_void,
            &mut mx_raw  as *mut _ as *mut c_void,
            &mut my_raw  as *mut _ as *mut c_void,
            &mut vx_raw  as *mut _ as *mut c_void,
            &mut vy_raw  as *mut _ as *mut c_void,
            &mut n_bk_mut as *mut _ as *mut c_void,
            &mut nk_raw  as *mut _ as *mut c_void,
            &mut sx_raw  as *mut _ as *mut c_void,
            &mut sy_raw  as *mut _ as *mut c_void,
            &mut sxx_raw as *mut _ as *mut c_void,
            &mut syy_raw as *mut _ as *mut c_void,
            &mut n_b3_mut as *mut _ as *mut c_void,
            &mut ll_raw  as *mut _ as *mut c_void,
        ];

        if launch_and_sync(&f_e, lcfg_e, &mut params_e, dev).is_none() {
            eprintln!("[GPU] E-step failed for chunk [{}, {})", snp0, snp0 + bc);
            return None;
        }

        // --- M-step (pointer-array) ---
        let lcfg_m = LaunchConfig {
            block_dim: (32, 1, 1),
            grid_dim: (bc as u32, 1, 1),
            shared_mem_bytes: 0
        };

        let mut w2_raw  = *d_w2.device_ptr();
        let mut mx2_raw = *d_mx2.device_ptr();
        let mut my2_raw = *d_my2.device_ptr();
        let mut vx2_raw = *d_vx2.device_ptr();
        let mut vy2_raw = *d_vy2.device_ptr();

        let mut params_m: [*mut c_void; 15] = [
            &mut s_i_mut  as *mut _ as *mut c_void,
            &mut b_i_mut  as *mut _ as *mut c_void,
            &mut k_i_mut  as *mut _ as *mut c_void,
            &mut nk_raw   as *mut _ as *mut c_void,
            &mut sx_raw   as *mut _ as *mut c_void,
            &mut sy_raw   as *mut _ as *mut c_void,
            &mut sxx_raw  as *mut _ as *mut c_void,
            &mut syy_raw  as *mut _ as *mut c_void,
            &mut n_b3_mut as *mut _ as *mut c_void,
            &mut w2_raw   as *mut _ as *mut c_void,
            &mut mx2_raw  as *mut _ as *mut c_void,
            &mut my2_raw  as *mut _ as *mut c_void,
            &mut vx2_raw  as *mut _ as *mut c_void,
            &mut vy2_raw  as *mut _ as *mut c_void,
            &mut n_bk_mut as *mut _ as *mut c_void,
        ];

        if launch_and_sync(&f_m, lcfg_m, &mut params_m, dev).is_none() {
            eprintln!("[GPU] M-step failed for chunk [{}, {})", snp0, snp0 + bc);
            return None;
        }

        // Copy results back from device
        let w2_chunk  = dev.dtoh_sync_copy(&d_w2 ).ok()?;
        let mx2_chunk = dev.dtoh_sync_copy(&d_mx2).ok()?;
        let my2_chunk = dev.dtoh_sync_copy(&d_my2).ok()?;
        let vx2_chunk = dev.dtoh_sync_copy(&d_vx2).ok()?;
        let vy2_chunk = dev.dtoh_sync_copy(&d_vy2).ok()?;
        let ll_chunk  = dev.dtoh_sync_copy(&d_ll ).ok()?;

        if gpu_debug() {
            eprintln!(
                "[GPU] DTOH chunk lens: w2={} mx2={} my2={} vx2={} vy2={} ll={}",
                w2_chunk.len(), mx2_chunk.len(), my2_chunk.len(),
                vx2_chunk.len(), vy2_chunk.len(), ll_chunk.len()
            );
            eprintln!(
                "[GPU] Host targets lens: w2={} mx2={} my2={} vx2={} vy2={} ll={}",
                w2.len(), mx2.len(), my2.len(), vx2.len(), vy2.len(), ll.len()
            );
            eprintln!(
                "[GPU] Copying host ranges: w2[{}..{}], ll[{}..{}]",
                pk_off, pk_off + bc * k, snp0, snp0 + bc
            );
        }

        // Safety asserts before host slice writes
        assert_eq!(w2_chunk.len(),  bc * k, "w2_chunk len mismatch (got {}, want {})", w2_chunk.len(), bc * k);
        assert_eq!(mx2_chunk.len(), bc * k, "mx2_chunk len mismatch");
        assert_eq!(my2_chunk.len(), bc * k, "my2_chunk len mismatch");
        assert_eq!(vx2_chunk.len(), bc * k, "vx2_chunk len mismatch");
        assert_eq!(vy2_chunk.len(), bc * k, "vy2_chunk len mismatch");
        assert_eq!(ll_chunk.len(),  bc,     "ll_chunk len mismatch");
        assert!(pk_off + bc * k <= w2.len(), "w2 host slice OOB: pk_off={} bc={} k={} len={}", pk_off, bc, k, w2.len());
        assert!(pk_off + bc * k <= mx2.len(), "mx2 host slice OOB");
        assert!(pk_off + bc * k <= my2.len(), "my2 host slice OOB");
        assert!(pk_off + bc * k <= vx2.len(), "vx2 host slice OOB");
        assert!(pk_off + bc * k <= vy2.len(), "vy2 host slice OOB");
        assert!(snp0 + bc <= ll.len(),        "ll host slice OOB: snp0={} bc={} len={}", snp0, bc, ll.len());

        // Copy into host outputs
        let pk_end = pk_off + bc * k;
        w2 [pk_off .. pk_end].copy_from_slice(&w2_chunk);
        mx2[pk_off .. pk_end].copy_from_slice(&mx2_chunk);
        my2[pk_off .. pk_end].copy_from_slice(&my2_chunk);
        vx2[pk_off .. pk_end].copy_from_slice(&vx2_chunk);
        vy2[pk_off .. pk_end].copy_from_slice(&vy2_chunk);
        ll [snp0 .. snp0 + bc].copy_from_slice(&ll_chunk);

        snp0 += bc;
    }

    Some((w2, mx2, my2, vx2, vy2, ll))
}

/// Compute BIC per SNP on GPU for fixed K.
pub fn bic_batched_gpu(s: usize, b: usize, k: usize, ll: &[f32]) -> Option<Vec<f32>> {
    if !(1..=3).contains(&k) || s == 0 || b == 0 {
        if gpu_debug() { eprintln!("[GPU] bic: invalid parameters k={k}, s={s}, b={b}"); }
        return None;
    }
    if ll.len() != b {
        if gpu_debug() { eprintln!("[GPU] bic: len(ll)={} != b={}", ll.len(), b); }
        return None;
    }

    let dev = device()?;
    let (_e, _m, f_b) = load_funcs(dev)?;

    let d_ll : CudaSlice<f32> = dev.htod_copy(ll.to_vec()).ok()?;
    let d_out: CudaSlice<f32> = dev.alloc_zeros(b).ok()?;

    let s_i: i32 = s.try_into().ok()?;
    let b_i: i32 = b.try_into().ok()?;
    let k_i: i32 = k.try_into().ok()?;

    let threads = 256u32;
    let blocks  = ((b as u32) + threads - 1) / threads;
    let lcfg = LaunchConfig {
        block_dim: (threads, 1, 1),
        grid_dim: (blocks, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut ll_raw  = *d_ll.device_ptr();
    let mut out_raw = *d_out.device_ptr();

    let mut s_i_mut = s_i;
    let mut b_i_mut = b_i;
    let mut k_i_mut = k_i;

    let mut params: [*mut c_void; 5] = [
        &mut ll_raw  as *mut _ as *mut c_void,  // const float* ll
        &mut s_i_mut as *mut _ as *mut c_void,
        &mut b_i_mut as *mut _ as *mut c_void,
        &mut k_i_mut as *mut _ as *mut c_void,
        &mut out_raw as *mut _ as *mut c_void,  // float* out
    ];

    if launch_and_sync(&f_b, lcfg, &mut params, dev).is_none() {
        eprintln!("[GPU] BIC kernel failed");
        return None;
    }

    dev.dtoh_sync_copy(&d_out).ok()
}

// ======= Single-SNP wrappers =======
pub fn em_one_iter_k3(
    points: &[(f64, f64)],
    weights: &[f64],
    means: &[(f64, f64)],
    vars: &[(f64, f64)],
) -> Option<(Vec<f64>, Vec<(f64, f64)>, Vec<(f64, f64)>, f64)> {
    let s = points.len();
    let k = means.len();
    if !(1..=3).contains(&k) || s == 0 { return None; }

    let xs: Vec<f32> = points.iter().map(|&(x, _)| x as f32).collect();
    let ys: Vec<f32> = points.iter().map(|&(_, y)| y as f32).collect();
    let w : Vec<f32> = weights.iter().map(|&v| v as f32).collect();
    let mx: Vec<f32> = means.iter().map(|m| m.0 as f32).collect();
    let my: Vec<f32> = means.iter().map(|m| m.1 as f32).collect();
    let vx: Vec<f32> = vars.iter().map(|v| v.0 as f32).collect();
    let vy: Vec<f32> = vars.iter().map(|v| v.1 as f32).collect();

    let (w2, mx2, my2, vx2, vy2, ll) = em_one_iter_k_batched(&xs, &ys, s, 1, k, &w, &mx, &my, &vx, &vy)?;
    let mut w_out = vec![0.0f64; k];
    let mut m_out = vec![(0.0f64, 0.0f64); k];
    let mut v_out = vec![(0.0f64, 0.0f64); k];
    for j in 0..k {
        w_out[j] = w2[j] as f64;
        m_out[j] = (mx2[j] as f64, my2[j] as f64);
        v_out[j] = (vx2[j] as f64, vy2[j] as f64);
    }
    Some((w_out, m_out, v_out, ll.get(0).copied().unwrap_or(0.0) as f64))
}

pub fn bic_gpu(n: usize, k: usize, ll: f64) -> Option<f64> {
    if !(1..=3).contains(&k) || n == 0 { return None; }
    let vals = bic_batched_gpu(n, 1, k, &[ll as f32])?;
    vals.first().copied().map(|v| v as f64)
}
