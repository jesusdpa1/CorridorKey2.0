//! CubeCL kernel for channels-last RMSNorm (`RMSNorm` and `LayerNorm2d` in `corridorkey/blocks.py`).
//!
//! Because tensors stay in channels-last `[..., C]` order throughout the Rust pipeline, both
//! `RMSNorm` and `LayerNorm2d` normalize contiguous rows of length `dim`.

use cubecl::prelude::*;

use corridorkey_base::core::CkFloat;

/// Normalizes each of `num_rows` contiguous rows of width `dim`:
/// `output[r, c] = input[r, c] * rsqrt(mean_c(input[r, c]²) + eps) * weight[c]`.
/// `input` and `output` may alias.
#[cube(launch)]
pub fn rms_norm_kernel<F: CkFloat>(
    input: &[F],
    weight: &[F],
    output: &mut [F],
    num_rows: u32,
    dim: u32,
    inv_dim: F,
    eps: F,
) {
    let row = ABSOLUTE_POS as u32;
    if row < num_rows {
        let base = (row * dim) as usize;
        let d = dim as usize;
        let mut sum_sq = F::new(0.0f32);
        for c in 0..d {
            let v = input[base + c];
            sum_sq += v * v;
        }
        let inv_rms = F::new(1.0f32) / F::sqrt(sum_sq * inv_dim + eps);
        for c in 0..d {
            output[base + c] = input[base + c] * inv_rms * weight[c];
        }
    }
}
