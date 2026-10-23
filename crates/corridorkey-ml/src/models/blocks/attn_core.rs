//! Shared attention weights (`AttentionCore`) and shifted-window region group helpers
//! (`ShiftedRegionGroups`, `sdpa_with_region_groups`) used across 2D and 3D attention blocks.

use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use burn_tensor::{Tensor, module::attention, ops::AttentionModuleOptions};
use corridorkey_base::core::{
    CkFloat,
    buffer::{empty, upload},
};
use corridorkey_core::{CkError, CkResult, compute::LaunchGeometry};
use cubecl::{prelude::BufferArg, server::Handle};

use crate::{
    hub::SafetensorsFile,
    models::blocks::{
        kernels::{
            add_row_bias_kernel, launch_sdpa_window, merge_heads_gate_kernel,
            prep_qkv_norm_rope_kernel,
        },
        local_mix::QkvDepthwiseSpatial,
        mlp::transpose_2d,
        norm::DEFAULT_RMS_NORM_EPS,
    },
    runtime::{MlContext, elem_dtype, into_contiguous_cube},
};

/// Maximum sequence length where [`launch_sdpa_window`] is preferred over `cubek::attention` even
/// when `n_tokens` is a multiple of 32.
pub const WINDOW_SDPA_MAX_TOKENS: usize = 256;

/// Dispatches unmasked SDPA (`[batches, num_heads, n_tokens, head_dim]`):
/// - Uses `burn_tensor::module::attention` (`BlackboxAccelerated` FlashAttention) when `n_tokens > 256`,
///   `n_tokens` is divisible by `32`, and the device exposes CMMA Tensor Cores.
/// - Otherwise uses our single-launch fused shared-memory [`launch_sdpa_window`] kernel, avoiding
///   `cubek-attention`'s 5-kernel global-memory `attention_fallback`.
pub fn run_unmasked_sdpa<F: CkFloat>(
    ctx: &MlContext,
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
) -> CkResult<Tensor<4>> {
    let [batches, num_heads, n_tokens, head_dim] = q.dims();
    let has_cmma = !ctx.client().properties().features.matmul.cmma.is_empty();
    if n_tokens > WINDOW_SDPA_MAX_TOKENS && n_tokens.is_multiple_of(32) && has_cmma {
        return Ok(attention(q, k, v, None, None, AttentionModuleOptions::default()));
    }
    let c_q = into_contiguous_cube(q)?;
    let c_k = into_contiguous_cube(k)?;
    let c_v = into_contiguous_cube(v)?;
    let out = launch_sdpa_window::<F>(
        ctx.client(),
        c_q.handle,
        c_k.handle,
        c_v.handle,
        None,
        batches,
        num_heads,
        n_tokens,
        head_dim,
    )?;
    Ok(ctx.tensor_from_handle(
        out,
        [batches, num_heads, n_tokens, head_dim],
        elem_dtype::<F>(),
    ))
}

/// Shared device-resident weights for `FullSpatialAttention2D`, `SwinAttention2D`,
/// `CSwinAttention2D`, `SwinAttention3D`, `CSwinAttention3D`, and `FullAttention3D`.
#[derive(Clone, Debug)]
pub struct AttentionCore {
    pub qkv_w_t: Tensor<2>,
    pub qkv_b: Handle,
    pub qkv_dw: Option<QkvDepthwiseSpatial>,
    pub q_norm_w: Handle,
    pub k_norm_w: Handle,
    pub gate_w_t: Tensor<2>,
    pub proj_w_t: Tensor<2>,
    pub proj_b: Handle,
    pub dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl AttentionCore {
    #[allow(clippy::too_many_arguments)]
    pub fn from_slices<F: CkFloat>(
        ctx: &MlContext,
        dim: usize,
        num_heads: usize,
        qkv_w: &[F],
        qkv_b: &[F],
        qkv_dw: Option<QkvDepthwiseSpatial>,
        q_norm_w: &[F],
        k_norm_w: &[F],
        gate_w: &[F],
        proj_w: &[F],
        proj_b: &[F],
    ) -> CkResult<Self> {
        if !dim.is_multiple_of(num_heads) {
            return Err(CkError::InvalidConfig(format!(
                "AttentionCore dim={dim} must be divisible by num_heads={num_heads}"
            )));
        }
        let head_dim = dim / num_heads;
        let c3 = 3 * dim;
        if qkv_w.len() != c3 * dim
            || qkv_b.len() != c3
            || q_norm_w.len() != head_dim
            || k_norm_w.len() != head_dim
            || gate_w.len() != num_heads * dim
            || proj_w.len() != dim * dim
            || proj_b.len() != dim
        {
            return Err(CkError::Shape("AttentionCore weight/bias shape mismatch".into()));
        }
        let qkv_wt = transpose_2d(qkv_w, c3, dim);
        let gate_wt = transpose_2d(gate_w, num_heads, dim);
        let proj_wt = transpose_2d(proj_w, dim, dim);
        Ok(Self {
            qkv_w_t: ctx.upload_tensor(&qkv_wt, [dim, c3])?,
            qkv_b: upload(ctx.client(), qkv_b),
            qkv_dw,
            q_norm_w: upload(ctx.client(), q_norm_w),
            k_norm_w: upload(ctx.client(), k_norm_w),
            gate_w_t: ctx.upload_tensor(&gate_wt, [dim, num_heads])?,
            proj_w_t: ctx.upload_tensor(&proj_wt, [dim, dim])?,
            proj_b: upload(ctx.client(), proj_b),
            dim,
            num_heads,
            head_dim,
        })
    }

    pub fn load(
        ctx: &MlContext,
        st: &SafetensorsFile,
        prefix: &str,
        num_heads: usize,
    ) -> CkResult<Self> {
        let (qkv_shape, qkv_w) = st.read_f32(&format!("{prefix}.qkv.qkv.weight"))?;
        let (_, qkv_b) = st.read_f32(&format!("{prefix}.qkv.qkv.bias"))?;
        if qkv_shape.len() != 2 || qkv_shape[0] != 3 * qkv_shape[1] {
            return Err(CkError::Weights(format!(
                "unexpected QKV weight shape {qkv_shape:?} at '{prefix}'"
            )));
        }
        let dim = qkv_shape[1];
        let dw_prefix = format!("{prefix}.qkv_dw");
        let qkv_dw = if st.entry(&format!("{dw_prefix}.dw.weight")).is_ok() {
            Some(QkvDepthwiseSpatial::load(ctx, st, &dw_prefix)?)
        } else {
            None
        };
        let (_, q_norm_w) = st.read_f32(&format!("{prefix}.q_norm.weight"))?;
        let (_, k_norm_w) = st.read_f32(&format!("{prefix}.k_norm.weight"))?;
        let (_, gate_w) = st.read_f32(&format!("{prefix}.attn_gate.proj.weight"))?;
        let (_, proj_w) = st.read_f32(&format!("{prefix}.proj.weight"))?;
        let (_, proj_b) = st.read_f32(&format!("{prefix}.proj.bias"))?;
        Self::from_slices(
            ctx, dim, num_heads, &qkv_w, &qkv_b, qkv_dw, &q_norm_w, &k_norm_w, &gate_w, &proj_w,
            &proj_b,
        )
    }

    /// Computes `qkv = x @ qkv_w_t` and, when `qkv_dw` is present, fuses `qkv_b` directly into the
    /// spatial depthwise filter on `[batches, h, w, 3 * dim]`.
    /// Returns `(qkv_tensor_4d, bias_already_applied)`.
    pub fn project_qkv_spatial<F: CkFloat>(
        &self,
        ctx: &MlContext,
        x_nhwc: Tensor<4>,
    ) -> CkResult<(Tensor<4>, bool)> {
        let [b, h, w, c] = x_nhwc.dims();
        let tokens = b * h * w;
        let c3 = 3 * self.dim;
        let qkv_2d = x_nhwc.reshape([tokens, c]).matmul(self.qkv_w_t.clone());
        let qkv_4d = qkv_2d.reshape([b, h, w, c3]);
        if let Some(dw) = &self.qkv_dw {
            let filtered = dw.forward_with_in_bias::<F, 4>(ctx, qkv_4d, Some(&self.qkv_b))?;
            Ok((filtered, true))
        } else {
            Ok((qkv_4d, false))
        }
    }

    /// Runs fused `(bias?) + split_heads + q_norm + k_norm + apply_rotary` on `qkv_handle`
    /// (`[batches, n_tokens, 3 * dim]`), returning `(q, k, v)` tensors of shape
    /// `[batches, num_heads, n_tokens, head_dim]`.
    pub fn prep_qkv_norm_rope<F: CkFloat>(
        &self,
        ctx: &MlContext,
        qkv_handle: Handle,
        cos: &Handle,
        sin: &Handle,
        batches: usize,
        n_tokens: usize,
        bias_already_applied: bool,
    ) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
        let total_bhn = batches * self.num_heads * n_tokens;
        let total_elems = total_bhn * self.head_dim;
        let c3 = 3 * self.dim;
        let q_h = empty::<F>(ctx.client(), total_elems);
        let k_h = empty::<F>(ctx.client(), total_elems);
        let v_h = empty::<F>(ctx.client(), total_elems);
        let geom = LaunchGeometry::elementwise(ctx.client(), total_bhn);
        unsafe {
            prep_qkv_norm_rope_kernel::launch::<F>(
                ctx.client(),
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(qkv_handle, batches * n_tokens * c3),
                BufferArg::from_raw_parts(self.qkv_b.clone(), c3),
                BufferArg::from_raw_parts(self.q_norm_w.clone(), self.head_dim),
                BufferArg::from_raw_parts(self.k_norm_w.clone(), self.head_dim),
                BufferArg::from_raw_parts(cos.clone(), n_tokens * (self.head_dim / 2)),
                BufferArg::from_raw_parts(sin.clone(), n_tokens * (self.head_dim / 2)),
                BufferArg::from_raw_parts(q_h.clone(), total_elems),
                BufferArg::from_raw_parts(k_h.clone(), total_elems),
                BufferArg::from_raw_parts(v_h.clone(), total_elems),
                total_bhn as u32,
                self.num_heads as u32,
                n_tokens as u32,
                self.head_dim as u32,
                u32::from(!bias_already_applied),
                corridorkey_base::core::cast::<F>(DEFAULT_RMS_NORM_EPS),
            );
        }
        let shape = [batches, self.num_heads, n_tokens, self.head_dim];
        let dt = elem_dtype::<F>();
        (
            ctx.tensor_from_handle(q_h, shape, dt),
            ctx.tensor_from_handle(k_h, shape, dt),
            ctx.tensor_from_handle(v_h, shape, dt),
        )
    }

    /// Applies `HeadwiseAttentionGate` (`sigmoid(gate_tokens @ gate_w_t)`) to `attn`
    /// (`[batches, num_heads, n_tokens, head_dim]`), merges heads to `[batches * n_tokens, dim]`,
    /// and runs output projection `proj_w_t + proj_b`, returning a `Handle` of length
    /// `batches * n_tokens * dim`.
    pub fn gate_and_project_out<F: CkFloat>(
        &self,
        ctx: &MlContext,
        gate_tokens_2d: Tensor<2>,
        attn: Tensor<4>,
        batches: usize,
        n_tokens: usize,
    ) -> CkResult<Handle> {
        let total_tokens = batches * n_tokens;
        let total_elems = total_tokens * self.dim;
        let gate_logits = gate_tokens_2d.matmul(self.gate_w_t.clone());
        let c_attn = into_contiguous_cube(attn)?;
        let c_gate = into_contiguous_cube(gate_logits)?;
        let merged = empty::<F>(ctx.client(), total_elems);
        let geom = LaunchGeometry::elementwise(ctx.client(), total_elems);
        unsafe {
            merge_heads_gate_kernel::launch::<F>(
                ctx.client(),
                geom.cube_count.clone(),
                geom.cube_dim,
                BufferArg::from_raw_parts(c_attn.handle, total_elems),
                BufferArg::from_raw_parts(c_gate.handle, total_tokens * self.num_heads),
                BufferArg::from_raw_parts(merged.clone(), total_elems),
                total_elems as u32,
                self.num_heads as u32,
                n_tokens as u32,
                self.head_dim as u32,
            );
        }
        let merged_2d =
            ctx.tensor_from_handle(merged, [total_tokens, self.dim], elem_dtype::<F>());
        let proj_2d = merged_2d.matmul(self.proj_w_t.clone());
        let c_proj = into_contiguous_cube(proj_2d)?;
        unsafe {
            add_row_bias_kernel::launch::<F>(
                ctx.client(),
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(c_proj.handle.clone(), total_elems),
                BufferArg::from_raw_parts(self.proj_b.clone(), self.dim),
                total_elems as u32,
                self.dim as u32,
            );
        }
        Ok(c_proj.handle)
    }
}

type DeviceRegionTables = Arc<Mutex<Option<Handle>>>;

/// Precomputed shifted-window region groups matching `_shifted_region_groups_2d_cached` and
/// `_shifted_region_groups_3d_cached` in `corridorkey/blocks.py`.
///
/// Precomputes compact per-token region ID tables (`region_ids: [num_windows * n_tokens]`) so
/// [`sdpa_window_kernel`](crate::models::blocks::kernels::sdpa_window_kernel) executes all
/// full, partial, and padded windows in a single kernel launch.
#[derive(Clone, Debug)]
pub struct ShiftedRegionGroups {
    pub num_windows: usize,
    pub full_windows: Vec<u32>,
    pub partial_groups: Vec<(Vec<u32>, Vec<u32>)>,
    pub region_ids: Vec<i32>,
    device_table: DeviceRegionTables,
}

impl ShiftedRegionGroups {
    /// Builds 2D shifted-window region groups for a grid `(h, w)` padded to `(hp, wp)` with
    /// window size `win_size` and cyclic shift `(shift_h, shift_w)`.
    pub fn build_2d(
        h: usize,
        w: usize,
        hp: usize,
        wp: usize,
        win_size: usize,
        shift_h: usize,
        shift_w: usize,
    ) -> Self {
        Self::build_3d(1, h, w, hp, wp, win_size, shift_h, shift_w)
    }

    /// Builds 3D shifted-window region groups (full temporal `t`, spatial window `win_size`).
    #[allow(clippy::too_many_arguments)]
    pub fn build_3d(
        t: usize,
        h: usize,
        w: usize,
        hp: usize,
        wp: usize,
        win_size: usize,
        shift_h: usize,
        shift_w: usize,
    ) -> Self {
        let h_ranges: Vec<(usize, usize)> = if shift_h > 0 {
            vec![(0, hp - win_size), (hp - win_size, hp - shift_h), (hp - shift_h, hp)]
        } else {
            vec![(0, hp)]
        };
        let w_ranges: Vec<(usize, usize)> = if shift_w > 0 {
            vec![(0, wp - win_size), (wp - win_size, wp - shift_w), (wp - shift_w, wp)]
        } else {
            vec![(0, wp)]
        };

        let mut region_grid = vec![-1i16; hp * wp];
        let mut label = 0i16;
        for &(hy0, hy1) in &h_ranges {
            for &(wx0, wx1) in &w_ranges {
                for y in hy0..hy1 {
                    for x in wx0..wx1 {
                        if y < h && x < w {
                            region_grid[y * wp + x] = label;
                        }
                    }
                }
                label += 1;
            }
        }

        let nh = hp / win_size;
        let nw = wp / win_size;
        let num_windows = nh * nw;
        let spatial_tokens = win_size * win_size;
        let n_tokens = t * spatial_tokens;

        let mut full_windows = Vec::new();
        let mut groups_map: BTreeMap<Vec<u32>, Vec<u32>> = BTreeMap::new();
        let mut region_ids = vec![-1i32; num_windows * n_tokens];

        for wy in 0..nh {
            for wx in 0..nw {
                let win_idx = wy * nw + wx;
                let mut win_labels = Vec::with_capacity(spatial_tokens);
                for ly in 0..win_size {
                    for lx in 0..win_size {
                        let py = (wy * win_size + ly + shift_h) % hp;
                        let px = (wx * win_size + lx + shift_w) % wp;
                        win_labels.push(region_grid[py * wp + px]);
                    }
                }

                for ti in 0..t {
                    for si in 0..spatial_tokens {
                        let tok = ti * spatial_tokens + si;
                        region_ids[win_idx * n_tokens + tok] = win_labels[si] as i32;
                    }
                }

                let first = win_labels[0];
                let is_full = first >= 0 && win_labels.iter().all(|&l| l == first);
                if is_full {
                    full_windows.push(win_idx as u32);
                } else {
                    let mut unique_labels: Vec<i16> =
                        win_labels.iter().copied().filter(|&l| l >= 0).collect();
                    unique_labels.sort_unstable();
                    unique_labels.dedup();
                    for lbl in unique_labels {
                        let sp_indices: Vec<usize> = win_labels
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &l)| (l == lbl).then_some(i))
                            .collect();
                        if !sp_indices.is_empty() {
                            let mut tok_tuple = Vec::with_capacity(t * sp_indices.len());
                            for ti in 0..t {
                                for &si in &sp_indices {
                                    tok_tuple.push((ti * spatial_tokens + si) as u32);
                                }
                            }
                            groups_map.entry(tok_tuple).or_default().push(win_idx as u32);
                        }
                    }
                }
            }
        }

        let partial_groups = groups_map.into_iter().collect();
        Self {
            num_windows,
            full_windows,
            partial_groups,
            region_ids,
            device_table: Arc::new(Mutex::new(None)),
        }
    }

    fn get_or_upload_table(&self, ctx: &MlContext) -> Handle {
        let mut guard = self.device_table.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = &*guard {
            return h.clone();
        }
        let h = upload(ctx.client(), &self.region_ids);
        *guard = Some(h.clone());
        h
    }

    /// Executes SDPA with shifted-window region isolation in a single fused CubeCL kernel launch,
    /// matching `_sdpa_with_region_groups` in `corridorkey/blocks.py`.
    pub fn run_sdpa<F: CkFloat>(
        &self,
        ctx: &MlContext,
        q: Tensor<4>,
        k: Tensor<4>,
        v: Tensor<4>,
    ) -> CkResult<Tensor<4>> {
        let [total_wins, num_heads, n_tokens, head_dim] = q.dims();
        if self.partial_groups.is_empty() && self.full_windows.len() == self.num_windows {
            return run_unmasked_sdpa::<F>(ctx, q, k, v);
        }
        let c_q = into_contiguous_cube(q)?;
        let c_k = into_contiguous_cube(k)?;
        let c_v = into_contiguous_cube(v)?;
        let reg_h = self.get_or_upload_table(ctx);
        let out_handle = launch_sdpa_window::<F>(
            ctx.client(),
            c_q.handle,
            c_k.handle,
            c_v.handle,
            Some((&reg_h, self.num_windows)),
            total_wins,
            num_heads,
            n_tokens,
            head_dim,
        )?;
        Ok(ctx.tensor_from_handle(
            out_handle,
            [total_wins, num_heads, n_tokens, head_dim],
            elem_dtype::<F>(),
        ))
    }
}
