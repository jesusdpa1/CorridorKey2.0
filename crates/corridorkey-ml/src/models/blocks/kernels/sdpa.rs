//! Fused shared-memory CubeCL Scaled Dot-Product Attention (`sdpa_window_kernel`) for local
//! windows (`SwinBlock2d`, `SwinBlock3d`), axial strips (`CSwinBlock2d`, `CSwinBlock3d`), and
//! non-tile-aligned sequence lengths where `cubek-attention` would fall back to multi-kernel
//! global-memory materialization.
//!
//! Supports optional per-window region-isolation tables (`q_offsets`, `q_lens`, `k_table`) so
//! shifted and padded Swin windows execute in a **single kernel launch** without gather/scatter
//! copies or CPU subset loops.

use cubecl::{prelude::*, server::Handle};

use corridorkey_base::core::{CkFloat, buffer::empty, cast};
use corridorkey_core::{
    CkError, CkResult,
    compute::{LaunchGeometry, negative_infinity, row_position},
};

/// Maximum per-head dimension staged in shared memory (`64` in Stage 1 `/16`, `80` in Stage 2,
/// `96` in Stage 1 `/64`).
pub const SDPA_MAX_HEAD_DIM: u32 = 96;

/// Number of keys processed per online-softmax shared-memory tile.
pub const SDPA_KEY_TILE: u32 = 128;

/// Target threads per cube for `LaunchGeometry::per_row` in [`launch_sdpa_window`].
pub const SDPA_TARGET_THREADS: usize = 128;

/// Fused shared-memory SDPA kernel with online softmax and optional region-isolation key tables.
///
/// One cube processes one query row `row = row_position()` in `[0, total_wins * num_heads * n_tokens)`:
/// - When `use_region_table != 0`, query `(w_local, q_tok)` attends only to the `q_lens[w_local, q_tok]`
///   keys listed at `k_table[q_offsets[w_local, q_tok] ..]`. Padded out-of-bounds queries have
///   `q_lens == 0` and immediately write `0.0`.
/// - When `use_region_table == 0`, query `q_tok` attends to all `0..n_tokens` keys in its window/strip.
#[cube(launch)]
pub fn sdpa_window_kernel<F: CkFloat>(
    q: &[F],
    k: &[F],
    v: &[F],
    q_offsets: &[u32],
    q_lens: &[u32],
    k_table: &[u32],
    out: &mut [F],
    total_rows: u32,
    num_windows: u32,
    num_heads: u32,
    n_tokens: u32,
    head_dim: u32,
    use_region_table: u32,
    scale: F,
    #[comptime] block_size: u32,
) {
    let mut s_q = Shared::<[F]>::new_slice(SDPA_MAX_HEAD_DIM as usize);
    let mut s_acc = Shared::<[F]>::new_slice(SDPA_MAX_HEAD_DIM as usize);
    let mut s_scores = Shared::<[F]>::new_slice(SDPA_KEY_TILE as usize);
    let mut s_k_idx = Shared::<[u32]>::new_slice(SDPA_KEY_TILE as usize);
    let mut s_meta = Shared::<[F]>::new_slice(4usize);

    let row = row_position();
    let tid = UNIT_POS_X;

    if row < total_rows {
        let q_tok = row % n_tokens;
        let wh = row / n_tokens;
        let win_idx = wh / num_heads;
        let w_local = win_idx % num_windows;

        let bh_base = wh * n_tokens * head_dim;
        let q_base = bh_base + q_tok * head_dim;

        let mut k_offset = 0u32;
        let mut n_valid_keys = n_tokens;
        if use_region_table != 0u32 {
            let meta_idx = (w_local * n_tokens + q_tok) as usize;
            k_offset = q_offsets[meta_idx];
            n_valid_keys = q_lens[meta_idx];
        }

        if n_valid_keys == 0u32 {
            let mut d = tid;
            while d < head_dim {
                out[(q_base + d) as usize] = F::new(0.0f32);
                d += block_size;
            }
        } else {
            let mut d = tid;
            while d < head_dim {
                s_q[d as usize] = q[(q_base + d) as usize];
                s_acc[d as usize] = F::new(0.0f32);
                d += block_size;
            }
            if tid == 0u32 {
                s_meta[0] = negative_infinity::<F>();
                s_meta[1] = F::new(0.0f32);
            }
            sync_cube();

            let mut k_base = 0u32;
            while k_base < n_valid_keys {
                let rem = n_valid_keys - k_base;
                let chunk_len = if rem < SDPA_KEY_TILE {
                    rem
                } else {
                    SDPA_KEY_TILE
                };

                // 1. Parallel Q · K^T dot products across keys in the tile
                let mut k_local = tid;
                while k_local < chunk_len {
                    let k_tok = if use_region_table != 0u32 {
                        k_table[(k_offset + k_base + k_local) as usize]
                    } else {
                        k_base + k_local
                    };
                    s_k_idx[k_local as usize] = k_tok;
                    let k_row_base = bh_base + k_tok * head_dim;
                    let mut dot = F::new(0.0f32);
                    for cd in 0..head_dim {
                        dot += s_q[cd as usize] * k[(k_row_base + cd) as usize];
                    }
                    s_scores[k_local as usize] = dot * scale;
                    k_local += block_size;
                }
                sync_cube();

                // 2. Online softmax max update & previous accumulator rescale factor
                if tid == 0u32 {
                    let m_old = s_meta[0];
                    let mut m_new = m_old;
                    for ki in 0..chunk_len {
                        let sc = s_scores[ki as usize];
                        if sc > m_new {
                            m_new = sc;
                        }
                    }
                    let rescale = if m_old == negative_infinity::<F>() {
                        F::new(0.0f32)
                    } else {
                        F::exp(m_old - m_new)
                    };
                    s_meta[0] = m_new;
                    s_meta[2] = rescale;
                }
                sync_cube();

                // 3. Parallel exp(score - m_new)
                let m_new = s_meta[0];
                let mut k_exp = tid;
                while k_exp < chunk_len {
                    s_scores[k_exp as usize] = F::exp(s_scores[k_exp as usize] - m_new);
                    k_exp += block_size;
                }
                sync_cube();

                // 4. Online denominator update (tid == 0) + parallel V accumulation across head_dim
                if tid == 0u32 {
                    let mut chunk_sum = F::new(0.0f32);
                    for ki in 0..chunk_len {
                        chunk_sum += s_scores[ki as usize];
                    }
                    s_meta[1] = s_meta[1] * s_meta[2] + chunk_sum;
                }

                let rescale = s_meta[2];
                let mut dv = tid;
                while dv < head_dim {
                    let mut chunk_v = F::new(0.0f32);
                    for ki in 0..chunk_len {
                        let k_tok = s_k_idx[ki as usize];
                        let v_idx = (bh_base + k_tok * head_dim + dv) as usize;
                        chunk_v += s_scores[ki as usize] * v[v_idx];
                    }
                    s_acc[dv as usize] = s_acc[dv as usize] * rescale + chunk_v;
                    dv += block_size;
                }
                sync_cube();

                k_base += SDPA_KEY_TILE;
            }

            let inv_l = F::new(1.0f32) / s_meta[1];
            let mut d_out = tid;
            while d_out < head_dim {
                out[(q_base + d_out) as usize] = s_acc[d_out as usize] * inv_l;
                d_out += block_size;
            }
        }
    }
}

/// Launches [`sdpa_window_kernel`] on contiguous `[total_wins, num_heads, n_tokens, head_dim]`
/// `Q`, `K`, `V` handles, returning the output `Handle`.
#[allow(clippy::too_many_arguments)]
pub fn launch_sdpa_window<F: CkFloat>(
    client: &Client,
    q: Handle,
    k: Handle,
    v: Handle,
    region_tables: Option<(&Handle, &Handle, &Handle, usize, usize)>,
    total_wins: usize,
    num_heads: usize,
    n_tokens: usize,
    head_dim: usize,
) -> CkResult<Handle> {
    if head_dim > SDPA_MAX_HEAD_DIM as usize {
        return Err(CkError::InvalidConfig(format!(
            "sdpa_window_kernel head_dim={head_dim} exceeds SDPA_MAX_HEAD_DIM={SDPA_MAX_HEAD_DIM}"
        )));
    }
    let total_rows = total_wins * num_heads * n_tokens;
    let total_elems = total_rows * head_dim;
    let out = empty::<F>(client, total_elems);
    let geom = LaunchGeometry::per_row(client, total_rows, SDPA_TARGET_THREADS);
    let scale = cast::<F>(1.0 / (head_dim as f64).sqrt());

    let (q_off_h, q_len_h, k_tab_h, meta_len, k_tab_len, num_windows, use_region) =
        if let Some((qo, ql, kt, nw, kt_len)) = region_tables {
            (qo.clone(), ql.clone(), kt.clone(), nw * n_tokens, kt_len, nw, 1u32)
        } else {
            // Bind 1-element dummy slices from `out` when `use_region_table == 0` (never read).
            (out.clone(), out.clone(), out.clone(), 1, 1, total_wins.max(1), 0u32)
        };

    unsafe {
        sdpa_window_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(q, total_elems),
            BufferArg::from_raw_parts(k, total_elems),
            BufferArg::from_raw_parts(v, total_elems),
            BufferArg::from_raw_parts(q_off_h, meta_len),
            BufferArg::from_raw_parts(q_len_h, meta_len),
            BufferArg::from_raw_parts(k_tab_h, k_tab_len.max(1)),
            BufferArg::from_raw_parts(out.clone(), total_elems),
            total_rows as u32,
            num_windows as u32,
            num_heads as u32,
            n_tokens as u32,
            head_dim as u32,
            use_region,
            scale,
            geom.cube_dim.x,
        );
    }

    Ok(out)
}
