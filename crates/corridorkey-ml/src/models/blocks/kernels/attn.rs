//! CubeCL kernels for 2D/3D attention blocks: fused QKV head-split + QK-RMSNorm + RoPE,
//! headwise sigmoid gating + head merge, CSwin strip packing/unpacking, and Swin window
//! partition/unpartition with cyclic shift.

use cubecl::prelude::*;

use corridorkey_base::core::CkFloat;

/// `sigmoid(x) = 1 / (1 + exp(-x))`.
#[cube]
pub fn sigmoid<F: Float>(x: F) -> F {
    F::new(1.0f32) / (F::new(1.0f32) + F::exp(-x))
}

/// Fuses QKV bias addition (when `apply_bias != 0`), head unpacking (`[B, N, 3, H, D]` ->
/// `[B, H, N, D]`), per-head `q_norm` / `k_norm` (`RMSNorm(head_dim)`), and `apply_rotary`
/// (`cos`, `sin` of shape `[N, D/2]`) into a single kernel launch.
///
/// One thread processes one `(batch, head, token)` vector of length `head_dim` (`64`, `80`, or `96`).
#[cube(launch)]
pub fn prep_qkv_norm_rope_kernel<F: CkFloat>(
    qkv: &[F],
    qkv_bias: &[F],
    q_norm_w: &[F],
    k_norm_w: &[F],
    cos: &[F],
    sin: &[F],
    q_out: &mut [F],
    k_out: &mut [F],
    v_out: &mut [F],
    total_bhn: u32,
    num_heads: u32,
    n_tokens: u32,
    head_dim: u32,
    apply_bias: u32,
    eps: F,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < total_bhn {
        let n = idx % n_tokens;
        let bh = idx / n_tokens;
        let h = bh % num_heads;
        let b = bh / num_heads;

        let dim = num_heads * head_dim;
        let c3 = 3u32 * dim;
        let tok_base = (b * n_tokens + n) * c3;
        let q_off = h * head_dim;
        let k_off = dim + h * head_dim;
        let v_off = 2u32 * dim + h * head_dim;
        let out_base = ((b * num_heads + h) * n_tokens + n) * head_dim;

        // 1. Compute mean square for Q and K, and copy V to v_out
        let mut sum_q2 = F::new(0.0f32);
        let mut sum_k2 = F::new(0.0f32);
        for d in 0..head_dim {
            let mut qv = qkv[(tok_base + q_off + d) as usize];
            let mut kv = qkv[(tok_base + k_off + d) as usize];
            let mut vv = qkv[(tok_base + v_off + d) as usize];
            if apply_bias != 0u32 {
                qv += qkv_bias[(q_off + d) as usize];
                kv += qkv_bias[(k_off + d) as usize];
                vv += qkv_bias[(v_off + d) as usize];
            }
            sum_q2 += qv * qv;
            sum_k2 += kv * kv;
            v_out[(out_base + d) as usize] = vv;
        }

        let inv_d = F::new(1.0f32) / F::cast_from(head_dim);
        let scale_q = F::new(1.0f32) / F::sqrt(sum_q2 * inv_d + eps);
        let scale_k = F::new(1.0f32) / F::sqrt(sum_k2 * inv_d + eps);

        // 2. Normalize and apply RoPE on each (2*p, 2*p+1) pair
        let half_d = head_dim / 2u32;
        let rope_base = n * half_d;
        for p in 0..half_d {
            let d0 = 2u32 * p;
            let d1 = d0 + 1u32;

            let mut q0 = qkv[(tok_base + q_off + d0) as usize];
            let mut q1 = qkv[(tok_base + q_off + d1) as usize];
            let mut k0 = qkv[(tok_base + k_off + d0) as usize];
            let mut k1 = qkv[(tok_base + k_off + d1) as usize];
            if apply_bias != 0u32 {
                q0 += qkv_bias[(q_off + d0) as usize];
                q1 += qkv_bias[(q_off + d1) as usize];
                k0 += qkv_bias[(k_off + d0) as usize];
                k1 += qkv_bias[(k_off + d1) as usize];
            }

            q0 = q0 * scale_q * q_norm_w[d0 as usize];
            q1 = q1 * scale_q * q_norm_w[d1 as usize];
            k0 = k0 * scale_k * k_norm_w[d0 as usize];
            k1 = k1 * scale_k * k_norm_w[d1 as usize];

            let c = cos[(rope_base + p) as usize];
            let s = sin[(rope_base + p) as usize];

            q_out[(out_base + d0) as usize] = q0 * c - q1 * s;
            q_out[(out_base + d1) as usize] = q1 * c + q0 * s;
            k_out[(out_base + d0) as usize] = k0 * c - k1 * s;
            k_out[(out_base + d1) as usize] = k1 * c + k0 * s;
        }
    }
}

/// Fuses headwise sigmoid gating (`HeadwiseAttentionGate`) with head transpose + flatten:
/// reads `attn` (`[B, num_heads, N, head_dim]`) and `gate_logits` (`[B, N, num_heads]`),
/// and writes `out` (`[B, N, num_heads * head_dim]`).
#[cube(launch)]
pub fn merge_heads_gate_kernel<F: CkFloat>(
    attn: &[F],
    gate_logits: &[F],
    output: &mut [F],
    total_elems: u32,
    num_heads: u32,
    n_tokens: u32,
    head_dim: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < total_elems {
        let d = idx % head_dim;
        let bnh = idx / head_dim;
        let h = bnh % num_heads;
        let bn = bnh / num_heads;
        let n = bn % n_tokens;
        let b = bn / n_tokens;

        let attn_idx = (((b * num_heads + h) * n_tokens + n) * head_dim + d) as usize;
        let gate_idx = (bn * num_heads + h) as usize;
        let g = sigmoid::<F>(gate_logits[gate_idx]);
        output[idx as usize] = attn[attn_idx] * g;
    }
}

/// Fuses `CSwinAttention2D` QKV preparation: splits heads into H-stripes (`0..h_heads`,
/// `[B * W, h_heads, H, D]`) and W-stripes (`h_heads..num_heads`, `[B * H, w_heads, W, D]`),
/// applies `q_norm` and `k_norm`, and applies axial 2D RoPE (`cos_h`/`sin_h` on H-stripes,
/// `cos_w`/`sin_w` on W-stripes).
#[cube(launch)]
pub fn prep_cswin2d_qkv_kernel<F: CkFloat>(
    qkv: &[F],
    qkv_bias: &[F],
    q_norm_w: &[F],
    k_norm_w: &[F],
    cos_h: &[F],
    sin_h: &[F],
    cos_w: &[F],
    sin_w: &[F],
    qh_out: &mut [F],
    kh_out: &mut [F],
    vh_out: &mut [F],
    qw_out: &mut [F],
    kw_out: &mut [F],
    vw_out: &mut [F],
    total_bhn: u32,
    h_spatial: u32,
    w_spatial: u32,
    num_heads: u32,
    head_dim: u32,
    apply_bias: u32,
    eps: F,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < total_bhn {
        let n_tokens = h_spatial * w_spatial;
        let tok = idx % n_tokens;
        let bh = idx / n_tokens;
        let h = bh % num_heads;
        let b = bh / num_heads;
        let y = tok / w_spatial;
        let x = tok % w_spatial;

        let dim = num_heads * head_dim;
        let c3 = 3u32 * dim;
        let tok_base = (b * n_tokens + tok) * c3;
        let q_off = h * head_dim;
        let k_off = dim + h * head_dim;
        let v_off = 2u32 * dim + h * head_dim;

        let mut sum_q2 = F::new(0.0f32);
        let mut sum_k2 = F::new(0.0f32);
        for d in 0..head_dim {
            let mut qv = qkv[(tok_base + q_off + d) as usize];
            let mut kv = qkv[(tok_base + k_off + d) as usize];
            if apply_bias != 0u32 {
                qv += qkv_bias[(q_off + d) as usize];
                kv += qkv_bias[(k_off + d) as usize];
            }
            sum_q2 += qv * qv;
            sum_k2 += kv * kv;
        }

        let inv_d = F::new(1.0f32) / F::cast_from(head_dim);
        let scale_q = F::new(1.0f32) / F::sqrt(sum_q2 * inv_d + eps);
        let scale_k = F::new(1.0f32) / F::sqrt(sum_k2 * inv_d + eps);

        let h_heads = num_heads / 2u32;
        let half_d = head_dim / 2u32;

        if h < h_heads {
            // H-group: shape [B * W, h_heads, H, head_dim]
            let out_base = (((b * w_spatial + x) * h_heads + h) * h_spatial + y) * head_dim;
            let rope_base = y * half_d;
            for d in 0..head_dim {
                let mut vv = qkv[(tok_base + v_off + d) as usize];
                if apply_bias != 0u32 {
                    vv += qkv_bias[(v_off + d) as usize];
                }
                vh_out[(out_base + d) as usize] = vv;
            }
            for p in 0..half_d {
                let d0 = 2u32 * p;
                let d1 = d0 + 1u32;
                let mut q0 = qkv[(tok_base + q_off + d0) as usize];
                let mut q1 = qkv[(tok_base + q_off + d1) as usize];
                let mut k0 = qkv[(tok_base + k_off + d0) as usize];
                let mut k1 = qkv[(tok_base + k_off + d1) as usize];
                if apply_bias != 0u32 {
                    q0 += qkv_bias[(q_off + d0) as usize];
                    q1 += qkv_bias[(q_off + d1) as usize];
                    k0 += qkv_bias[(k_off + d0) as usize];
                    k1 += qkv_bias[(k_off + d1) as usize];
                }
                q0 = q0 * scale_q * q_norm_w[d0 as usize];
                q1 = q1 * scale_q * q_norm_w[d1 as usize];
                k0 = k0 * scale_k * k_norm_w[d0 as usize];
                k1 = k1 * scale_k * k_norm_w[d1 as usize];
                let c = cos_h[(rope_base + p) as usize];
                let s = sin_h[(rope_base + p) as usize];
                qh_out[(out_base + d0) as usize] = q0 * c - q1 * s;
                qh_out[(out_base + d1) as usize] = q1 * c + q0 * s;
                kh_out[(out_base + d0) as usize] = k0 * c - k1 * s;
                kh_out[(out_base + d1) as usize] = k1 * c + k0 * s;
            }
        } else {
            // W-group: shape [B * H, w_heads, W, head_dim]
            let w_heads = num_heads - h_heads;
            let hw = h - h_heads;
            let out_base = (((b * h_spatial + y) * w_heads + hw) * w_spatial + x) * head_dim;
            let rope_base = x * half_d;
            for d in 0..head_dim {
                let mut vv = qkv[(tok_base + v_off + d) as usize];
                if apply_bias != 0u32 {
                    vv += qkv_bias[(v_off + d) as usize];
                }
                vw_out[(out_base + d) as usize] = vv;
            }
            for p in 0..half_d {
                let d0 = 2u32 * p;
                let d1 = d0 + 1u32;
                let mut q0 = qkv[(tok_base + q_off + d0) as usize];
                let mut q1 = qkv[(tok_base + q_off + d1) as usize];
                let mut k0 = qkv[(tok_base + k_off + d0) as usize];
                let mut k1 = qkv[(tok_base + k_off + d1) as usize];
                if apply_bias != 0u32 {
                    q0 += qkv_bias[(q_off + d0) as usize];
                    q1 += qkv_bias[(q_off + d1) as usize];
                    k0 += qkv_bias[(k_off + d0) as usize];
                    k1 += qkv_bias[(k_off + d1) as usize];
                }
                q0 = q0 * scale_q * q_norm_w[d0 as usize];
                q1 = q1 * scale_q * q_norm_w[d1 as usize];
                k0 = k0 * scale_k * k_norm_w[d0 as usize];
                k1 = k1 * scale_k * k_norm_w[d1 as usize];
                let c = cos_w[(rope_base + p) as usize];
                let s = sin_w[(rope_base + p) as usize];
                qw_out[(out_base + d0) as usize] = q0 * c - q1 * s;
                qw_out[(out_base + d1) as usize] = q1 * c + q0 * s;
                kw_out[(out_base + d0) as usize] = k0 * c - k1 * s;
                kw_out[(out_base + d1) as usize] = k1 * c + k0 * s;
            }
        }
    }
}

/// Fuses `CSwinAttention2D` strip un-permutation (`attn_h: [B*W, h_heads, H, D]` and
/// `attn_w: [B*H, w_heads, W, D]`) and `HeadwiseAttentionGate` (`gate_logits: [B, H, W, num_heads]`)
/// into `[B, H, W, num_heads * D]`.
#[cube(launch)]
pub fn merge_cswin2d_gate_kernel<F: CkFloat>(
    attn_h: &[F],
    attn_w: &[F],
    gate_logits: &[F],
    output: &mut [F],
    total_elems: u32,
    h_spatial: u32,
    w_spatial: u32,
    num_heads: u32,
    head_dim: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < total_elems {
        let d = idx % head_dim;
        let bnh = idx / head_dim;
        let h = bnh % num_heads;
        let btok = bnh / num_heads;
        let n_tokens = h_spatial * w_spatial;
        let tok = btok % n_tokens;
        let b = btok / n_tokens;
        let y = tok / w_spatial;
        let x = tok % w_spatial;

        let h_heads = num_heads / 2u32;
        let val = if h < h_heads {
            let src = (((b * w_spatial + x) * h_heads + h) * h_spatial + y) * head_dim + d;
            attn_h[src as usize]
        } else {
            let w_heads = num_heads - h_heads;
            let hw = h - h_heads;
            let src = (((b * h_spatial + y) * w_heads + hw) * w_spatial + x) * head_dim + d;
            attn_w[src as usize]
        };

        let gate_idx = (btok * num_heads + h) as usize;
        output[idx as usize] = val * sigmoid::<F>(gate_logits[gate_idx]);
    }
}

/// Pads `[B, H, W, C]` with zeros to `[B, Hp, Wp, C]` and rolls by `(-shift_h, -shift_w)` in one pass.
#[cube(launch)]
pub fn pad_roll_2d_kernel<F: CkFloat>(
    input: &[F],
    output: &mut [F],
    total_elems: u32,
    h: u32,
    w: u32,
    hp: u32,
    wp: u32,
    c: u32,
    shift_h: u32,
    shift_w: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < total_elems {
        let ch = idx % c;
        let pix = idx / c;
        let px = pix % wp;
        let py = (pix / wp) % hp;
        let b = pix / (wp * hp);

        let src_y = (py + shift_h) % hp;
        let src_x = (px + shift_w) % wp;
        if src_y < h && src_x < w {
            let src_idx = (((b * h + src_y) * w + src_x) * c + ch) as usize;
            output[idx as usize] = input[src_idx];
        } else {
            output[idx as usize] = F::new(0.0f32);
        }
    }
}

/// Partitions `[B, Hp, Wp, C]` into `[B * (Hp/w) * (Wp/w), w * w, C]`.
#[cube(launch)]
pub fn window_partition_2d_kernel<F: CkFloat>(
    input: &[F],
    output: &mut [F],
    total_elems: u32,
    hp: u32,
    wp: u32,
    win_size: u32,
    c: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < total_elems {
        let ch = idx % c;
        let rem = idx / c;
        let lx = rem % win_size;
        let rem2 = rem / win_size;
        let ly = rem2 % win_size;
        let win_idx = rem2 / win_size;

        let nw = wp / win_size;
        let nh = hp / win_size;
        let wx = win_idx % nw;
        let wy = (win_idx / nw) % nh;
        let b = win_idx / (nw * nh);

        let py = wy * win_size + ly;
        let px = wx * win_size + lx;
        let src_idx = (((b * hp + py) * wp + px) * c + ch) as usize;
        output[idx as usize] = input[src_idx];
    }
}

/// Unpartitions `[B * (Hp/w) * (Wp/w), w * w, C]`, rolls by `(+shift_h, +shift_w)`, crops to
/// `[B, H, W, C]`, and adds `shortcut` (`[B, H, W, C]`) in a single pass!
#[cube(launch)]
pub fn window_unpartition_unroll_residual_2d_kernel<F: CkFloat>(
    windows: &[F],
    shortcut: &[F],
    output: &mut [F],
    total_elems: u32,
    h: u32,
    w: u32,
    hp: u32,
    wp: u32,
    win_size: u32,
    c: u32,
    shift_h: u32,
    shift_w: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < total_elems {
        let ch = idx % c;
        let pix = idx / c;
        let x = pix % w;
        let y = (pix / w) % h;
        let b = pix / (w * h);

        let py = (y + hp - shift_h) % hp;
        let px = (x + wp - shift_w) % wp;
        let wy = py / win_size;
        let ly = py % win_size;
        let wx = px / win_size;
        let lx = px % win_size;

        let nw = wp / win_size;
        let nh = hp / win_size;
        let win_idx = (b * nh + wy) * nw + wx;
        let tok_idx = ly * win_size + lx;
        let src_idx = ((win_idx * (win_size * win_size) + tok_idx) * c + ch) as usize;
        output[idx as usize] = shortcut[idx as usize] + windows[src_idx];
    }
}

/// Fills `output` with zeros (`0.0`).
#[cube(launch)]
pub fn zero_buffer_kernel<F: CkFloat>(output: &mut [F], total_elems: u32) {
    let idx = ABSOLUTE_POS as u32;
    if idx < total_elems {
        output[idx as usize] = F::new(0.0f32);
    }
}

/// Gathers `src[win_indices[w], h, token_indices[t], d]` from `src` (`[total_wins, num_heads, n_tokens, head_dim]`)
/// into `dst` (`[n_sel_wins, num_heads, n_sel_tokens, head_dim]`).
#[cube(launch)]
pub fn gather_window_tokens_kernel<F: CkFloat>(
    src: &[F],
    win_indices: &[u32],
    token_indices: &[u32],
    dst: &mut [F],
    total_elems: u32,
    num_heads: u32,
    n_tokens_full: u32,
    n_sel_tokens: u32,
    head_dim: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < total_elems {
        let d = idx % head_dim;
        let rem = idx / head_dim;
        let t_sel = rem % n_sel_tokens;
        let rem2 = rem / n_sel_tokens;
        let h = rem2 % num_heads;
        let w_sel = rem2 / num_heads;

        let win_full = win_indices[w_sel as usize];
        let tok_full = token_indices[t_sel as usize];
        let src_idx = (((win_full * num_heads + h) * n_tokens_full + tok_full) * head_dim + d) as usize;
        dst[idx as usize] = src[src_idx];
    }
}

/// Scatters `src` (`[n_sel_wins, num_heads, n_sel_tokens, head_dim]`) back into
/// `dst[win_indices[w], h, token_indices[t], d]` (`[total_wins, num_heads, n_tokens, head_dim]`).
#[cube(launch)]
pub fn scatter_window_tokens_kernel<F: CkFloat>(
    src: &[F],
    win_indices: &[u32],
    token_indices: &[u32],
    dst: &mut [F],
    total_elems: u32,
    num_heads: u32,
    n_tokens_full: u32,
    n_sel_tokens: u32,
    head_dim: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < total_elems {
        let d = idx % head_dim;
        let rem = idx / head_dim;
        let t_sel = rem % n_sel_tokens;
        let rem2 = rem / n_sel_tokens;
        let h = rem2 % num_heads;
        let w_sel = rem2 / num_heads;

        let win_full = win_indices[w_sel as usize];
        let tok_full = token_indices[t_sel as usize];
        let dst_idx = (((win_full * num_heads + h) * n_tokens_full + tok_full) * head_dim + d) as usize;
        dst[dst_idx] = src[idx as usize];
    }
}
