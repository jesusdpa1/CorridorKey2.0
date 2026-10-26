//! From decoded frames to the network inputs, on the device (`video_io.py`,
//! `runtime._build_foundation_inputs`).
//!
//! Frames go up once, as the decoder's 8-bit RGB (half the bytes of `f16`, a quarter of `f32`;
//! dsp-kitchen's "cross in the most compact form"). One kernel then writes the model input and
//! the backbones' view at the padded size; the backbone inputs are resized from that view. Nothing
//! comes back to the host.

pub mod kernels;

use corridorkey_core::compute::LaunchGeometry;
use corridorkey_core::frame::{FrameMeta, LinearAdjust};
use corridorkey_core::{CkError, CkResult};
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::color::asinh_params;
use crate::core::buffer::{empty, upload};
use crate::core::{CkFloat, cast};
use kernels::{area_resize_kernel, bilinear_resize_kernel, frame_views_kernel};

/// Packs bytes four per little-endian `u32` word (zero-padded), the form kernels read 8-bit data in.
pub fn pack_bytes(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks(4)
        .map(|c| {
            let mut w = [0u8; 4];
            w[..c.len()].copy_from_slice(c);
            u32::from_le_bytes(w)
        })
        .collect()
}

/// Uploads `[T, h0, w0, 3]` 8-bit RGB frames as packed words.
pub fn upload_frames(client: &Client, bytes: &[u8]) -> Handle {
    upload(client, &pack_bytes(bytes))
}

/// Device views of `t` frames at the padded model size `[t, H, W, 3]`.
#[derive(Debug, Clone)]
pub struct FrameViews {
    /// Stage 1 / Stage 2 input: `asinh(adjust(srgb_to_linear(display)))`.
    pub model: Handle,
    /// What the backbones see: the display frame, or `linear_to_vfm_sdr(adjusted)` when a
    /// linear adjustment is set.
    pub foundation: Handle,
    pub t: usize,
    pub hw: (usize, usize),
}

/// [`FrameViews`] of `t` packed frames of `meta`'s native size (`model_scale` 1: the product's
/// render path supports only that), reflect-101 padded to `meta.shape_hw`.
pub fn frame_views<F: CkFloat>(client: &Client, frames: &Handle, t: usize, meta: &FrameMeta, adjust: LinearAdjust) -> CkResult<FrameViews> {
    if meta.model_native_hw != meta.native_hw {
        return Err(CkError::InvalidConfig(format!(
            "model_scale {} is not supported: frames must be prepared at their native size",
            meta.model_scale
        )));
    }
    let (h0, w0) = meta.native_hw;
    let (h, w) = meta.shape_hw;
    let p = meta.pad;
    if p.top.max(p.bottom) >= h0 || p.left.max(p.right) >= w0 {
        return Err(CkError::InvalidConfig(format!("padding {p:?} exceeds the {h0}x{w0} frame (reflect-101)")));
    }
    let total = t * h * w * 3;
    let (model, foundation) = (empty::<F>(client, total), empty::<F>(client, total));
    let (scale, offset) = adjust.affine();
    let (knee, denom) = asinh_params();
    let geom = LaunchGeometry::elementwise(client, total);
    unsafe {
        frame_views_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            1,
            BufferArg::from_raw_parts(frames.clone(), (t * h0 * w0 * 3).div_ceil(4)),
            BufferArg::from_raw_parts(model.clone(), total),
            BufferArg::from_raw_parts(foundation.clone(), total),
            total as u32,
            h as u32,
            w as u32,
            h0 as u32,
            w0 as u32,
            p.top as u32,
            p.left as u32,
            cast::<F>(scale),
            cast::<F>(offset),
            cast::<F>(1.0 / knee),
            cast::<F>(1.0 / denom),
            !adjust.is_identity(),
        );
    }
    Ok(FrameViews { model, foundation, t, hw: (h, w) })
}

/// `interpolate(mode="area")` of `[t, h, w, ch]` to `[t, oh, ow, ch]`.
pub fn area_resize<F: CkFloat>(client: &Client, input: &Handle, (t, h, w, ch): (usize, usize, usize, usize), (oh, ow): (usize, usize)) -> Handle {
    let total = t * oh * ow * ch;
    let out = empty::<F>(client, total);
    let geom = LaunchGeometry::elementwise(client, total);
    unsafe {
        area_resize_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(input.clone(), t * h * w * ch),
            BufferArg::from_raw_parts(out.clone(), total),
            total as u32,
            h as u32,
            w as u32,
            ch as u32,
            oh as u32,
            ow as u32,
        );
    }
    out
}

/// `interpolate(mode="bilinear", align_corners=False)` of `[t, h, w, ch]` to `[t, oh, ow, ch]`,
/// clamped to [0, 1] when `clamp`.
pub fn bilinear_resize<F: CkFloat>(client: &Client, input: &Handle, (t, h, w, ch): (usize, usize, usize, usize), (oh, ow): (usize, usize), clamp: bool) -> Handle {
    let total = t * oh * ow * ch;
    let out = empty::<F>(client, total);
    let geom = LaunchGeometry::elementwise(client, total);
    unsafe {
        bilinear_resize_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(input.clone(), t * h * w * ch),
            BufferArg::from_raw_parts(out.clone(), total),
            total as u32,
            h as u32,
            w as u32,
            ch as u32,
            oh as u32,
            ow as u32,
            clamp,
        );
    }
    out
}

/// The three backbones' inputs from the foundation view, at the product's sizes.
#[derive(Debug, Clone)]
pub struct BackboneInputs {
    /// `[t, H/4, W/4, 3]`, area resize.
    pub cradio: Handle,
    /// `[t, 7H/32, 7W/32, 3]`, area resize.
    pub moge: Handle,
    /// `[t, H/4, W/4, 3]`, bilinear resize clamped to [0, 1] (`precompute_rvm_to_memmap`).
    pub rvm: Handle,
    pub quarter_hw: (usize, usize),
    pub moge_hw: (usize, usize),
}

impl BackboneInputs {
    pub fn from_views<F: CkFloat>(client: &Client, views: &FrameViews) -> Self {
        let (t, (h, w)) = (views.t, views.hw);
        let dims = (t, h, w, 3);
        let quarter_hw = (h / 4, w / 4);
        let moge_hw = (h * 7 / 32, w * 7 / 32);
        Self {
            cradio: area_resize::<F>(client, &views.foundation, dims, quarter_hw),
            moge: area_resize::<F>(client, &views.foundation, dims, moge_hw),
            rvm: bilinear_resize::<F>(client, &views.foundation, dims, quarter_hw, true),
            quarter_hw,
            moge_hw,
        }
    }
}
