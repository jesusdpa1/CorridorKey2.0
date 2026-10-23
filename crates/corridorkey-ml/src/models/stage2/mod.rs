//! Stage 2 (`/16` pure-transformer refinement + `16×` depatchify to `/1`, `corridorkey/stage2.py`).
//!
//! Operates end-to-end on channels-last `[BT, H, W, C]` tensors on the device:
//! 1. `patch_1`: `[BT, H, W, 3]` -> `[BT, H/16, W/16, d_rgb]` (`768`).
//! 2. Concatenates `[baton (512) | rgb_feat (768)]` -> `[BT, H/16, W/16, 1280]`.
//! 3. `8 × pre_blocks` (`FullSpatial`, `Swin`, `CSwin`).
//! 4. `GatedRgbSkip` (`patch_2` + `patch_2_gate`).
//! 5. `2 × final_blocks` (`Swin(hw, 0)`, `Swin(0, hw)`).
//! 6. `depatchify`: `[BT, H/16, W/16, 1280]` -> `[BT, H, W, 4]` (`[alpha_logit, fg_r, fg_g, fg_b]`).

use burn_tensor::Tensor;
use corridorkey_base::core::CkFloat;
use corridorkey_core::{CkError, CkResult};

use crate::{
    hub::{ModelConfig, SafetensorsFile},
    models::{
        blocks::{
            CSwinBlock2d, Depatchify, FullSpatialBlock2d, GatedRgbSkip, PatchifyConv, Stage2Block2d,
            SwinBlock2d,
        },
        rope::Rope2d,
    },
    Provenance,
    provenance::corridorkey_v2_provenance,
    runtime::MlContext,
};

/// Patch stride for Stage 2 (`/16`).
pub const STAGE2_PATCH_SIZE: usize = 16;

/// Output channels of Stage 2 (`1` alpha logit + `3` FG RGB channels).
pub const STAGE2_OUT_CHANNELS: usize = 4;

/// Device-resident Stage 2 refinement network (`corridorkey/stage2.py`).
#[derive(Clone, Debug)]
pub struct Stage2 {
    pub patch_1: PatchifyConv,
    pub pre_blocks: Vec<Stage2Block2d>,
    pub patch_2_skip: Option<GatedRgbSkip>,
    pub final_blocks: Vec<Stage2Block2d>,
    pub depatchify: Depatchify,
    pub d: usize,
    pub d_baton: usize,
    pub window_size: usize,
}

impl Stage2 {
    pub fn provenance() -> Provenance {
        corridorkey_v2_provenance()
    }

    /// Loads `Stage2` weights from `SafetensorsFile` under `prefix` (typically `"stage2"`).
    pub fn load(
        ctx: &MlContext,
        st: &SafetensorsFile,
        cfg: &ModelConfig,
        prefix: &str,
    ) -> CkResult<Self> {
        let d = cfg.d_stage2;
        let d_baton = cfg.d_16;
        let num_heads = cfg.num_heads_stage2;
        let win = cfg.window_size_stage2;
        let hw = win / 2;
        let head_dim = d / num_heads;
        let rope = Rope2d::new(head_dim, cfg.rope_base)?;

        let patch_1 = PatchifyConv::load(ctx, st, &format!("{prefix}.patch_1"), STAGE2_PATCH_SIZE)?;

        let num_pre = cfg.n_blocks_stage2.saturating_sub(cfg.n_final_blocks_stage2);
        let pre_specs: &[(&str, (usize, usize))] = match num_pre {
            8 => &[
                ("full", (0, 0)),
                ("swin", (hw, hw)),
                ("cswin", (0, 0)),
                ("swin", (hw, 0)),
                ("full", (0, 0)),
                ("cswin", (0, 0)),
                ("swin", (0, 0)),
                ("swin", (hw, hw)),
            ],
            6 => &[
                ("swin", (0, 0)),
                ("swin", (hw, hw)),
                ("cswin", (0, 0)),
                ("cswin", (0, 0)),
                ("swin", (hw, 0)),
                ("swin", (0, hw)),
            ],
            other => {
                return Err(CkError::InvalidConfig(format!(
                    "Stage2 num_pre_blocks must be 6 or 8, got {other}"
                )));
            }
        };

        let mut pre_blocks = Vec::with_capacity(pre_specs.len());
        for (i, &(kind, shift)) in pre_specs.iter().enumerate() {
            let blk_prefix = format!("{prefix}.pre_blocks.{i}");
            let blk = match kind {
                "full" => Stage2Block2d::Full(FullSpatialBlock2d::load(
                    ctx,
                    st,
                    &blk_prefix,
                    num_heads,
                    rope.clone(),
                )?),
                "swin" => Stage2Block2d::Swin(SwinBlock2d::load(
                    ctx,
                    st,
                    &blk_prefix,
                    num_heads,
                    win,
                    shift,
                    rope.clone(),
                )?),
                "cswin" => Stage2Block2d::CSwin(CSwinBlock2d::load(
                    ctx,
                    st,
                    &blk_prefix,
                    num_heads,
                    rope.clone(),
                )?),
                _ => unreachable!(),
            };
            pre_blocks.push(blk);
        }

        let patch_2_skip = if st.entry(&format!("{prefix}.patch_2.conv.weight")).is_ok() {
            Some(GatedRgbSkip::load(
                ctx,
                st,
                prefix,
                d,
                d_baton,
                STAGE2_PATCH_SIZE,
            )?)
        } else {
            None
        };

        let final_shifts = [(hw, 0), (0, hw)];
        let mut final_blocks = Vec::with_capacity(final_shifts.len());
        for (i, &shift) in final_shifts.iter().enumerate() {
            let blk_prefix = format!("{prefix}.final_blocks.{i}");
            final_blocks.push(Stage2Block2d::Swin(SwinBlock2d::load(
                ctx,
                st,
                &blk_prefix,
                num_heads,
                win,
                shift,
                rope.clone(),
            )?));
        }

        let depatchify = Depatchify::load(
            ctx,
            st,
            &format!("{prefix}.depatchify"),
            STAGE2_PATCH_SIZE,
        )?;

        Ok(Self {
            patch_1,
            pre_blocks,
            patch_2_skip,
            final_blocks,
            depatchify,
            d,
            d_baton,
            window_size: win,
        })
    }

    /// Runs Stage 2 forward pass on channels-last `rgb_nhwc` (`[BT, H, W, 3]`) and
    /// `baton_nhwc` (`[BT, H / 16, W / 16, d_baton]`), returning `[BT, H, W, 4]`.
    pub fn forward<F: CkFloat>(
        &self,
        ctx: &MlContext,
        rgb_nhwc: Tensor<4>,
        baton_nhwc: Tensor<4>,
    ) -> CkResult<Tensor<4>> {
        let [bt, h, w, c_rgb] = rgb_nhwc.dims();
        if c_rgb != 3 || !h.is_multiple_of(STAGE2_PATCH_SIZE) || !w.is_multiple_of(STAGE2_PATCH_SIZE)
        {
            return Err(CkError::Shape(format!(
                "Stage2::forward expects rgb_nhwc [BT, H, W, 3] with /16-aligned H, W; got [{bt}, {h}, {w}, {c_rgb}]"
            )));
        }
        let h16 = h / STAGE2_PATCH_SIZE;
        let w16 = w / STAGE2_PATCH_SIZE;
        if baton_nhwc.dims() != [bt, h16, w16, self.d_baton] {
            return Err(CkError::Shape(format!(
                "Stage2::forward expected baton shape [{bt}, {h16}, {w16}, {}], got {:?}",
                self.d_baton,
                baton_nhwc.dims()
            )));
        }

        let rgb_feat = self.patch_1.forward::<F>(ctx, rgb_nhwc.clone())?;
        let mut x = Tensor::cat(vec![baton_nhwc, rgb_feat], 3);

        for blk in &self.pre_blocks {
            x = blk.forward::<F>(ctx, x)?;
        }

        if let Some(skip) = &self.patch_2_skip {
            x = skip.forward::<F>(ctx, x, rgb_nhwc)?;
        }

        for blk in &self.final_blocks {
            x = blk.forward::<F>(ctx, x)?;
        }

        self.depatchify.forward::<F>(ctx, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use corridorkey_core::ComputeTarget;

    #[test]
    fn loads_and_runs_full_stage2_from_v2_checkpoint() {
        let ckpt_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../weights/corridorkey_v2.safetensors");
        if !ckpt_path.exists() {
            return;
        }
        let st = SafetensorsFile::open(&ckpt_path).unwrap();
        let header = st.checkpoint_header().unwrap();

        for target in ComputeTarget::available() {
            let ctx = MlContext::new(target).unwrap();
            let stage2 = Stage2::load(&ctx, &st, &header.model_config, "stage2").unwrap();

            let (bt, h, w) = (1, 128, 128);
            let h16 = h / STAGE2_PATCH_SIZE;
            let w16 = w / STAGE2_PATCH_SIZE;
            let rgb: Vec<f32> = (0..bt * h * w * 3)
                .map(|i| ((i as f32) * 0.017).sin() * 0.25)
                .collect();
            let baton: Vec<f32> = (0..bt * h16 * w16 * stage2.d_baton)
                .map(|i| ((i as f32) * 0.011).cos() * 0.1)
                .collect();

            let t_rgb = ctx.upload_tensor(&rgb, [bt, h, w, 3]).unwrap();
            let t_baton = ctx.upload_tensor(&baton, [bt, h16, w16, stage2.d_baton]).unwrap();
            let t_out = stage2.forward::<f32>(&ctx, t_rgb, t_baton).unwrap();
            assert_eq!(t_out.dims(), [bt, h, w, STAGE2_OUT_CHANNELS]);
            let out = ctx.download_tensor::<f32, 4>(t_out).unwrap();
            assert!(out.iter().all(|v| v.is_finite()), "{target} Stage2 output had non-finite values");
        }
    }
}
