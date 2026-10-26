//! `infer` on the Rust engine: the clip pipeline (`corridorkey_ml::pipeline::Keyer`) between the
//! ffmpeg / image readers and the ffmpeg / EXR writers (`runtime.run_inference`).

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use burn_tensor::DType;
use corridorkey_base::core::CkFloat;
use corridorkey_base::frame::RenderOutput;
use corridorkey_core::{ComputeTarget, FrameMeta, LinearAdjust};
use corridorkey_ml::MlContext;
use corridorkey_ml::pipeline::{FrameReader, Keyer, ModelPaths, PipelineConfig};
use half::f16;

use super::infer::{ExrDtype, InferArgs, Output, OutputFormat, Precision, StartMode};
use crate::io::images::{ImageDirReader, frame_paths};
use crate::io::video::{VideoReader, probe};
use crate::io::writers::{EncoderConfig, OutputSink};
use crate::io::{OpenedInput, clip_name};

/// `hann_chunk` / `hann_stride` without `--hann-chunk` (`InferenceConfig` defaults; the product can
/// instead pick them from free GPU memory, which the Rust engine does not do yet).
const DEFAULT_CHUNK: usize = 80;

fn unsupported(args: &InferArgs) -> Result<()> {
    let h = &args.hints;
    let mut not_yet = vec![];
    if h.hint_first_frame.is_some() {
        not_yet.push("--hint-first-frame");
    }
    if h.hint_video.is_some() {
        not_yet.push("--hint-video");
    }
    if args.memory.low_vram {
        not_yet.push("--low-vram");
    }
    if args.frames.model_scale != 1.0 {
        not_yet.push("--model-scale other than 1 (the product's render path also needs 1)");
    }
    if args.frames.frame_dir_linear {
        not_yet.push("--frame-dir-linear");
    }
    if !not_yet.is_empty() {
        bail!("not supported by the Rust engine yet: {} (use --engine python)", not_yet.join(", "));
    }
    Ok(())
}

fn open_input(args: &InferArgs) -> Result<OpenedInput> {
    let f = &args.frames;
    if args.input.is_dir() {
        let mut paths = frame_paths(&args.input)?;
        if let Some(n) = f.num_frames.filter(|&n| n > 0) {
            paths.truncate(n);
        }
        let reader = ImageDirReader::new(paths)?;
        return Ok(OpenedInput { native_hw: reader.native_hw(), fps: f.frame_dir_fps, frames: reader.len(), reader: Box::new(reader), is_frame_dir: true });
    }
    let ffprobe = Path::new(&args.encoding.ffmpeg_bin).with_file_name("ffprobe");
    let ffprobe = if args.encoding.ffmpeg_bin.contains('/') { ffprobe.display().to_string() } else { "ffprobe".into() };
    let info = probe(&ffprobe, &args.input)?;
    // `probe_video_meta`: frames to load and the start frame of the chosen window.
    let n_total = info.frames;
    let n = f.num_frames.filter(|&n| n > 0).map_or(n_total, |n| n.min(n_total));
    let max_start = n_total.saturating_sub(n);
    let start = match (f.start_mode, f.num_frames.is_some_and(|n| n > 0) && max_start > 0) {
        (StartMode::Begin, _) | (_, false) => 0,
        (StartMode::Middle, true) => max_start / 2,
        (StartMode::RandomMiddle, true) => bail!("--start-mode random-middle needs NumPy's generator; not in the Rust engine yet"),
    };
    let reader = VideoReader::new(&args.encoding.ffmpeg_bin, &args.input, &info, start);
    Ok(OpenedInput { native_hw: (info.height, info.width), fps: info.fps, frames: n, reader: Box::new(reader), is_frame_dir: false })
}

/// Runs `infer` on the Rust engine with the runtime `target`.
pub fn run(root: &Path, target: ComputeTarget, args: &InferArgs) -> Result<()> {
    unsupported(args)?;
    let started = Instant::now();
    let mut input = open_input(args)?;
    let mut paths = ModelPaths::from_root(root);
    if let Some(w) = &args.weights {
        paths.checkpoint = w.clone();
    }
    let quantum = corridorkey_ml::SafetensorsFile::open(&paths.checkpoint)?.checkpoint_header()?.metadata.shape_quantum;
    let mut meta = FrameMeta::new(input.native_hw, 1.0, quantum, input.fps, input.frames)?;
    if !input.is_frame_dir {
        let c = &args.color;
        meta.linear_adjust = LinearAdjust { gain: c.linear_brightness, contrast: c.linear_contrast, pivot: c.linear_contrast_pivot };
    }
    let ctx = MlContext::new(target)?;
    let precision = match args.engine.precision {
        Precision::Auto => match ctx.preferred_dtype() {
            DType::F16 => Precision::F16,
            _ => Precision::F32,
        },
        p => p,
    };
    let (h0, w0) = input.native_hw;
    println!(
        "  {} frames @ {}x{} (native {w0}x{h0}, fps {:.2}, pad {:?}) on {target} ({precision:?})",
        meta.num_frames, meta.shape_hw.1, meta.shape_hw.0, meta.fps, meta.pad
    );
    match precision {
        Precision::F16 => run_typed::<f16>(root, &ctx, &paths, &meta, input.reader.as_mut(), args, started),
        _ => run_typed::<f32>(root, &ctx, &paths, &meta, input.reader.as_mut(), args, started),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_typed<F: CkFloat>(
    _root: &Path,
    ctx: &MlContext,
    paths: &ModelPaths,
    meta: &FrameMeta,
    reader: &mut dyn FrameReader,
    args: &InferArgs,
    started: Instant,
) -> Result<()> {
    let t = Instant::now();
    let keyer = Keyer::load::<F>(ctx, paths).context("loading the networks (did you run scripts/convert_foundations.py?)")?;
    println!("  loaded networks in {:.1} s", t.elapsed().as_secs_f64());

    let outputs: Vec<RenderOutput> = args
        .outputs
        .iter()
        .map(|o| match o {
            Output::Alpha => RenderOutput::Alpha,
            Output::Fg => RenderOutput::Fg,
            Output::Checker => RenderOutput::Checker,
            Output::Cutout => RenderOutput::Cutout,
        })
        .collect();
    let chunk = args.frames.hann_chunk.unwrap_or(DEFAULT_CHUNK).min(meta.num_frames).max(1);
    let stride = args.frames.hann_stride.unwrap_or((chunk / 2).max(1)).min(chunk);
    let cfg = PipelineConfig {
        chunk,
        stride,
        carry_hint: !args.hints.no_carry_hint,
        carry_hint_quality: args.hints.carry_hint_quality,
        stage2_batch: args.memory.stage2_batch,
        outputs: outputs.clone(),
        exr: args.encoding.output_format == OutputFormat::Exr,
        despill: args.color.despill_strength,
        cutout_linear: args.color.cutout_linear,
        product_rounding: true,
    };
    let e = &args.encoding;
    let enc = EncoderConfig {
        ffmpeg: e.ffmpeg_bin.clone(),
        codec: e.ffmpeg_codec.clone(),
        crf: e.crf,
        preset: e.preset.clone(),
        pix_fmt: e.pix_fmt.clone(),
        bitrate: e.bitrate.clone(),
        threads: e.ffmpeg_threads,
    };
    let clip = clip_name(&args.input);
    let exr = cfg.exr.then_some(e.exr_dtype == ExrDtype::Half);
    let (h0, w0) = meta.native_hw;
    let (mut sink, written) = OutputSink::open(&args.output_dir, &clip, &outputs, (w0, h0), meta.fps, cfg.despill, exr, cfg.cutout_linear, &enc)?;
    write_settings(&args.output_dir.join(format!("{clip}_settings.txt")), args, meta, &cfg)?;

    let t = Instant::now();
    keyer.run::<F>(ctx, meta, reader, &mut sink, &cfg)?;
    sink.finish()?;
    let secs = t.elapsed().as_secs_f64();
    println!(
        "  wrote {} output(s) for {clip} ({} frames @ {w0}x{h0}) in {secs:.1} s ({:.2} frames/s), total {:.1} s",
        written.len(),
        meta.num_frames,
        meta.num_frames as f64 / secs,
        started.elapsed().as_secs_f64()
    );
    for p in &written {
        println!("    {}", p.display());
    }
    Ok(())
}

/// `<clip>_settings.txt`, the run log `run_inference` writes next to the outputs.
fn write_settings(path: &Path, args: &InferArgs, meta: &FrameMeta, cfg: &PipelineConfig) -> Result<()> {
    std::fs::create_dir_all(path.parent().unwrap_or(Path::new(".")))?;
    let (h0, w0) = meta.native_hw;
    let c = &args.color;
    let text = format!(
        "CorridorKey v2 Inference Settings Log\n====================================\n\
         Input: {}\nOutput Directory: {}\nResolution: {w0}x{h0}\nFPS: {}\nOutputs: {}\nOutput Format: {}\n\
         Hint First Frame: None\nHint Video: None\nHint Quality (Trust): {}\nCarry Hint: {}\nCarry Hint Quality: {}\n\
         Despill Strength: {}\nCutout Linear: {}\nFrame Dir Linear: {}\nHann Chunk: {}\nHann Stride: {}\nLow VRAM: false\n\
         Model Scale: 1\nNum Frames: {}\nStart Mode: {:?}\nSeed: {}\nLinear Brightness: {}\nLinear Contrast: {}\n\
         Linear Contrast Pivot: {}\nEngine: rust\n",
        args.input.display(),
        args.output_dir.display(),
        meta.fps,
        cfg.outputs.iter().map(|o| format!("{o:?}").to_lowercase()).collect::<Vec<_>>().join(", "),
        if cfg.exr { "exr" } else { "video" },
        args.hints.hint_quality,
        cfg.carry_hint,
        cfg.carry_hint_quality,
        cfg.despill,
        cfg.cutout_linear,
        args.frames.frame_dir_linear,
        cfg.chunk,
        cfg.stride,
        meta.num_frames,
        args.frames.start_mode,
        args.frames.seed,
        c.linear_brightness,
        c.linear_contrast,
        c.linear_contrast_pivot,
    );
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))
}
