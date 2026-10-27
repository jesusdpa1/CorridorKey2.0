//! Hint inputs (`load_initial_hint_frame`, `read_hint_image`, `make_hint_reader`,
//! `preprocess_hint_frame`): mattes guiding Stage 1, as `[h0, w0]` values in [0, 1] at the native
//! size (the pipeline pads them like the frames).
//!
//! The product reads them two ways, reproduced here:
//! - the first-frame hint (`read_hint_image`): the alpha channel of a 4-channel image, else the
//!   first channel in OpenCV's order (blue of an RGB image), / 255 or / 65535;
//! - hint videos and hint frame directories (`preprocess_hint_frame`): OpenCV's integer BGR→gray
//!   (`(4899 R + 9617 G + 1868 B + 8192) >> 14`), / 255 or / 65535.
//!
//! Hints must be at the native size (the product would Lanczos-resize other sizes; with model scale
//! 1, only the native size is valid for the first-frame hint anyway).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use corridorkey_core::{CkError, CkResult};
use corridorkey_ml::pipeline::{FrameReader, HintReader};
use image::DynamicImage;

use crate::images::frame_paths;
use crate::video::{VideoInfo, VideoReader, ffprobe_for, probe};

/// OpenCV's `COLOR_BGR2GRAY` on integer pixels.
fn cv_gray(r: u32, g: u32, b: u32) -> u32 {
    (4899 * r + 9617 * g + 1868 * b + 8192) >> 14
}

fn check_hw(path: &Path, hw: (usize, usize), want: (usize, usize)) -> Result<()> {
    if hw != want {
        bail!("{}: hint is {}x{}, the frames are {}x{} (hints must be at the frames' size)", path.display(), hw.1, hw.0, want.1, want.0);
    }
    Ok(())
}

fn open_image(path: &Path) -> Result<DynamicImage> {
    if path.extension().is_some_and(|e| e == "exr") {
        bail!("{}: EXR hints are not supported by the Rust engine yet (use PNG)", path.display());
    }
    image::open(path).with_context(|| format!("reading hint {}", path.display()))
}

/// `read_hint_image` (see the module docs).
fn hint_image_values(img: &DynamicImage) -> Result<Vec<f32>> {
    let v8 = |v: u8| v as f32 / 255.0;
    let v16 = |v: u16| v as f32 / 65535.0;
    Ok(match img {
        DynamicImage::ImageLuma8(i) => i.pixels().map(|p| v8(p[0])).collect(),
        DynamicImage::ImageLumaA8(i) => i.pixels().map(|p| v8(p[1])).collect(),
        DynamicImage::ImageRgb8(i) => i.pixels().map(|p| v8(p[2])).collect(),
        DynamicImage::ImageRgba8(i) => i.pixels().map(|p| v8(p[3])).collect(),
        DynamicImage::ImageLuma16(i) => i.pixels().map(|p| v16(p[0])).collect(),
        DynamicImage::ImageLumaA16(i) => i.pixels().map(|p| v16(p[1])).collect(),
        DynamicImage::ImageRgb16(i) => i.pixels().map(|p| v16(p[2])).collect(),
        DynamicImage::ImageRgba16(i) => i.pixels().map(|p| v16(p[3])).collect(),
        other => bail!("unsupported hint pixel format {:?}", other.color()),
    })
}

/// `preprocess_hint_frame` on a decoded image (see the module docs).
fn hint_frame_values(img: &DynamicImage) -> Result<Vec<f32>> {
    let v8 = |v: u32| v as f32 / 255.0;
    let v16 = |v: u32| v as f32 / 65535.0;
    let gray = |p: &[u8]| cv_gray(p[0] as u32, p[1] as u32, p[2] as u32);
    let gray16 = |p: &[u16]| cv_gray(p[0] as u32, p[1] as u32, p[2] as u32);
    Ok(match img {
        DynamicImage::ImageLuma8(i) => i.pixels().map(|p| v8(p[0] as u32)).collect(),
        // OpenCV decodes gray + alpha as BGRA: gray of equal channels is the gray value
        DynamicImage::ImageLumaA8(i) => i.pixels().map(|p| v8(p[0] as u32)).collect(),
        DynamicImage::ImageRgb8(i) => i.pixels().map(|p| v8(gray(&p.0))).collect(),
        DynamicImage::ImageRgba8(i) => i.pixels().map(|p| v8(gray(&p.0))).collect(),
        DynamicImage::ImageLuma16(i) => i.pixels().map(|p| v16(p[0] as u32)).collect(),
        DynamicImage::ImageLumaA16(i) => i.pixels().map(|p| v16(p[0] as u32)).collect(),
        DynamicImage::ImageRgb16(i) => i.pixels().map(|p| v16(gray16(&p.0))).collect(),
        DynamicImage::ImageRgba16(i) => i.pixels().map(|p| v16(gray16(&p.0))).collect(),
        other => bail!("unsupported hint pixel format {:?}", other.color()),
    })
}

/// The first-frame hint (`load_initial_hint_frame`): `path`, or the first frame of the directory
/// `path`, as `[h0, w0]` values.
pub fn load_first_frame_hint(path: &Path, native_hw: (usize, usize)) -> Result<Vec<f32>> {
    let file = if path.is_dir() { frame_paths(path)?.remove(0) } else { path.to_path_buf() };
    let img = open_image(&file)?;
    check_hw(&file, (img.height() as usize, img.width() as usize), native_hw)?;
    hint_image_values(&img)
}

/// A hint frame directory (`SequentialHintFrameDirWindowReader`).
pub struct HintDirReader {
    paths: Vec<PathBuf>,
    native_hw: (usize, usize),
}

impl HintReader for HintDirReader {
    fn read_hints(&mut self, start: usize, count: usize) -> CkResult<Vec<f32>> {
        let mut out = Vec::with_capacity(count * self.native_hw.0 * self.native_hw.1);
        for p in &self.paths[start..start + count] {
            let values = open_image(p)
                .and_then(|img| {
                    check_hw(p, (img.height() as usize, img.width() as usize), self.native_hw)?;
                    hint_frame_values(&img)
                })
                .map_err(|e| CkError::Io(format!("{e:#}")))?;
            out.extend(values);
        }
        Ok(out)
    }
}

/// A hint video (`SequentialHintVideoWindowReader`): decoded like the input video, from the same
/// start frame.
pub struct HintVideoReader {
    video: VideoReader,
}

impl HintReader for HintVideoReader {
    fn read_hints(&mut self, start: usize, count: usize) -> CkResult<Vec<f32>> {
        let rgb = self.video.read_rgb8(start, count)?;
        Ok(rgb.chunks_exact(3).map(|p| cv_gray(p[0] as u32, p[1] as u32, p[2] as u32) as f32 / 255.0).collect())
    }
}

/// The hint reader of `path` (`make_hint_reader`): a frame directory or a video, checked to cover
/// `start_frame + frames` frames at `native_hw`.
pub fn open_hint_reader(path: &Path, ffmpeg: &str, native_hw: (usize, usize), start_frame: usize, frames: usize) -> Result<Box<dyn HintReader>> {
    if path.is_dir() {
        let paths = frame_paths(path)?;
        if paths.len() < frames {
            bail!("{}: only {} hint frames for {frames} source frames", path.display(), paths.len());
        }
        return Ok(Box::new(HintDirReader { paths, native_hw }));
    }
    let info: VideoInfo = probe(&ffprobe_for(ffmpeg), path)?;
    check_hw(path, (info.height, info.width), native_hw)?;
    if info.frames < start_frame + frames {
        bail!(
            "hint video {} has only {} frames, but the source requires {} (start={start_frame}, num={frames})",
            path.display(),
            info.frames,
            start_frame + frames
        );
    }
    Ok(Box::new(HintVideoReader { video: VideoReader::new(ffmpeg, path, &info, start_frame) }))
}
