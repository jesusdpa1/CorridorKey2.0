//! Decoding inputs and encoding outputs for `corridorkey infer` (`video_io.py`): ffmpeg pipes for
//! video, the `image` crate for frame directories, the `exr` crate for EXR sequences. The keyer
//! itself only sees [`FrameReader`] / [`FrameSink`].

pub mod images;
pub mod video;
pub mod writers;

use std::path::Path;

use corridorkey_ml::pipeline::FrameReader;

/// Clip name of an input (`raw_clip_name`): the file stem, or the parent directory's name for
/// generic names (`input`, `frames`, `rgb`).
pub fn clip_name(path: &Path) -> String {
    let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    if ["input", "frames", "rgb"].contains(&stem.to_lowercase().as_str())
        && let Some(parent) = path.parent().and_then(|p| p.file_name())
    {
        return parent.to_string_lossy().into_owned();
    }
    stem
}

/// What an input decodes to.
pub struct OpenedInput {
    pub reader: Box<dyn FrameReader>,
    pub native_hw: (usize, usize),
    pub fps: f64,
    pub frames: usize,
    /// Frame directories ignore the linear adjustment (`probe_frame_dir_meta`).
    pub is_frame_dir: bool,
}
