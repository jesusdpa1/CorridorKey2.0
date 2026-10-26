//! Frame-directory input (`frame_paths`, `probe_frame_dir_meta`, `read_image_rgb`): 8-bit PNG / JPEG
//! frames, sorted by path. Scene-linear EXR frames need the float upload path and are not read here
//! yet.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use corridorkey_core::{CkError, CkResult};
use corridorkey_ml::pipeline::FrameReader;

/// Frame files of `dir` with the product's extensions (matched as written, lower case), sorted.
pub fn frame_paths(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("listing {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|e| e.to_str()).is_some_and(|e| ["exr", "png", "jpg", "jpeg"].contains(&e)))
        .collect();
    paths.sort();
    if paths.is_empty() {
        bail!("no frames (exr, png, jpg, jpeg) under {}", dir.display());
    }
    if let Some(p) = paths.iter().find(|p| p.extension().is_some_and(|e| e == "exr")) {
        bail!("{}: EXR frame directories are not supported by the Rust engine yet (use --engine python)", p.display());
    }
    Ok(paths)
}

/// Reads frames as 8-bit RGB (`cv2.imread(..., IMREAD_COLOR)`, converted to RGB).
pub struct ImageDirReader {
    paths: Vec<PathBuf>,
    hw: (usize, usize),
}

impl ImageDirReader {
    pub fn new(paths: Vec<PathBuf>) -> Result<Self> {
        let first = image::open(&paths[0]).with_context(|| format!("reading {}", paths[0].display()))?;
        Ok(Self { hw: (first.height() as usize, first.width() as usize), paths })
    }

    pub fn native_hw(&self) -> (usize, usize) {
        self.hw
    }

    pub fn len(&self) -> usize {
        self.paths.len()
    }
}

impl FrameReader for ImageDirReader {
    fn read_rgb8(&mut self, start: usize, count: usize) -> CkResult<Vec<u8>> {
        let mut out = Vec::with_capacity(count * self.hw.0 * self.hw.1 * 3);
        for p in &self.paths[start..start + count] {
            let img = image::open(p).map_err(|e| CkError::Io(format!("{}: {e}", p.display())))?.into_rgb8();
            if (img.height() as usize, img.width() as usize) != self.hw {
                return Err(CkError::Shape(format!("{}: {}x{} differs from the first frame", p.display(), img.width(), img.height())));
            }
            out.extend_from_slice(img.as_raw());
        }
        Ok(out)
    }
}
