#!/usr/bin/env python3
"""End-to-end fp32 reference of the product's `runtime.run_inference`, for the Rust pipeline.

Runs the product's own driver on a short PNG frame sequence (the portrait with horizontal motion)
with two overlapping Stage 1 windows (`hann_chunk` 8, `hann_stride` 4 over 12 frames), carried
hints, every output as float32 EXR frames. Three patches make it an fp32 reference without changing
its logic:

- `torch.autocast` is disabled (the product runs the backbones and stages under bf16 autocast);
- `video_io.preprocess_frame_dir_path` keeps the model and display views in float32 (the product
  rounds them to bf16 on the CPU);
- `runtime.ExrSequenceWriter` is replaced by a collector of the exact arrays handed to the writer.

The product's other roundings stay, being part of its logic: batons to float16 after the Hann
blend, carried hints and hint quality to bf16.

Writes `frames/NNNN.png`, `<output>.f32` (`[T, h, w, C]` as written: BGR / BGRA / alpha) and
`meta.json`.

usage: pipeline_reference.py <out_dir>
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from corridorkey import color_utils as cu  # noqa: E402
from corridorkey import runtime, video_io  # noqa: E402
from corridorkey.runtime import InferenceConfig, RuntimePaths, load_model  # noqa: E402
from corridorkey.video_io import WriterConfig  # noqa: E402

FRAMES, CHUNK, STRIDE = 12, 8, 4
OUTPUTS = ("alpha", "fg", "cutout", "checker")


def preprocess_frame_dir_path_f32(path: str, meta: dict, *, need_model: bool):
    """`video_io.preprocess_frame_dir_path` without the bf16 rounding of its views."""
    h_out, w_out = meta["shape_hw"]
    h_model, w_model = meta["model_native_hw"]
    display, linear = video_io._read_frame_dir_views(path, exr_linear=bool(meta.get("exr_linear", False)))
    display = video_io.resize_hwc_lanczos(display, (h_model, w_model))
    linear = video_io.resize_hwc_lanczos(linear, (h_model, w_model))
    display, _ = video_io.center_pad_to_shape(np.clip(display, 0.0, 1.0), h_out, w_out)
    display_t = torch.from_numpy(display.transpose(2, 0, 1).copy()).float()
    if not need_model:
        return None, display_t, display_t
    linear, _ = video_io.center_pad_to_shape(linear, h_out, w_out)
    model_t = torch.from_numpy(cu.linear_to_asinh(linear).transpose(2, 0, 1).copy()).float()
    return model_t, display_t, display_t


class Collector:
    """Stands in for `ExrSequenceWriter`: keeps every frame handed to it."""

    frames: dict[str, list[np.ndarray]] = {}

    def __init__(self, dir_path, fps, size, *, channels, dtype, frame_name):
        self.name = frame_name
        Collector.frames.setdefault(frame_name, [])

    def write_many(self, frames):
        Collector.frames[self.name].extend(np.asarray(f, dtype=np.float32) for f in frames)

    def write(self, frame):
        self.write_many([frame])

    def release(self):
        pass


def main() -> None:
    out = Path(sys.argv[1])
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    portrait = cv2.imread(str(ROOT / "data" / "green_screen_portrait.png"), cv2.IMREAD_COLOR)
    for i in range(FRAMES):
        cv2.imwrite(str(frames_dir / f"{i:04d}.png"), np.roll(portrait, 6 * i, axis=1))

    torch.autocast = lambda *a, **k: contextlib.nullcontext()  # fp32 reference
    video_io.preprocess_frame_dir_path = preprocess_frame_dir_path_f32
    runtime.ExrSequenceWriter = Collector
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = load_model(RuntimePaths.from_package(ROOT), "cuda")
    cfg = InferenceConfig(
        hann_chunk=CHUNK, hann_stride=STRIDE, outputs=set(OUTPUTS), output_format="exr", exr_dtype="float",
        temp_dir=out / "tmp",
    )
    runtime.run_inference(model, str(frames_dir), str(out / "unused"), cfg, WriterConfig(),
                          initial_hint_path=None, hint_video_path=None)

    shapes = {}
    for name, frames in Collector.frames.items():
        arr = np.stack(frames).astype(np.float32)
        arr = arr[..., None] if arr.ndim == 3 else arr
        arr.tofile(out / f"{name}.f32")
        shapes[name] = list(arr.shape)
    meta = {"frames": FRAMES, "chunk": CHUNK, "stride": STRIDE, "shapes": shapes,
            "carry_hint_quality": cfg.carry_hint_quality, "despill": cfg.despill_strength}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
