#!/usr/bin/env python3
"""CorridorKey 2.0: Unified Inference & Latency Comparison Benchmark Suite

Evaluates:
  1. PyTorch CUDA (reference)
  2. CorridorKey Rust CUDA (CubeCL / Burn)
  3. CorridorKey Rust WebGPU (CubeCL / Burn WGPU)

Outputs:
  • benchmarks/results/benchmark_report.json
  • benchmarks/results/benchmark_summary.md
  • benchmarks/results/comparison_grid.png
  • Individual alpha, composite, and error heat-map assets
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from corridorkey.runtime import InferenceConfig, RuntimePaths, load_model
from corridorkey.video_io import ceil_to_multiple, make_checker


def run_monitored_subprocess(cmd: list[str]) -> tuple[subprocess.CompletedProcess, float, float]:
    """Runs a subprocess and monitors peak GPU VRAM (MB) and peak Host RAM RSS (MB)."""
    peak_vram = [0.0]
    peak_rss = [0.0]
    stop_event = threading.Event()

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    pid = proc.pid

    def monitor():
        while not stop_event.is_set():
            # Check host RSS from /proc/<pid>/status
            try:
                with open(f"/proc/{pid}/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            rss_kb = float(line.split()[1])
                            peak_rss[0] = max(peak_rss[0], rss_kb / 1024.0)
                            break
            except Exception:
                pass
            # Check GPU VRAM from nvidia-smi
            try:
                smi = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=0.1
                )
                for row in smi.stdout.strip().split("\n"):
                    if row:
                        parts = row.split(",")
                        if len(parts) >= 2 and int(parts[0].strip()) == pid:
                            peak_vram[0] = max(peak_vram[0], float(parts[1].strip()))
            except Exception:
                pass
            time.sleep(0.015)

    t = threading.Thread(target=monitor, daemon=True)
    t.start()
    stdout, stderr = proc.communicate()
    stop_event.set()
    t.join(timeout=0.2)

    completed = subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    return completed, peak_vram[0], peak_rss[0]


def parse_args():
    parser = argparse.ArgumentParser(description="CorridorKey Unified Latency & Parity Benchmark")
    parser.add_argument("--input", default=str(ROOT / "data" / "green_screen_portrait.png"),
                        help="Input green screen image")
    parser.add_argument("--output-dir", default=str(ROOT / "benchmarks" / "results"),
                        help="Directory to save benchmark results")
    parser.add_argument("--iters", type=int, default=5,
                        help="Number of iterations for warm latency timing")
    parser.add_argument("--skip-wgpu", action="store_true",
                        help="Skip WebGPU benchmark and only compare PyTorch vs Rust CUDA")
    return parser.parse_args()


def pad_to_multiple(img_np: np.ndarray, quantum: int = 128):
    h, w, c = img_np.shape
    h_out = ceil_to_multiple(h, quantum)
    w_out = ceil_to_multiple(w, quantum)
    pad_l = (w_out - w) // 2
    pad_r = w_out - w - pad_l
    pad_t = (h_out - h) // 2
    pad_b = h_out - h - pad_t

    t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    padded = F.pad(t, (pad_l, pad_r, pad_t, pad_b), mode="reflect")
    return padded, (pad_l, pad_r, pad_t, pad_b), (h, w)


def unpad(img_np: np.ndarray, pad: tuple[int, int, int, int], orig_hw: tuple[int, int]):
    pad_l, pad_r, pad_t, pad_b = pad
    h, w = orig_hw
    return img_np[pad_t : pad_t + h, pad_l : pad_l + w]


def simple_despill(fg: np.ndarray) -> np.ndarray:
    res = fg.copy()
    rb_max = np.maximum(res[:, :, 0], res[:, :, 2])
    res[:, :, 1] = np.minimum(res[:, :, 1], rb_max * 1.05)
    return np.clip(res, 0.0, 1.0)


def to_pil(arr: np.ndarray, is_gray: bool = False) -> Image.Image:
    u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if is_gray:
        return Image.fromarray(u8, mode="L").convert("RGB")
    return Image.fromarray(u8, mode="RGB")


def diff_heatmap(diff: np.ndarray, scale: float = 100.0) -> Image.Image:
    val = np.clip(diff * scale, 0.0, 1.0)
    if val.ndim == 3:
        val = val.mean(axis=2)
    r = np.clip(val * 2.0, 0.0, 1.0)
    g = np.clip(val * 2.0 - 0.7, 0.0, 1.0)
    b = np.clip(val * 3.0 - 2.0, 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=2)
    return to_pil(rgb)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = out_dir / ".staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    img_path = Path(args.input)
    if not img_path.exists():
        raise FileNotFoundError(f"Input image not found: {img_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("  CorridorKey 2.0: Unified Inference & Latency Benchmark")
    print(f"  Input: {img_path}")
    print(f"  Target Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print("=" * 72)

    pil_in = Image.open(img_path).convert("RGB")
    rgb_orig = np.array(pil_in, dtype=np.float32) / 255.0
    h_orig, w_orig, _ = rgb_orig.shape

    padded_t, pad, (h_orig, w_orig) = pad_to_multiple(rgb_orig, quantum=128)
    _, _, H_pad, W_pad = padded_t.shape
    print(f"Resolution: Native {w_orig}x{h_orig} -> Padded {W_pad}x{H_pad} (pad={pad})")

    # -------------------------------------------------------------------------
    # 1. PyTorch CUDA Benchmark
    # -------------------------------------------------------------------------
    print("\n[1/3] Benchmarking PyTorch CUDA...")
    torch.cuda.reset_peak_memory_stats()
    t_load_start = time.perf_counter()
    paths = RuntimePaths.from_package(ROOT)
    model = load_model(paths, str(device)).to(device)
    model.eval()
    torch_load_ms = (time.perf_counter() - t_load_start) * 1000
    print(f"  Loaded PyTorch model in {torch_load_ms:.1f} ms")

    rgb_dev = padded_t.to(device)
    rgb_5d = rgb_dev.unsqueeze(1)
    hint = torch.zeros(1, 1, 1, H_pad, W_pad, device=device)
    hq = torch.zeros(1, 1, device=device)

    # Foundations
    rgb_cradio = F.interpolate(rgb_dev, size=(H_pad // 4, W_pad // 4), mode="area").unsqueeze(1)
    rgb_moge = F.interpolate(rgb_dev, size=(7 * H_pad // 32, 7 * W_pad // 32), mode="area").unsqueeze(1)

    with torch.no_grad():
        foundation_feats = model.foundations(
            rgb_moge=rgb_moge.reshape(1, 3, *rgb_moge.shape[-2:]),
            rgb_cradio=rgb_cradio.reshape(1, 3, *rgb_cradio.shape[-2:]),
            rgb_rvm=rgb_cradio.reshape(1, 1, 3, *rgb_cradio.shape[-2:]),
        )
        vfm_parts = []
        if "cradio" in foundation_feats and foundation_feats["cradio"] is not None:
            vfm_parts.append(foundation_feats["cradio"].reshape(1, 1, -1, H_pad // 64, W_pad // 64))
        if "moge" in foundation_feats and foundation_feats["moge"] is not None:
            vfm_parts.append(foundation_feats["moge"].reshape(1, 1, -1, H_pad // 64, W_pad // 64))
        if "rvm" in foundation_feats and foundation_feats["rvm"] is not None:
            vfm_parts.append(foundation_feats["rvm"].reshape(1, 1, -1, H_pad // 64, W_pad // 64))
        vfm_cat = torch.cat(vfm_parts, dim=2)

    # Dump input tensors for Rust
    vfm_c = vfm_cat.shape[2]
    meta = {"b": 1, "t": 1, "h": H_pad, "w": W_pad, "vfm_channels": vfm_c}
    (staging_dir / "meta.json").write_text(json.dumps(meta))
    rgb_hint_bthwc = torch.cat([rgb_5d, hint], dim=2).permute(0, 1, 3, 4, 2).contiguous().cpu().numpy().astype(np.float32)
    rgb_bthwc = rgb_5d.permute(0, 1, 3, 4, 2).contiguous().cpu().numpy().astype(np.float32)
    vfm_64_bthwc = vfm_cat.permute(0, 1, 3, 4, 2).contiguous().cpu().numpy().astype(np.float32)
    (staging_dir / "rgb_hint_bthwc.f32").write_bytes(rgb_hint_bthwc.tobytes())
    (staging_dir / "rgb_bthwc.f32").write_bytes(rgb_bthwc.tobytes())
    (staging_dir / "vfm_64_bthwc.f32").write_bytes(vfm_64_bthwc.tobytes())
    (staging_dir / "hint_quality_bt.f32").write_bytes(hq.cpu().numpy().astype(np.float32).tobytes())

    # PyTorch Cold Run
    torch.cuda.synchronize()
    t_cold_start = time.perf_counter()
    with torch.no_grad():
        s1 = model.stage1(rgb_5d, hint, vfm_parts[1], cradio_feat=vfm_parts[0] if model.stage1.cradio_dim > 0 else None,
                          rvm_feat=vfm_parts[2] if model.stage1.rvm_dim > 0 else None, hint_quality=hq, t_stride=1.0)
        s2 = model.stage2(rgb_5d, s1["baton"])
        if isinstance(s2, tuple):
            s2 = s2[0]
    torch.cuda.synchronize()
    torch_cold_ms = (time.perf_counter() - t_cold_start) * 1000

    # PyTorch Warm Timed Iterations
    s1_times = []
    s2_times = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            s1 = model.stage1(rgb_5d, hint, vfm_parts[1], cradio_feat=vfm_parts[0] if model.stage1.cradio_dim > 0 else None,
                              rvm_feat=vfm_parts[2] if model.stage1.rvm_dim > 0 else None, hint_quality=hq, t_stride=1.0)
        torch.cuda.synchronize()
        s1_times.append((time.perf_counter() - t0) * 1000)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.no_grad():
            s2 = model.stage2(rgb_5d, s1["baton"])
            if isinstance(s2, tuple):
                s2 = s2[0]
        torch.cuda.synchronize()
        s2_times.append((time.perf_counter() - t1) * 1000)

    s1_times.sort()
    s2_times.sort()
    torch_s1_ms = s1_times[len(s1_times) // 2]
    torch_s2_ms = s2_times[len(s2_times) // 2]
    torch_warm_total_ms = torch_s1_ms + torch_s2_ms
    # Host RSS for PyTorch
    def get_current_rss_mb() -> float:
        try:
            with open(f"/proc/{os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024.0
        except Exception:
            pass
        return 0.0

    torch_rss_mb = get_current_rss_mb()
    torch_vram_alloc_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    torch_vram_res_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)

    alpha_torch = s2[0, 0, 0].clamp(0.0, 1.0).cpu().numpy()
    fg_torch = s2[0, 0, 1:4].permute(1, 2, 0).cpu().numpy()
    alpha_torch_unpad = unpad(alpha_torch, pad, (h_orig, w_orig))
    fg_torch_unpad = unpad(fg_torch, pad, (h_orig, w_orig))

    torch_fps = 1000.0 / torch_warm_total_ms
    torch_mpix = (w_orig * h_orig * torch_fps) / 1e6
    print(f"  PyTorch Stage 1: {torch_s1_ms:.2f} ms | Stage 2: {torch_s2_ms:.2f} ms | Total: {torch_warm_total_ms:.2f} ms")
    print(f"  Throughput: {torch_fps:.1f} FPS ({torch_mpix:.2f} MPixels/s)")
    print(f"  PyTorch Memory: Peak VRAM {torch_vram_alloc_mb:.1f} MB (reserved {torch_vram_res_mb:.1f} MB) | Host RSS {torch_rss_mb:.1f} MB")

    # Free PyTorch VRAM completely
    del model, rgb_dev, rgb_5d, hint, hq, rgb_cradio, rgb_moge, vfm_cat, s1, s2
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 2. Rust CUDA Benchmark
    # -------------------------------------------------------------------------
    print("\n[2/3] Benchmarking CorridorKey Rust (CUDA)...")
    rust_cuda_out = out_dir / "rust_cuda"
    rust_cuda_out.mkdir(parents=True, exist_ok=True)
    cmd_cuda = [
        str(ROOT / "target" / "release" / "run_stage1_stage2"),
        str(staging_dir), str(rust_cuda_out),
        "--runtime", "cuda"
    ]
    res_cuda, cuda_peak_vram, cuda_peak_rss = run_monitored_subprocess(cmd_cuda)
    if res_cuda.returncode != 0:
        print("Rust CUDA stderr:\n", res_cuda.stderr)
        raise RuntimeError("Rust CUDA execution failed")

    cuda_timing = json.loads((rust_cuda_out / "rust_timing.json").read_text())
    cuda_logits = np.frombuffer((rust_cuda_out / "rust_logits_bthwc.f32").read_bytes(), dtype=np.float32).reshape(1, H_pad, W_pad, 4)
    alpha_cuda = np.clip(cuda_logits[0, :, :, 0], 0.0, 1.0)
    fg_cuda = cuda_logits[0, :, :, 1:4]
    alpha_cuda_unpad = unpad(alpha_cuda, pad, (h_orig, w_orig))
    fg_cuda_unpad = unpad(fg_cuda, pad, (h_orig, w_orig))

    cuda_alpha_diff = np.abs(alpha_torch_unpad - alpha_cuda_unpad)
    cuda_fg_diff = np.abs(fg_torch_unpad - fg_cuda_unpad)
    cuda_fps = 1000.0 / cuda_timing["warm_total_ms"]
    cuda_mpix = (w_orig * h_orig * cuda_fps) / 1e6

    print(f"  Rust CUDA Stage 1: {cuda_timing['warm_stage1_ms']:.2f} ms | Stage 2: {cuda_timing['warm_stage2_ms']:.2f} ms | Total: {cuda_timing['warm_total_ms']:.2f} ms")
    print(f"  Throughput: {cuda_fps:.1f} FPS ({cuda_mpix:.2f} MPixels/s)")
    print(f"  Memory: Peak VRAM {cuda_peak_vram:.1f} MB | Host RSS {cuda_peak_rss:.1f} MB")
    print(f"  Speedup vs PyTorch: {torch_warm_total_ms / cuda_timing['warm_total_ms']:.2f}x")
    print(f"  Alpha Parity: mean={cuda_alpha_diff.mean():.3e}, max={cuda_alpha_diff.max():.3e}")
    print(f"  FG Parity:    mean={cuda_fg_diff.mean():.3e}, max={cuda_fg_diff.max():.3e}")

    # -------------------------------------------------------------------------
    # 3. Rust WebGPU Benchmark (if not skipped)
    # -------------------------------------------------------------------------
    wgpu_timing = None
    alpha_wgpu_unpad = None
    fg_wgpu_unpad = None
    wgpu_alpha_diff = None
    wgpu_fg_diff = None
    wgpu_peak_vram = 0.0
    wgpu_peak_rss = 0.0
    wgpu_fps = 0.0
    wgpu_mpix = 0.0

    if not args.skip_wgpu:
        print("\n[3/3] Benchmarking CorridorKey Rust (WebGPU)...")
        rust_wgpu_out = out_dir / "rust_wgpu"
        rust_wgpu_out.mkdir(parents=True, exist_ok=True)
        cmd_wgpu = [
            str(ROOT / "target" / "release" / "run_stage1_stage2"),
            str(staging_dir), str(rust_wgpu_out),
            "--runtime", "wgpu"
        ]
        res_wgpu, wgpu_peak_vram, wgpu_peak_rss = run_monitored_subprocess(cmd_wgpu)
        if res_wgpu.returncode != 0:
            print("Rust WGPU stderr:\n", res_wgpu.stderr)
            raise RuntimeError("Rust WGPU execution failed")

        wgpu_timing = json.loads((rust_wgpu_out / "rust_timing.json").read_text())
        wgpu_logits = np.frombuffer((rust_wgpu_out / "rust_logits_bthwc.f32").read_bytes(), dtype=np.float32).reshape(1, H_pad, W_pad, 4)
        alpha_wgpu = np.clip(wgpu_logits[0, :, :, 0], 0.0, 1.0)
        fg_wgpu = wgpu_logits[0, :, :, 1:4]
        alpha_wgpu_unpad = unpad(alpha_wgpu, pad, (h_orig, w_orig))
        fg_wgpu_unpad = unpad(fg_wgpu, pad, (h_orig, w_orig))

        wgpu_alpha_diff = np.abs(alpha_torch_unpad - alpha_wgpu_unpad)
        wgpu_fg_diff = np.abs(fg_torch_unpad - fg_wgpu_unpad)
        wgpu_fps = 1000.0 / wgpu_timing["warm_total_ms"]
        wgpu_mpix = (w_orig * h_orig * wgpu_fps) / 1e6

        print(f"  Rust WGPU Stage 1: {wgpu_timing['warm_stage1_ms']:.2f} ms | Stage 2: {wgpu_timing['warm_stage2_ms']:.2f} ms | Total: {wgpu_timing['warm_total_ms']:.2f} ms")
        print(f"  Throughput: {wgpu_fps:.1f} FPS ({wgpu_mpix:.2f} MPixels/s)")
        print(f"  Memory: Peak VRAM {wgpu_peak_vram:.1f} MB | Host RSS {wgpu_peak_rss:.1f} MB")
        print(f"  Speedup vs PyTorch: {torch_warm_total_ms / wgpu_timing['warm_total_ms']:.2f}x")
        print(f"  Alpha Parity: mean={wgpu_alpha_diff.mean():.3e}, max={wgpu_alpha_diff.max():.3e}")

    # -------------------------------------------------------------------------
    # 4. Compositing and Visual Comparison Grid
    # -------------------------------------------------------------------------
    print("\nGenerating visual outputs and composite grid...")
    checker = make_checker(h_orig, w_orig).transpose(1, 2, 0)
    a_torch = alpha_torch_unpad[:, :, None]
    a_cuda = alpha_cuda_unpad[:, :, None]

    fg_torch_despill = simple_despill(fg_torch_unpad)
    fg_cuda_despill = simple_despill(fg_cuda_unpad)

    comp_torch = a_torch * fg_torch_despill + (1.0 - a_torch) * checker
    comp_cuda = a_cuda * fg_cuda_despill + (1.0 - a_cuda) * checker

    # Save individual asset files
    to_pil(rgb_orig).save(out_dir / "input.png")
    to_pil(alpha_torch_unpad, is_gray=True).save(out_dir / "pytorch_alpha.png")
    to_pil(alpha_cuda_unpad, is_gray=True).save(out_dir / "rust_cuda_alpha.png")
    to_pil(comp_torch).save(out_dir / "pytorch_composite.png")
    to_pil(comp_cuda).save(out_dir / "rust_cuda_composite.png")

    if alpha_wgpu_unpad is not None:
        a_wgpu = alpha_wgpu_unpad[:, :, None]
        fg_wgpu_despill = simple_despill(fg_wgpu_unpad)
        comp_wgpu = a_wgpu * fg_wgpu_despill + (1.0 - a_wgpu) * checker
        to_pil(alpha_wgpu_unpad, is_gray=True).save(out_dir / "rust_wgpu_alpha.png")
        to_pil(comp_wgpu).save(out_dir / "rust_wgpu_composite.png")

    # Create 8-panel comparison grid
    cell_w, cell_h = w_orig, h_orig
    margin_top = 35
    grid_w = cell_w * 4
    grid_h = (cell_h + margin_top) * 2

    canvas = Image.new("RGB", (grid_w, grid_h), color=(26, 26, 28))
    draw = ImageDraw.Draw(canvas)

    panels = [
        ("1. Input Green Screen Photo", to_pil(rgb_orig), 0, 0),
        ("2. PyTorch Alpha Matte (Ref)", to_pil(alpha_torch_unpad, is_gray=True), 1, 0),
        ("3. Rust CUDA Alpha Matte", to_pil(alpha_cuda_unpad, is_gray=True), 2, 0),
        (f"4. |PyTorch - Rust CUDA| x 100", diff_heatmap(cuda_alpha_diff, 100.0), 3, 0),
        ("5. Checkerboard Reference", to_pil(checker), 0, 1),
        ("6. PyTorch Despilled Composite", to_pil(comp_torch), 1, 1),
        ("7. Rust CUDA Composite", to_pil(comp_cuda), 2, 1),
        ("8. |Comp Diff| x 100 (FP32 Identity)", diff_heatmap(np.abs(comp_torch - comp_cuda), 100.0), 3, 1),
    ]

    for title, img, col, row in panels:
        x = col * cell_w
        y = row * (cell_h + margin_top)
        draw.text((x + 10, y + 8), title, fill=(245, 245, 245))
        canvas.paste(img, (x, y + margin_top))

    grid_path = out_dir / "comparison_grid.png"
    canvas.save(grid_path, quality=95)

    # -------------------------------------------------------------------------
    # 5. Output Structured Report
    # -------------------------------------------------------------------------
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "resolution": {"width": w_orig, "height": h_orig, "padded_w": W_pad, "padded_h": H_pad},
        "device": str(device),
        "pytorch_cuda": {
            "load_ms": torch_load_ms,
            "cold_ms": torch_cold_ms,
            "stage1_ms": torch_s1_ms,
            "stage2_ms": torch_s2_ms,
            "total_resident_ms": torch_warm_total_ms,
            "fps": torch_fps,
            "mpixels_per_sec": torch_mpix,
            "peak_vram_alloc_mb": torch_vram_alloc_mb,
            "peak_vram_reserved_mb": torch_vram_res_mb,
            "peak_host_rss_mb": torch_rss_mb,
        },
        "rust_cuda": {
            "load_ms": cuda_timing["load_ms"],
            "cold_ms": cuda_timing["cold_ms"],
            "stage1_ms": cuda_timing["warm_stage1_ms"],
            "stage2_ms": cuda_timing["warm_stage2_ms"],
            "total_resident_ms": cuda_timing["warm_total_ms"],
            "fps": cuda_fps,
            "mpixels_per_sec": cuda_mpix,
            "peak_vram_mb": cuda_peak_vram,
            "peak_host_rss_mb": cuda_peak_rss,
            "speedup_vs_torch": torch_warm_total_ms / cuda_timing["warm_total_ms"],
            "alpha_parity": {"mean_abs": float(cuda_alpha_diff.mean()), "max_abs": float(cuda_alpha_diff.max())},
            "fg_parity": {"mean_abs": float(cuda_fg_diff.mean()), "max_abs": float(cuda_fg_diff.max())},
        },
    }
    if wgpu_timing is not None:
        report_data["rust_wgpu"] = {
            "load_ms": wgpu_timing["load_ms"],
            "cold_ms": wgpu_timing["cold_ms"],
            "stage1_ms": wgpu_timing["warm_stage1_ms"],
            "stage2_ms": wgpu_timing["warm_stage2_ms"],
            "total_resident_ms": wgpu_timing["warm_total_ms"],
            "fps": wgpu_fps,
            "mpixels_per_sec": wgpu_mpix,
            "peak_vram_mb": wgpu_peak_vram,
            "peak_host_rss_mb": wgpu_peak_rss,
            "speedup_vs_torch": torch_warm_total_ms / wgpu_timing["warm_total_ms"],
            "alpha_parity": {"mean_abs": float(wgpu_alpha_diff.mean()), "max_abs": float(wgpu_alpha_diff.max())},
        }

    (out_dir / "benchmark_report.json").write_text(json.dumps(report_data, indent=2))

    # Clean staging
    shutil.rmtree(staging_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # 6. Generate Markdown Summary
    # -------------------------------------------------------------------------
    md_lines = [
        "# CorridorKey 2.0: Benchmark Report",
        "",
        f"- **Date**: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- **Image Resolution**: `{w_orig}x{h_orig}` (padded to `{W_pad}x{H_pad}`)",
        f"- **Compute Device**: `{device}` (`{torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}`)",
        "",
        "## Performance & Latency Comparison",
        "",
        "| Runtime | Stage 1 | Stage 2 | Total Warm | FPS | Throughput | Peak VRAM | Host RSS | Speedup vs PyTorch | Alpha Max Error |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **PyTorch CUDA (Ref)** | {torch_s1_ms:.2f} ms | {torch_s2_ms:.2f} ms | {torch_warm_total_ms:.2f} ms | {torch_fps:.1f} | {torch_mpix:.2f} MP/s | {torch_vram_alloc_mb:.1f} MB | {torch_rss_mb:.1f} MB | 1.00x | Reference |",
        f"| **Rust CUDA (CubeCL)** | {cuda_timing['warm_stage1_ms']:.2f} ms | {cuda_timing['warm_stage2_ms']:.2f} ms | {cuda_timing['warm_total_ms']:.2f} ms | {cuda_fps:.1f} | {cuda_mpix:.2f} MP/s | {cuda_peak_vram:.1f} MB | {cuda_peak_rss:.1f} MB | **{torch_warm_total_ms/cuda_timing['warm_total_ms']:.2f}x** | `{cuda_alpha_diff.max():.2e}` |",
    ]
    if wgpu_timing is not None:
        md_lines.append(
            f"| **Rust WebGPU (CubeCL)** | {wgpu_timing['warm_stage1_ms']:.2f} ms | {wgpu_timing['warm_stage2_ms']:.2f} ms | {wgpu_timing['warm_total_ms']:.2f} ms | {wgpu_fps:.1f} | {wgpu_mpix:.2f} MP/s | {wgpu_peak_vram:.1f} MB | {wgpu_peak_rss:.1f} MB | **{torch_warm_total_ms/wgpu_timing['warm_total_ms']:.2f}x** | `{wgpu_alpha_diff.max():.2e}` |"
        )
    md_lines.extend([
        "",
        "## Parity Statistics (vs PyTorch FP32 Reference)",
        "",
        "- **Rust CUDA Alpha Error**:",
        f"  - Mean Absolute Error: `{cuda_alpha_diff.mean():.4e}`",
        f"  - Max Absolute Error:  `{cuda_alpha_diff.max():.4e}`",
        f"  - RMSE:                `{np.sqrt((cuda_alpha_diff**2).mean()):.4e}`",
        "- **Rust CUDA Foreground Error**:",
        f"  - Mean Absolute Error: `{cuda_fg_diff.mean():.4e}`",
        f"  - Max Absolute Error:  `{cuda_fg_diff.max():.4e}`",
    ])
    if wgpu_timing is not None:
        md_lines.extend([
            "- **Rust WebGPU Alpha Error**:",
            f"  - Mean Absolute Error: `{wgpu_alpha_diff.mean():.4e}`",
            f"  - Max Absolute Error:  `{wgpu_alpha_diff.max():.4e}`",
            f"  - RMSE:                `{np.sqrt((wgpu_alpha_diff**2).mean()):.4e}`",
        ])
    (out_dir / "benchmark_summary.md").write_text("\n".join(md_lines) + "\n")

    # -------------------------------------------------------------------------
    # 7. Print Summary Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("                              BENCHMARK SUMMARY RESULTS                              ")
    print("=" * 96)
    print(f"{'Runtime':<22} | {'Stage 1':<9} | {'Stage 2':<9} | {'Total Warm':<10} | {'FPS':<6} | {'VRAM':<8} | {'RSS':<8} | {'vs Torch':<8} | {'Alpha Max Err'}")
    print("-" * 96)
    print(f"{'PyTorch CUDA (Ref)':<22} | {torch_s1_ms:6.2f} ms | {torch_s2_ms:6.2f} ms | {torch_warm_total_ms:7.2f} ms | {torch_fps:5.1f} | {torch_vram_alloc_mb:6.1f} MB | {torch_rss_mb:6.1f} MB | {'1.00x':<8} | {'Reference':<12}")
    print(f"{'Rust CUDA (CubeCL)':<22} | {cuda_timing['warm_stage1_ms']:6.2f} ms | {cuda_timing['warm_stage2_ms']:6.2f} ms | {cuda_timing['warm_total_ms']:7.2f} ms | {cuda_fps:5.1f} | {cuda_peak_vram:6.1f} MB | {cuda_peak_rss:6.1f} MB | {torch_warm_total_ms/cuda_timing['warm_total_ms']:6.2f}x | {cuda_alpha_diff.max():.2e}")
    if wgpu_timing is not None:
        print(f"{'Rust WGPU (CubeCL)':<22} | {wgpu_timing['warm_stage1_ms']:6.2f} ms | {wgpu_timing['warm_stage2_ms']:6.2f} ms | {wgpu_timing['warm_total_ms']:7.2f} ms | {wgpu_fps:5.1f} | {wgpu_peak_vram:6.1f} MB | {wgpu_peak_rss:6.1f} MB | {torch_warm_total_ms/wgpu_timing['warm_total_ms']:6.2f}x | {wgpu_alpha_diff.max():.2e}")
    print("=" * 96)
    print(f"Results saved to: {out_dir}")
    print(f"Comparison Grid:  {grid_path}")
    print(f"Markdown Summary: {out_dir / 'benchmark_summary.md'}")


if __name__ == "__main__":
    main()
