use std::{fs, path::Path, time::Instant};

use corridorkey_core::ComputeTarget;
use corridorkey_ml::{MlContext, SafetensorsFile, Stage1, Stage2};
use serde::Deserialize;

#[derive(Deserialize)]
struct RunMeta {
    b: usize,
    t: usize,
    h: usize,
    w: usize,
    vfm_channels: usize,
}

fn read_f32_file(path: &Path) -> Vec<f32> {
    let bytes =
        fs::read(path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: run_stage1_stage2 <in_dir> <out_dir>");
        std::process::exit(1);
    }
    let in_dir = Path::new(&args[1]);
    let out_dir = Path::new(&args[2]);
    fs::create_dir_all(out_dir).unwrap();

    let meta: RunMeta =
        serde_json::from_str(&fs::read_to_string(in_dir.join("meta.json")).unwrap()).unwrap();
    let (b, t, h, w) = (meta.b, meta.t, meta.h, meta.w);
    let bt = b * t;
    let h64 = h / 64;
    let w64 = w / 64;
    let h16 = h / 16;
    let w16 = w / 16;

    let rgb_hint = read_f32_file(&in_dir.join("rgb_hint_bthwc.f32"));
    let rgb = read_f32_file(&in_dir.join("rgb_bthwc.f32"));
    let vfm_64 = read_f32_file(&in_dir.join("vfm_64_bthwc.f32"));
    let hq = read_f32_file(&in_dir.join("hint_quality_bt.f32"));

    let ckpt_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../weights/corridorkey_v2.safetensors");
    let st = SafetensorsFile::open(&ckpt_path).unwrap();
    let header = st.checkpoint_header().unwrap();

    // Prefer CUDA, fallback to WGPU
    let target = if ComputeTarget::Cuda.is_available() {
        ComputeTarget::Cuda
    } else {
        ComputeTarget::Wgpu
    };
    println!("Running CorridorKey Rust inference on target: {target:?}");

    let ctx = MlContext::new(target).unwrap();
    let t0 = Instant::now();
    let stage1 = Stage1::load(&ctx, &st, &header.model_config, "stage1").unwrap();
    let stage2 = Stage2::load(&ctx, &st, &header.model_config, "stage2").unwrap();
    println!("Loaded models in {:.2} ms", t0.elapsed().as_secs_f64() * 1e3);

    let t_rgb_hint = ctx.upload_tensor(&rgb_hint, [b, t, h, w, 4]).unwrap();
    let t_rgb = ctx.upload_tensor(&rgb, [bt, h, w, 3]).unwrap();
    let t_vfm = ctx
        .upload_tensor(&vfm_64, [b, t, h64, w64, meta.vfm_channels])
        .unwrap();
    let t_hq = ctx.upload_tensor(&hq, [bt, 1]).unwrap();

    let t_run = Instant::now();
    // 1. Stage 1 -> Baton
    let baton = stage1
        .forward_baton::<f32>(&ctx, t_rgb_hint, t_vfm, Some(t_hq), 1.0)
        .unwrap();
    let baton_nhwc = baton.reshape([bt, h16, w16, stage1.d_16]);

    // 2. Stage 2 -> Logits [BT, H, W, 4]
    let logits = stage2.forward::<f32>(&ctx, t_rgb, baton_nhwc).unwrap();

    let dl_logits = ctx.download_tensor::<f32, 4>(logits).unwrap();
    let run_ms = t_run.elapsed().as_secs_f64() * 1e3;
    println!("Stage 1 + Stage 2 resident execution: {:.2} ms", run_ms);

    let raw_bytes: &[u8] = bytemuck::cast_slice(&dl_logits);
    fs::write(out_dir.join("rust_logits_bthwc.f32"), raw_bytes).unwrap();
    println!("Wrote rust_logits_bthwc.f32 ({} elements)", dl_logits.len());
}
