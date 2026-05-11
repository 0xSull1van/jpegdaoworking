use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernel/keccak_grinder.cu");
    println!("cargo:rerun-if-changed=kernel/keccak_device.cuh");
    println!("cargo:rerun-if-changed=kernel/result_codec.cuh");

    // Only build CUDA when feature is enabled.
    if std::env::var("CARGO_FEATURE_CUDA_RUNTIME").is_err() {
        return;
    }

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernel = manifest_dir.join("kernel/keccak_grinder.cu");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("keccak_grinder.ptx");

    let nvcc = which::which("nvcc").unwrap_or_else(|_| {
        panic!("nvcc not in PATH; install CUDA toolkit or build without --features cuda-runtime")
    });

    // PTX virtual arch. Default = compute_89 (Ada Lovelace, RTX 40-series).
    // Override via `CUDA_ARCH=compute_120 cargo build ...` for native Blackwell (RTX 5090).
    // PTX is forward-compatible: compute_89 PTX runs on Hopper/Blackwell via driver JIT,
    // but native arch gives ~5-10% better throughput.
    let arch = std::env::var("CUDA_ARCH").unwrap_or_else(|_| "compute_89".to_string());
    let arch_flag = format!("-arch={}", arch);
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");

    // Register cap — configurable via CUDA_MAXREG env var.
    //   CUDA_MAXREG=64       → tight, max occupancy, may spill
    //   CUDA_MAXREG=80       → previous default (kept available)
    //   CUDA_MAXREG=128      → loose, may reduce occupancy but no spill
    //   CUDA_MAXREG=auto     → no cap, let compiler decide (DEFAULT)
    //   (unset)              → auto (no cap)
    //
    // The named-state Keccak kernel needs ~100+ live 32-bit registers; capping
    // at 80 caused spills and degraded throughput. `auto` lets nvcc pick.
    let maxreg = std::env::var("CUDA_MAXREG").unwrap_or_else(|_| "auto".to_string());
    println!("cargo:rerun-if-env-changed=CUDA_MAXREG");

    let mut args: Vec<String> = vec![
        "-ptx".into(),
        "-O3".into(),
        arch_flag,
        "-Xptxas".into(), "-O3".into(),
        "-Xptxas".into(), "-v".into(),
        "--use_fast_math".into(),
    ];
    if maxreg != "auto" {
        args.push(format!("--maxrregcount={}", maxreg));
    }

    let output = Command::new(&nvcc)
        .args(&args)
        .arg("-o")
        .arg(&out)
        .arg(&kernel)
        .output()
        .expect("failed to invoke nvcc");

    // Forward nvcc stderr (which carries -Xptxas -v register/spill info) as cargo warnings
    // so they're visible during `cargo build` without needing -vv.
    for line in String::from_utf8_lossy(&output.stderr).lines() {
        if !line.trim().is_empty() {
            println!("cargo:warning=nvcc: {}", line);
        }
    }
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if !line.trim().is_empty() {
            println!("cargo:warning=nvcc: {}", line);
        }
    }

    if !output.status.success() {
        panic!("nvcc exited with {}", output.status);
    }

    println!("cargo:rustc-env=PTX_PATH={}", out.display());
}
