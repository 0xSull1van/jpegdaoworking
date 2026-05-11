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

    let output = Command::new(&nvcc)
        .args([
            "-ptx",
            "-O3",
            "-arch=compute_89",       // Ada Lovelace (RTX 40-series)
            "-Xptxas", "-O3",         // aggressive backend (PTX→SASS) optimization
            "-Xptxas", "-v",          // log register/spill counts for tuning visibility
            "--use_fast_math",        // FP optimizations
            "--maxrregcount=80",      // cap registers so more blocks fit per SM
        ])
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
