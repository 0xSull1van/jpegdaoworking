fn main() {
    // CUDA kernel build is wired in Phase 4 (gated on cuda-runtime feature).
    println!("cargo:rerun-if-changed=build.rs");
}
