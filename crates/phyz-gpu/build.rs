//! Builds `cuda/phyz_kernels.cu` as host C++ when the `cuda-host` feature is
//! on. The `cuda` feature needs nothing here: NVRTC compiles the same file at
//! runtime, and `cudarc` dlopens the driver.

fn main() {
    println!("cargo:rerun-if-changed=cuda/phyz_kernels.cu");
    println!("cargo:rerun-if-changed=cuda/phyz_train.cu");
    #[cfg(feature = "cuda-host")]
    {
        cc::Build::new()
            .cpp(true)
            .file("cuda/phyz_kernels.cu")
            .file("cuda/phyz_train.cu")
            .flag("-xc++")
            // Bit-identity tests compare fused vs unfused kernels; GCC defaults
            // to -ffp-contract=fast for GNU C++, so FMA contraction can differ
            // per kernel when the target has FMA. Keep rounding strict.
            .flag_if_supported("-ffp-contract=off")
            .std("c++14")
            .opt_level(2)
            .warnings(true)
            .compile("phyz_cuda_host_kernels");
    }
}
