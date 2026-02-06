fn main() {
    println!("cargo:rerun-if-changed=cuda/stc_gpu.cu");

    // Check for nvcc
    let nvcc_check = std::process::Command::new("nvcc")
        .arg("--version")
        .output();

    if let Ok(output) = nvcc_check {
        if output.status.success() {
            // Found nvcc, compile CUDA
            cc::Build::new()
                .cuda(true)
                .flag("-cudart=shared")
                .flag("-gencode").flag("arch=compute_75,code=sm_75") // T4/RTX 20xx+
                .file("cuda/stc_gpu.cu")
                .file("cuda/fri_gpu.cu")
                .compile("stc_gpu");

            println!("cargo:rustc-cfg=feature=\"gpu\"");
            println!("cargo:rustc-link-lib=cudart");
        } else {
            println!("cargo:warning=nvcc found but returned error. GPU support disabled.");
        }
    } else {
        println!("cargo:warning=nvcc not found. GPU support disabled.");
    }
}
