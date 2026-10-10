// Compiles the CUDA kernel only with `--features gpu` (requires nvcc).
use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=gpu/vearl_gpu.cu");
    println!("cargo:rerun-if-changed=gpu/vearl_render2.cu");
    println!("cargo:rerun-if-changed=gpu/vearl_render3.cu");
    if env::var("CARGO_FEATURE_GPU").is_err() {
        return;
    }
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    // Windows (MSVC): nvcc needs cl.exe on the PATH (build from a "x64 Native Tools" prompt), no -fPIC, no ar, CUDA_PATH for the runtime library.
    let windows = env::var("CARGO_CFG_TARGET_OS").map_or(false, |o| o == "windows");
    println!("cargo:rerun-if-env-changed=VEARL_CUDA_ARCH");
    // Target architectures, comma separated. Default: a MULTI-ARCHITECTURE binary. Ampere (sm_80 A100, sm_86 RTX 30xx / A10), Ada (sm_89 RTX 40xx / L4),
    // Hopper (sm_90 H100) and, when nvcc is 12.8 or newer, Blackwell (sm_100 B200, sm_120 RTX 50xx), plus the PTX of the newest one (JIT-compiled on later GPUs).
    // Minimum sm_80: the kernel uses cp.async, ldmatrix and mma.m16n8k32 int8 (absent on Turing sm_75). VEARL_CUDA_ARCH=86: fast single-card build.
    let archs = env::var("VEARL_CUDA_ARCH").unwrap_or_else(|_| default_archs());
    let list: Vec<String> = archs.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    assert!(list.iter().all(|a| a.parse::<u32>().map_or(false, |v| v >= 80)), "VEARL_CUDA_ARCH: architectures >= 80 only (e.g. 80,86,89,90,120)");
    let sources = [("gpu/vearl_gpu.cu", "vearl_gpu.o"), ("gpu/vearl_render2.cu", "vearl_render2.o"), ("gpu/vearl_render3.cu", "vearl_render3.o")];
    let mut objs = vec![];
    for (src, name) in sources {
        let obj = out.join(name);
        let mut cmd = Command::new("nvcc");
        cmd.args(["-O3", "-std=c++17"]);
        if !windows {
            cmd.args(["-Xcompiler", "-fPIC"]);
        }
        for (k, a) in list.iter().enumerate() {
            cmd.arg("-gencode");
            if k + 1 == list.len() {
                cmd.arg(format!("arch=compute_{a},code=[sm_{a},compute_{a}]"));
            } else {
                cmd.arg(format!("arch=compute_{a},code=sm_{a}"));
            }
        }
        let st = cmd.args(["-c", src, "-o"]).arg(&obj).status().expect("nvcc not found");
        assert!(st.success(), "nvcc failed");
        objs.push(obj);
    }
    let lib = out.join(if windows { "vearl_gpu.lib" } else { "libvearl_gpu.a" });
    let st = if windows {
        Command::new("nvcc").arg("-lib").arg("-o").arg(&lib).args(&objs).status().expect("nvcc not found")
    } else {
        Command::new("ar").arg("crs").arg(&lib).args(&objs).status().expect("ar not found")
    };
    assert!(st.success());
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=vearl_gpu");
    if windows {
        let cuda = env::var("CUDA_PATH").expect("CUDA_PATH is not set: install the CUDA Toolkit");
        println!("cargo:rustc-link-search=native={cuda}/lib/x64");
        println!("cargo:rustc-link-lib=cudart_static");
        return;
    }
    // Linux: the CUDA runtime is linked STATICALLY when libcudart_static.a is found (CUDA toolkit of the build machine), so the binary needs
    // only the NVIDIA driver (libcuda.so.1) on the user's machine, like the Windows build. VEARL_CUDART=dynamic forces libcudart.so.
    println!("cargo:rerun-if-env-changed=VEARL_CUDART");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    let mut dirs: Vec<String> = vec!["/usr/lib/x86_64-linux-gnu".into(), "/usr/local/cuda/lib64".into(), "/usr/local/cuda/targets/x86_64-linux/lib".into()];
    if let Ok(c) = env::var("CUDA_PATH") {
        dirs.push(format!("{c}/lib64"));
        dirs.push(format!("{c}/targets/x86_64-linux/lib"));
    }
    for d in &dirs {
        println!("cargo:rustc-link-search=native={d}");
    }
    let want_static = env::var("VEARL_CUDART").map_or(true, |v| v != "dynamic");
    let have_static = dirs.iter().any(|d| std::path::Path::new(d).join("libcudart_static.a").exists());
    if want_static && have_static {
        println!("cargo:rustc-link-lib=static=cudart_static");
        println!("cargo:rustc-link-lib=dylib=dl");
        println!("cargo:rustc-link-lib=dylib=rt");
        println!("cargo:rustc-link-lib=dylib=pthread");
    } else {
        println!("cargo:rustc-link-lib=cudart");
    }
    // static libstdc++ when available (the executable then does not depend on the libstdc++ version of the machine)
    let a = Command::new("g++").arg("-print-file-name=libstdc++.a").output().ok().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    match a.as_deref().map(std::path::Path::new) {
        Some(p) if p.is_absolute() && p.exists() => {
            println!("cargo:rustc-link-search=native={}", p.parent().unwrap().display());
            println!("cargo:rustc-link-lib=static=stdc++");
        }
        _ => println!("cargo:rustc-link-lib=stdc++"),
    }
}

/// `nvcc --version` reports "release 12.9, V12.9.86": the (major, minor) pair, or None when nvcc cannot be run.
fn nvcc_release() -> Option<(u32, u32)> {
    let out = Command::new("nvcc").arg("--version").output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let rest = text.split("release ").nth(1)?;
    let mut it = rest.split(|c: char| c == '.' || c == ',').map(|x| x.trim().parse::<u32>());
    Some((it.next()?.ok()?, it.next()?.ok()?))
}

/// Blackwell (sm_100, sm_120) needs CUDA 12.8; with an older toolkit the default list stops at Hopper.
fn default_archs() -> String {
    match nvcc_release() {
        Some((major, minor)) if (major, minor) >= (12, 8) => "80,86,89,90,100,120".into(),
        _ => "80,86,89,90".into(),
    }
}
