//! Run metadata: hardware, OS, toolchain, and library versions.
//!
//! Every emitted result carries this block. A benchmark number without the
//! machine it was measured on is not a result, it is a rumour.

use serde::{Deserialize, Serialize};
use std::process::Command;

/// Everything needed to reproduce (or at least contextualise) a run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMeta {
    /// ISO-8601 UTC timestamp, or `"unknown"` if `date` is unavailable.
    pub timestamp: String,
    pub hardware: Hardware,
    pub os: Os,
    pub toolchain: Toolchain,
    /// Versions of every library that produced a number in this run.
    pub libraries: Vec<Library>,
    /// Git commit of the phyz tree, when available.
    pub git_commit: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hardware {
    /// CPU brand string, e.g. `"Apple M4 Max"`.
    pub cpu: String,
    /// Logical core count.
    pub cores: usize,
    /// Physical RAM in GiB, rounded.
    pub memory_gib: Option<u64>,
    /// GPU adapter name, filled in by the GPU suite when it runs.
    pub gpu: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Os {
    /// `std::env::consts::OS`.
    pub family: String,
    /// `uname -sr` or equivalent.
    pub release: String,
    /// `std::env::consts::ARCH`.
    pub arch: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Toolchain {
    /// Full `rustc --version` string.
    pub rustc: String,
    /// Cargo profile the harness was compiled with.
    pub profile: String,
    /// Value of `RUSTFLAGS` at runtime, if set (target-cpu=native changes a lot).
    pub rustflags: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Library {
    pub name: String,
    pub version: String,
    /// Free-form note, e.g. `"f64 build"`.
    pub note: Option<String>,
}

fn run(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

fn cpu_brand() -> String {
    #[cfg(target_os = "macos")]
    {
        if let Some(s) = run("sysctl", &["-n", "machdep.cpu.brand_string"]) {
            return s;
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(txt) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in txt.lines() {
                if let Some((k, v)) = line.split_once(':')
                    && k.trim() == "model name"
                {
                    return v.trim().to_string();
                }
            }
        }
    }
    "unknown".to_string()
}

fn core_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0)
}

fn memory_gib() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        let bytes: u64 = run("sysctl", &["-n", "hw.memsize"])?.parse().ok()?;
        return Some(bytes / (1024 * 1024 * 1024));
    }
    #[cfg(target_os = "linux")]
    {
        let txt = std::fs::read_to_string("/proc/meminfo").ok()?;
        let line = txt.lines().find(|l| l.starts_with("MemTotal:"))?;
        let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        return Some(kib / (1024 * 1024));
    }
    #[allow(unreachable_code)]
    None
}

impl RunMeta {
    /// Collect metadata for the current machine and build.
    pub fn collect() -> Self {
        Self {
            timestamp: run("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"])
                .unwrap_or_else(|| "unknown".into()),
            hardware: Hardware {
                cpu: cpu_brand(),
                cores: core_count(),
                memory_gib: memory_gib(),
                gpu: None,
            },
            os: Os {
                family: std::env::consts::OS.to_string(),
                release: run("uname", &["-sr"]).unwrap_or_else(|| "unknown".into()),
                arch: std::env::consts::ARCH.to_string(),
            },
            toolchain: Toolchain {
                rustc: run("rustc", &["--version"]).unwrap_or_else(|| "unknown".into()),
                profile: if cfg!(debug_assertions) {
                    // Loud on purpose: a debug-profile number is meaningless.
                    "debug (INVALID FOR BENCHMARKING)".into()
                } else {
                    "release".into()
                },
                rustflags: std::env::var("RUSTFLAGS").ok().filter(|s| !s.is_empty()),
            },
            libraries: libraries(),
            git_commit: run("git", &["rev-parse", "HEAD"]),
        }
    }
}

fn libraries() -> Vec<Library> {
    // `mut` is only exercised when a comparison feature is on.
    #[allow(unused_mut)]
    let mut v = vec![Library {
        name: "phyz".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        note: Some("workspace version; f64 throughout".into()),
    }];
    #[cfg(feature = "rapier")]
    v.push(Library {
        name: "rapier3d-f64".into(),
        // Compile-time constant from the dependency itself, so it cannot drift
        // from what actually ran.
        version: rapier3d_f64::VERSION.into(),
        note: Some("f64 build, chosen to match phyz precision".into()),
    });
    #[cfg(feature = "gpu")]
    v.push(Library {
        name: "wgpu".into(),
        version: "23.0 (per phyz-gpu manifest)".into(),
        note: None,
    });
    #[cfg(feature = "gpu-cuda")]
    v.push(Library {
        name: "cudarc".into(),
        version: "0.19 (per phyz-gpu manifest, CUDA 12.8 API)".into(),
        note: Some("NVRTC-compiled kernels; driver dlopened at runtime".into()),
    });
    v
}
