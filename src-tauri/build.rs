use std::process::Command;

fn main() {
    let build_id = std::env::var("MUSIC_LOOPER_BUILD_ID")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            Command::new("git")
                .args(["rev-parse", "--short", "HEAD"])
                .output()
                .ok()
                .filter(|output| output.status.success())
                .and_then(|output| String::from_utf8(output.stdout).ok())
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        })
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=MUSIC_LOOPER_BUILD_ID={build_id}");
    println!("cargo:rerun-if-env-changed=MUSIC_LOOPER_BUILD_ID");
    println!("cargo:rerun-if-changed=../.git/HEAD");
    println!("cargo:rerun-if-changed=../.git/refs");

    tauri_build::build()
}
