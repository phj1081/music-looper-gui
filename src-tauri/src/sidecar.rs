use serde_json::Value;
use std::env;
use std::path::PathBuf;
use std::sync::Mutex;
use tauri::AppHandle;
use tauri::Manager;
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

/// The only state Rust needs: the port the Python HTTP server is listening on.
pub struct ServerState {
    pub port: Mutex<Option<u16>>,
    pub startup_error: Mutex<Option<String>>,
    pub child: Mutex<Option<CommandChild>>,
}

impl Default for ServerState {
    fn default() -> Self {
        Self {
            port: Mutex::new(None),
            startup_error: Mutex::new(None),
            child: Mutex::new(None),
        }
    }
}

fn build_sidecar_path_env() -> Option<String> {
    fn push_unique(paths: &mut Vec<PathBuf>, path: PathBuf) {
        if path.as_os_str().is_empty() {
            return;
        }
        if !paths.iter().any(|candidate| candidate == &path) {
            paths.push(path);
        }
    }

    let mut paths: Vec<PathBuf> = Vec::new();

    if let Some(existing_path) = env::var_os("PATH") {
        for path in env::split_paths(&existing_path) {
            push_unique(&mut paths, path);
        }
    }

    #[cfg(target_os = "macos")]
    for candidate in [
        "/opt/homebrew/bin",
        "/usr/local/bin",
        "/usr/bin",
        "/bin",
        "/usr/sbin",
        "/sbin",
    ] {
        push_unique(&mut paths, PathBuf::from(candidate));
    }

    #[cfg(target_os = "linux")]
    for candidate in [
        "/usr/local/bin",
        "/usr/bin",
        "/bin",
        "/usr/sbin",
        "/sbin",
    ] {
        push_unique(&mut paths, PathBuf::from(candidate));
    }

    env::join_paths(paths)
        .ok()
        .map(|value| value.to_string_lossy().into_owned())
}

/// Spawn the Python sidecar and read the port it prints to stdout.
pub async fn spawn_server(app: &AppHandle) -> Result<(), String> {
    let mut sidecar_command = app
        .shell()
        .sidecar("music-looper-sidecar")
        .map_err(|e| format!("Failed to create sidecar command: {}", e))?;

    if let Some(path_env) = build_sidecar_path_env() {
        sidecar_command = sidecar_command.env("PATH", path_env);
    }

    let (mut rx, child) = sidecar_command
        .spawn()
        .map_err(|e| format!("Failed to spawn sidecar: {}", e))?;

    // Read the first stdout line: {"port": N}
    let port: u16 = loop {
        match rx.recv().await {
            Some(CommandEvent::Stdout(line_bytes)) => {
                let line = String::from_utf8_lossy(&line_bytes);
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                match serde_json::from_str::<Value>(line) {
                    Ok(val) => {
                        if let Some(p) = val.get("port").and_then(|v| v.as_u64()) {
                            break p as u16;
                        }
                    }
                    Err(e) => {
                        log::warn!("Ignoring non-JSON sidecar output: {} ({})", line, e);
                    }
                }
            }
            Some(CommandEvent::Stderr(line_bytes)) => {
                let line = String::from_utf8_lossy(&line_bytes);
                log::warn!("Sidecar stderr: {}", line.trim());
            }
            Some(CommandEvent::Error(err)) => {
                return Err(format!("Sidecar error: {}", err));
            }
            Some(CommandEvent::Terminated(status)) => {
                return Err(format!("Sidecar terminated early: {:?}", status));
            }
            None => {
                return Err("Sidecar channel closed before port was received".to_string());
            }
            _ => {}
        }
    };

    log::info!("Sidecar HTTP server on port {}", port);
    let state = app.state::<ServerState>();
    if let Ok(mut stored_port) = state.port.lock() {
        *stored_port = Some(port);
    }
    if let Ok(mut startup_error) = state.startup_error.lock() {
        *startup_error = None;
    }
    if let Ok(mut child_guard) = state.child.lock() {
        *child_guard = Some(child);
    }
    Ok(())
}

/// Kill the sidecar process (called on app exit).
pub fn kill_sidecar(state: &ServerState) {
    if let Ok(mut guard) = state.child.lock() {
        if let Some(child) = guard.take() {
            let _ = child.kill();
            log::info!("Sidecar process killed");
        }
    }
}
