use serde_json::Value;
use std::sync::Mutex;
use tauri::AppHandle;
use tauri::Manager;
use tauri_plugin_shell::process::CommandEvent;
use tauri_plugin_shell::ShellExt;

/// The only state Rust needs: the port the Python HTTP server is listening on.
pub struct ServerState {
    pub port: Mutex<Option<u16>>,
    pub startup_error: Mutex<Option<String>>,
}

impl Default for ServerState {
    fn default() -> Self {
        Self {
            port: Mutex::new(None),
            startup_error: Mutex::new(None),
        }
    }
}

/// Spawn the Python sidecar and read the port it prints to stdout.
pub async fn spawn_server(app: &AppHandle) -> Result<(), String> {
    let sidecar_command = app
        .shell()
        .sidecar("music-looper-sidecar")
        .map_err(|e| format!("Failed to create sidecar command: {}", e))?;

    let (mut rx, _child) = sidecar_command
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
    Ok(())
}
